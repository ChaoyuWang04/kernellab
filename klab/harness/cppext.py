"""在后端用 nvcc 把算子目录里的 .cu 编成 torch 扩展。cuda 与 tk 工具链共用。

- 编译缓存:~/.cache/klab/<算子名>-<架构>(SSH 后端持久;Modal 挂了 Volume 也持久)
- 架构:按当前设备的 cc 生成 -gencode,只编本卡
- tk=True:加 ThunderKittens 的 include 与 KITTENS_<ARCH> 宏,源码位置读环境变量 KLAB_TK_ROOT
"""
from __future__ import annotations

import os
from pathlib import Path


# 本进程最近一次编译用的构建目录,按算子名记。klab ptx 靠它精确定位 .so ——
# 按 glob + 时间戳去猜会取到别的答案的产物(缓存命中时不产生新文件,时间戳骗人)。
LAST_BUILD: dict[str, Path] = {}


def _arch() -> tuple[int, int]:
    import torch  # 只在后端有;本地 pytest 只用 tk_macro 这类纯函数

    return torch.cuda.get_device_capability(0)


def tk_macro(major: int, minor: int) -> str:
    """TK 的 common.mk 按 SM80 / SM90 / SM100 / SM103 / SM107 / SM120 分支,宏名 KITTENS_<那个>。"""
    cc = major * 10 + minor
    if cc >= 120:
        return "KITTENS_SM120"
    if cc >= 107:
        return "KITTENS_SM107"
    if cc >= 103:
        return "KITTENS_SM103"
    if cc >= 100:
        return "KITTENS_SM100"
    if cc >= 90:
        return "KITTENS_SM90"
    return "KITTENS_SM80"


def cutlass_includes() -> list[str]:
    """CUTLASS 的 C++ 头。随 pip 包 `nvidia-cutlass` 装进 site-packages,不用 git clone。

    裸 CUDA 到 Hopper 之后没有官方的手写封装(WMMA 只覆盖到 Ampere),NVIDIA 自己的
    答案就是用 CUTLASS —— 参考答案里「改 CUTLASS 的参数」那几级要这些头。
    """
    try:
        import cutlass_library
    except ImportError:
        raise SystemExit("没装 CUTLASS:在该后端跑一次 klab setup --toolchain cuda(envs/cuda 里有 nvidia-cutlass)")
    root = Path(cutlass_library.__file__).resolve().parent / "source"
    paths = [root / "include", root / "tools" / "util" / "include"]
    missing = [p for p in paths if not p.is_dir()]
    if missing:
        raise SystemExit(f"CUTLASS 头不完整,缺:{missing}")
    return [str(p) for p in paths]


def load_extension(name: str, spec_file: str, sources: list[str], tk: bool = False,
                   cutlass: bool = False,
                   extra_cuda_cflags: list[str] | None = None, spec_sources: list[str] | None = None):
    """从 specs/<名>/spec.py 调用:编译 kernels/<名>/ 下的 sources。

    spec_sources 是 specs/<名>/ 下的文件,一起编进来。torch / pybind 的绑定样板放这里,
    用户的 .cu 就只剩 __global__ kernel 与启动它的那几行,不必 include torch。
    """
    from klab.harness.spec import kernel_dir_of, override_file

    sdir = Path(spec_file).resolve().parent
    kdir = kernel_dir_of(sdir)
    over = override_file()
    if over:                       # --solution:第一个源文件换成参考答案那一份,其余(多文件算子)照旧
        sources = [str(over)] + list(sources[1:])
    major, minor = _arch()
    arch = f"sm_{major}{minor}"
    # TK 在 sm_90 及以后要求架构专属特性集(compute_90a / 120a 那种带 a 的),裸 CUDA 也一并用它,不损失什么
    # TK 与 CUTLASS 在 sm_90 及以后都要求架构专属特性集(compute_90a 那种带 a 的)
    suffix = "a" if ((tk or cutlass) and major >= 9) else ""
    cflags = ["-O3", "-std=c++20", "--use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
              # code 同时给 sm_(SASS)与 compute_(PTX):不嵌 PTX 的话 klab ptx 用 cuobjdump 什么也抠不出来
              f"-gencode=arch=compute_{major}{minor}{suffix},code=[sm_{major}{minor}{suffix},compute_{major}{minor}{suffix}]",
              "-lineinfo"]
    includes: list[str] = []
    if tk:
        root = os.path.expanduser(os.environ.get("KLAB_TK_ROOT") or str(kdir.parents[1] / "envs" / "tk" / "ThunderKittens"))  # 环境变量里可能带 ~
        if not Path(root, "include", "kittens.cuh").exists():
            raise SystemExit(f"找不到 ThunderKittens:{root}(先跑 klab setup --toolchain tk)")
        includes += [f"{root}/include", f"{root}/prototype"]
        cflags += [f"-D{tk_macro(major, minor)}", "-DNDEBUG", "-forward-unknown-to-host-compiler",
                   "-Xcompiler=-Wno-psabi", "-Xcompiler=-fno-strict-aliasing", "-ftemplate-backtrace-limit=0"]
    if cutlass:
        includes += cutlass_includes()
        cflags += ["-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "--expt-relaxed-constexpr"]
    cflags += extra_cuda_cflags or []
    # 构建目录带上源码内容的哈希。**不能只按名字+架构缓存**:Modal 用 add_local_dir 挂载仓库,
    # 文件 mtime 不随编辑变化,ninja 于是认为没改动、直接复用 Volume 上旧的 .so ——
    # 表现是「你改了代码,跑的还是上一版」,而且一声不吭。按内容哈希就不会再骗人。
    import hashlib

    src_paths = [kdir / s for s in sources] + [sdir / s for s in (spec_sources or [])]
    if over:
        src_paths[0] = over
    h = hashlib.sha256()
    for sp in src_paths:
        h.update(Path(sp).read_bytes())
    h.update(" ".join(cflags).encode())
    build = Path(os.path.expanduser(f"~/.cache/klab/{name}-{arch}-{h.hexdigest()[:8]}"))
    build.mkdir(parents=True, exist_ok=True)
    LAST_BUILD[name] = build

    from torch.utils.cpp_extension import load

    # TK 调 CUDA driver API(cuGetErrorString 等),要显式链 libcuda,否则导入时才报 undefined symbol
    ldflags = ["-lcuda"] if tk else []
    return load(
        name=name,
        extra_ldflags=ldflags,
        sources=[s if os.path.isabs(s) else str(kdir / s) for s in sources]
                + [str(sdir / s) for s in (spec_sources or [])],
        extra_cuda_cflags=cflags,
        extra_include_paths=includes,
        build_directory=str(build),
        verbose=False,
    )
