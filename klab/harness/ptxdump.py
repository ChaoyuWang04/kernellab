"""从后端的编译缓存里取 PTX,按「哪一代的指令」分类计数。

只回答一个问题:我写的这份 kernel,到底降到了哪一代的 tensor core 与异步拷贝指令?
体检单答不了它 —— NCU 说 Tensor pipe 用了百分之多少,不说走的是 mma.sync 还是 wgmma。

在后端运行,只依赖 torch 与标准库(triton 等在函数里按需导入)。
每种工具链一个取法,登记在 _COLLECTORS;没接的工具链直接报清楚,不猜。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# (助记符前缀, 最低架构, 是否世代标志, 说明)。匹配按前缀从长到短,否则 cp.async.bulk 会被 cp.async 吃掉。
# 「世代标志」才参与判定:ldmatrix / stmatrix / mbarrier 这些是辅助指令,某一代起就有,
# 出现了并不说明你用上了那一代的核心能力(5090 是 sm_120,照样会发 stmatrix)。
PTX_FAMILIES: list[tuple[str, str, bool, str]] = [
    ("wgmma",         "sm_90",  True,  "warpgroup 异步 MMA —— Hopper 的标志指令"),
    ("tcgen05",       "sm_100", True,  "第五代 tensor core / tmem —— 数据中心 Blackwell 的标志指令"),
    ("cp.async.bulk", "sm_90",  True,  "TMA 批量异步拷贝 —— Hopper 的标志指令"),
    ("mma.sync",      "sm_80",  True,  "同步 tensor core MMA —— Ampere 的标志指令(手写 PTX / Triton 的 tl.dot 走这条)"),
    ("wmma.mma",      "sm_80",  True,  "WMMA 的矩阵乘 —— nvcuda::wmma 降下来的样子,同样是 tensor core(bf16 要 sm_80)"),
    ("wmma.load",     "sm_70",  False, "WMMA 装载 fragment"),
    ("wmma.store",    "sm_70",  False, "WMMA 写回 fragment"),
    ("cp.async",      "sm_80",  True,  "global -> shared 异步拷贝(多级流水)—— Ampere 的标志指令"),
    ("ldmatrix",      "sm_75",  False, "shared -> 寄存器的 tile 装载"),
    ("stmatrix",      "sm_90",  False, "寄存器 -> shared 的 tile 回写"),
    ("mbarrier",      "sm_80",  False, "异步屏障"),
    ("setmaxnreg",    "sm_90",  False, "warp specialization 的寄存器再分配"),
    ("fma",           "-",      False, "CUDA core 上的标量/向量 FMA"),
    ("bar.sync",      "-",      False, "__syncthreads"),
    ("ld.global",     "-",      False, "全局读"),
    ("st.global",     "-",      False, "全局写"),
    ("ld.shared",     "-",      False, "共享内存读"),
    ("st.shared",     "-",      False, "共享内存写"),
]

# 标志指令 -> 它代表的世代。只有这张表参与「命中世代」的判定。
MARKER_GENERATION = {
    "mma.sync": "Ampere",
    "wmma.mma": "Ampere",
    "cp.async": "Ampere",
    "wgmma": "Hopper",
    "cp.async.bulk": "Hopper",
    "tcgen05": "Blackwell",
}

# 标志指令无论有没有都要显示:看不到它们本身就是结论(5090 上 wgmma 必然是 0)
ALWAYS_SHOW = tuple(MARKER_GENERATION)

_FAMILIES_LONGEST_FIRST = sorted(PTX_FAMILIES, key=lambda f: -len(f[0]))


def mnemonics(ptx: str):
    """逐条产出 PTX 指令的助记符。跳过注释、.directive、标签;`@%p1 cp.async ...` 这种谓词指令剥掉谓词。"""
    for line in ptx.splitlines():
        s = line.strip()
        if not s or s.startswith(("//", ".", "{", "}", "$")):
            continue
        if s.startswith("@"):
            parts = s.split(None, 1)
            if len(parts) < 2:
                continue
            s = parts[1].lstrip()
        tok = s.split()[0].rstrip(";,")
        if tok:
            yield tok


def count(ptx: str) -> dict[str, int]:
    """助记符 -> 指令族计数。族名即 PTX_FAMILIES 里的前缀。"""
    counts: dict[str, int] = {}
    for m in mnemonics(ptx):
        for prefix, _arch, _marker, _desc in _FAMILIES_LONGEST_FIRST:
            if m == prefix or m.startswith(prefix + "."):
                counts[prefix] = counts.get(prefix, 0) + 1
                break
    return counts


def generations(counts: dict[str, int]) -> list[str]:
    """出现过标志指令的世代,按 PTX_FAMILIES 的顺序去重。辅助指令不参与判定。"""
    seen: list[str] = []
    for prefix, _arch, marker, _desc in PTX_FAMILIES:
        gen = MARKER_GENERATION.get(prefix)
        if marker and gen and counts.get(prefix, 0) > 0 and gen not in seen:
            seen.append(gen)
    return seen


def _walk(obj, depth: int = 0):
    """triton 的编译缓存是 device -> (dict, ...) 的嵌套容器,层级各版本不同(3.8 是 device_caches,
    值还套了一层 tuple),所以按容器递归找叶子,不写死结构。"""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v, depth + 1)
    elif isinstance(obj, (tuple, list, set)):
        for v in obj:
            yield from _walk(v, depth + 1)
    elif obj is not None:
        yield obj


def _from_triton(spec, mod) -> dict[str, str]:
    """扫用户 kernel 模块里的 @triton.jit 函数,从它们的编译缓存取 PTX。"""
    from triton.runtime.jit import JITFunction

    out: dict[str, str] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("userkernel."):  # kernel_module() 用的前缀
            continue
        for attr, obj in vars(module).items():
            # @triton.autotune / @triton.heuristics 会把 JITFunction 包一层,真身在 .fn
            while not isinstance(obj, JITFunction) and hasattr(obj, "fn"):
                obj = obj.fn
            if not isinstance(obj, JITFunction):
                continue
            caches = [getattr(obj, a) for a in ("device_caches", "cache") if hasattr(obj, a)]
            compiled = [c for c in _walk(caches)
                        if isinstance(getattr(c, "asm", None), dict) and "ptx" in c.asm]
            for i, c in enumerate(compiled):
                out[attr if len(compiled) == 1 else f"{attr}#{i}"] = c.asm["ptx"]
    return out


def _from_cppext(spec, mod) -> dict[str, str]:
    """cuda / tk:cppext 把 .cu 编成 torch 扩展的 .so,用 cuobjdump 把 PTX 抠出来。

    编译产物在 ~/.cache/klab/<算子名>-<架构>/;按名字直接定位,不去猜 sys.modules
    里模块的 __file__(torch 的 cpp_extension 不保证把它设成 .so 路径)。
    """
    import subprocess

    # 精确定位:cppext 编译时把构建目录记在 LAST_BUILD 里。不要按 glob 猜 ——
    # 同一张卡上每份源码一个目录,靠时间戳挑会取到上一个答案的产物(缓存命中时不产生新文件)。
    from klab.harness import cppext

    build = cppext.LAST_BUILD.get(spec.name)
    sos = sorted(build.glob("*.so")) if build and build.is_dir() else []
    if not sos:
        raise SystemExit(
            f"没找到 {spec.name} 的编译产物(cppext.LAST_BUILD={build}):"
            "klab ptx 会自己跑一次算子触发编译,走到这里说明 spec.py 没用 load_extension"
        )
    out: dict[str, str] = {}
    for so in sos:
        r = subprocess.run(["cuobjdump", "-ptx", str(so)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"cuobjdump 失败({so.name}):{r.stderr.strip()[:200]}")
        if r.stdout.strip():
            out[so.stem] = r.stdout
    return out


def _objects(root, depth: int = 0):
    """从 spec 模块的全局变量往下找对象(编译产物通常被 spec 缓存在 dict 里)。"""
    if depth > 3:
        return
    if isinstance(root, dict):
        for v in root.values():
            yield from _objects(v, depth + 1)
    elif isinstance(root, (list, tuple, set)):
        for v in root:
            yield from _objects(v, depth + 1)
    else:
        yield root


def _from_tilelang(spec, mod) -> dict[str, str]:
    """TileLang:编译产物是 JITKernel,它自己就有 PTX(_get_ptx / export_ptx)。

    去 spec 模块的全局变量里找 —— 接线按形状缓存编译结果,缓存里装的就是它。
    """
    out: dict[str, str] = {}
    for i, obj in enumerate(_objects(vars(mod))):
        get = getattr(obj, "_get_ptx", None)
        if not callable(get):
            continue
        try:
            ptx = get()
        except Exception:
            continue
        if ptx and ptx.strip():
            out[spec.name if not out else f"{spec.name}#{i}"] = ptx
    return out


_CUTE_DUMP: Path | None = None   # prepare() 建的临时目录,_from_cute() 去里面捡 .ptx


def prepare(spec) -> None:
    """有些工具链必须在 import 之前就把「留下 PTX」打开 —— 目前只有 cute。

    CuTe DSL 的环境变量在 `import cutlass` 时就被读进 env_manager,之后再设没用。
    所以 runner 在 spec.load_module() 之前调这个函数。对其他工具链是空操作。
    """
    global _CUTE_DUMP
    if spec.toolchain != "cute":
        return
    import tempfile

    _CUTE_DUMP = Path(tempfile.mkdtemp(prefix="klab-cute-ptx-"))
    os.environ["CUTE_DSL_KEEP"] = "ptx"          # 还可以是 cubin / sass / ir
    os.environ["CUTE_DSL_DUMP_DIR"] = str(_CUTE_DUMP)


def _from_cute(spec, mod) -> dict[str, str]:
    """CuTe DSL:靠 CUTE_DSL_KEEP=ptx 让编译器把 PTX 留在 CUTE_DSL_DUMP_DIR 里。

    产物名字是「函数名 + 整个签名 + 架构」,长得没法看,这里截到第一个 _Tensor 之前。
    """
    if _CUTE_DUMP is None:
        raise SystemExit("cute 取 PTX 前必须先调 ptxdump.prepare(spec)(要在 import cutlass 之前)")
    files = sorted(_CUTE_DUMP.rglob("*.ptx"))
    if not files:
        raise SystemExit(
            f"{_CUTE_DUMP} 下没有 .ptx:确认 cute.compile 真的跑了一次"
            "(接线按形状缓存编译结果,进程内第一次调用才会编)"
        )
    out: dict[str, str] = {}
    for i, f in enumerate(files):
        name = f.name.split(".sm_")[0].split("_Tensor")[0] or spec.name
        out[name if len(files) == 1 else f"{name}#{i}"] = f.read_text()
    return out


# 一种工具链一个取法。
_COLLECTORS = {
    "triton": _from_triton,
    "tilelang": _from_tilelang,
    "cuda": _from_cppext,
    "tk": _from_cppext,
    "cute": _from_cute,
}


def collect(spec, mod) -> dict[str, str]:
    """{kernel 名: PTX 文本}。调用前必须已经跑过一次算子,否则编译缓存是空的。"""
    toolchain = spec.toolchain
    fn = _COLLECTORS.get(toolchain)
    if fn is None:
        raise SystemExit(
            f"klab ptx 还没接 {toolchain} 工具链(目前只有 {list(_COLLECTORS)});"
            "加法:在 klab/harness/ptxdump.py 的 _COLLECTORS 里补一个取 PTX 的函数"
        )
    asm = fn(spec, mod)
    if not asm:
        raise SystemExit(
            "编译缓存里没有 PTX:确认 kernel 真的被启动过一次,"
            "且 @triton.jit 函数定义在 kernels/<名>/ 的模块里(klab ptx 只扫用户模块)"
        )
    return asm


def dump(spec, mod, out_dir: Path) -> list[dict]:
    """取 PTX、写盘、计数。返回给 result.json 用的记录。"""
    d = out_dir / "ptx"
    d.mkdir(parents=True, exist_ok=True)
    records = []
    for name, ptx in collect(spec, mod).items():
        path = d / f"{name}.ptx"
        path.write_text(ptx)
        c = count(ptx)
        records.append({
            "kernel": name,
            "file": f"ptx/{path.name}",
            "ptx_lines": ptx.count("\n") + 1,
            "counts": c,
            "generations": generations(c),
        })
    return records
