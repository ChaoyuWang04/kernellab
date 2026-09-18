"""算子的两层文件:

    kernels/<名>/        用户写的:算子源码(kernel.py / kernel.cu / 任何 DSL),harness 不动它
    specs/<名>/          agent 写的接线:meta.toml(工具链、case、容差、扫参)+ spec.py(四个契约函数)

spec.py 必须提供(与 DSL 无关):
    make_inputs(case: dict, device) -> dict[str, Tensor]   按 case 造输入(固定种子)
    run(**inputs) -> Tensor                                 调用用户的算子
    reference(**inputs) -> Tensor                           torch 参考实现;bench 也会给它计时,作为「相对 torch」的标尺
    workload(case: dict, **inputs) -> {"flops": int, "bytes": int}
可选:
    configure(**params)   扫参用:把 meta.toml [sweep] 的一组参数应用到算子

spec.py 里拿用户算子:`k = kernel_module(__file__)` 会按目录名导入 kernels/<名>/kernel.py;
cuda / tk 用 `load_extension(...)` 编译 kernels/<名>/ 下的 .cu。
"""
from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType


def repo_root_of(spec_dir: Path) -> Path:
    return spec_dir.resolve().parent.parent


def kernel_dir_of(spec_dir: Path) -> Path:
    return repo_root_of(spec_dir) / "kernels" / spec_dir.name


def kernel_module(spec_file: str, filename: str = "kernel.py") -> ModuleType:
    """给 spec.py 用:导入同名 kernels/<名>/<filename>,并把该目录放进 sys.path(用户的算子可以拆多文件)。"""
    spec_dir = Path(spec_file).resolve().parent
    kdir = kernel_dir_of(spec_dir)
    path = kdir / filename
    if not path.exists():
        raise SystemExit(f"用户算子文件不存在:{path}")
    if str(kdir) not in sys.path:
        sys.path.insert(0, str(kdir))
    name = f"userkernel.{spec_dir.name}.{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    ispec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(ispec)
    sys.modules[name] = mod
    ispec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@dataclass
class KernelSpec:
    dir: Path            # specs/<名>
    kernel_dir: Path     # kernels/<名>
    name: str
    toolchain: str
    kernel_regex: str
    min_cc: float
    features: list[str]
    atol: float
    rtol: float
    cases: list[dict] = field(default_factory=list)
    sweep: dict[str, list] = field(default_factory=dict)

    @classmethod
    def load(cls, spec_dir: Path) -> "KernelSpec":
        spec_dir = Path(spec_dir)
        meta = tomllib.loads((spec_dir / "meta.toml").read_text())
        req = meta.get("requires", {})
        tol = meta.get("tolerance", {})
        return cls(
            dir=spec_dir,
            kernel_dir=kernel_dir_of(spec_dir),
            name=meta.get("name", spec_dir.name),
            toolchain=meta["toolchain"],
            kernel_regex=meta.get("kernel_regex", ""),
            min_cc=float(req.get("min_cc", 0)),
            features=list(req.get("features", [])),
            atol=float(tol.get("atol", 1e-5)),
            rtol=float(tol.get("rtol", 1e-5)),
            cases=list(meta.get("cases", [])),
            sweep={k: list(v) for k, v in meta.get("sweep", {}).items()},
        )

    def load_module(self) -> ModuleType:
        path = self.dir / "spec.py"
        if not path.exists():
            raise SystemExit(f"{self.dir} 缺 spec.py(接线文件由 agent 按 docs/01-AGENT-PLAYBOOK.md 生成)")
        ispec = importlib.util.spec_from_file_location(f"specs.{self.name}", path)
        mod = importlib.util.module_from_spec(ispec)
        ispec.loader.exec_module(mod)  # type: ignore[union-attr]
        for fn in ("make_inputs", "run", "reference", "workload"):
            if not hasattr(mod, fn):
                raise SystemExit(f"{path} 缺少函数 {fn}()")
        return mod

    def select_cases(self, names: list[str] | None) -> list[dict]:
        if not names:
            return self.cases
        by = {c["name"]: c for c in self.cases}
        missing = [n for n in names if n not in by]
        if missing:
            raise SystemExit(f"meta.toml 里没有 case {missing};可选:{list(by)}")
        return [by[n] for n in names]
