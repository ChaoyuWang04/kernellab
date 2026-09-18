"""算子单元契约:目录里一份 meta.toml + 一份 kernel.py。

kernel.py 必须提供四个函数(都是普通 Python,与 DSL 无关):
    make_inputs(case: dict, device) -> dict[str, Tensor]   按 case 造输入(用固定种子)
    run(**inputs) -> Tensor                                 调用你的算子
    reference(**inputs) -> Tensor                           torch 参考实现
    workload(case: dict, **inputs) -> {"flops": int, "bytes": int}   用来换算 TFLOPS / GB/s
"""
from __future__ import annotations

import importlib.util
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType


@dataclass
class KernelSpec:
    dir: Path
    name: str
    toolchain: str
    kernel_regex: str
    min_cc: float
    features: list[str]
    atol: float
    rtol: float
    cases: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, kernel_dir: Path) -> "KernelSpec":
        meta = tomllib.loads((kernel_dir / "meta.toml").read_text())
        req = meta.get("requires", {})
        tol = meta.get("tolerance", {})
        return cls(
            dir=kernel_dir,
            name=meta.get("name", kernel_dir.name),
            toolchain=meta["toolchain"],
            kernel_regex=meta.get("kernel_regex", ""),
            min_cc=float(req.get("min_cc", 0)),
            features=list(req.get("features", [])),
            atol=float(tol.get("atol", 1e-5)),
            rtol=float(tol.get("rtol", 1e-5)),
            cases=list(meta.get("cases", [])),
        )

    def load_module(self) -> ModuleType:
        path = self.dir / "kernel.py"
        spec = importlib.util.spec_from_file_location(f"kernels.{self.name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
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
