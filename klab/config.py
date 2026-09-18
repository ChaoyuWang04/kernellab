"""读取仓库根目录的 targets.toml,定位仓库根。"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """从 start 向上找 targets.toml 所在目录。"""
    p = (start or Path.cwd()).resolve()
    for d in [p, *p.parents]:
        if (d / "targets.toml").is_file():
            return d
    raise SystemExit("找不到 targets.toml:请在 kernellab 仓库内运行")


@dataclass
class TargetConfig:
    name: str
    kind: str
    root: str = "~/klab"
    host: str = ""
    python: str = "3.12"
    cuda_bin: str = "/usr/local/cuda/bin"
    peaks: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "TargetConfig":
        peaks = {k: float(v) for k, v in d.items() if k.startswith("peak_")}
        return cls(
            name=name,
            kind=d["kind"],
            root=d.get("root", "~/klab"),
            host=d.get("host", ""),
            python=str(d.get("python", "3.12")),
            cuda_bin=d.get("cuda_bin", "/usr/local/cuda/bin"),
            peaks=peaks,
        )


def load_targets(root: Path) -> dict[str, TargetConfig]:
    data = tomllib.loads((root / "targets.toml").read_text())
    return {n: TargetConfig.from_dict(n, d) for n, d in data.get("targets", {}).items()}
