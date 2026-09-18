"""Target 接口:后端只回答「在哪跑」。

四个动作:probe_cmd 之外全在这里——sync(把仓库送过去)、run(在远端 repo 目录执行命令)、
fetch(把结果目录拉回来)、shell(交互)。编译、跑、测速、NCU 都是 run() 之上组合出来的命令串。
"""
from __future__ import annotations

import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from klab.config import TargetConfig

SYNC_EXCLUDES = [".git", ".venv", "runs", "__pycache__", "*.pyc", ".klab-cache", ".vscode"]


class Target(ABC):
    def __init__(self, cfg: TargetConfig):
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.cfg.name

    # ---- 路径约定(远端) ----
    @property
    def root(self) -> str:
        return self.cfg.root  # 可能含 ~,交给远端 shell 展开

    @property
    def repo_dir(self) -> str:
        return f"{self.root}/repo"

    @property
    def runs_dir(self) -> str:
        return f"{self.root}/runs"

    def env_python(self, toolchain: str) -> str:
        return f"{self.root}/envs/{toolchain}/bin/python"

    def env_prefix(self) -> str:
        """远端非交互 shell 的 PATH 里没有 ~/.local/bin 与 cuda,统一补上。"""
        return f'export PATH="$HOME/.local/bin:{self.cfg.cuda_bin}:$PATH"; export PYTHONPATH="{self.repo_dir}"; '

    # ---- 抽象动作 ----
    @abstractmethod
    def sync(self, local_root: Path) -> None: ...

    @abstractmethod
    def run(self, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        """在远端以 bash 执行 script(已 cd 到 repo 目录、已设 PATH)。capture=True 时返回 stdout。"""

    @abstractmethod
    def fetch(self, remote_path: str, local_dir: Path) -> None:
        """把远端目录/文件拉到本地目录下。"""

    @abstractmethod
    def shell(self) -> None: ...

    # ---- 公共组合 ----
    def wrap(self, script: str, cwd: str | None = None) -> str:
        cwd = cwd or self.repo_dir
        return f"{self.env_prefix()}mkdir -p {self.runs_dir} && cd {cwd} && {script}"


def q(s: str) -> str:
    return shlex.quote(s)
