"""本机后端:在当前机器上直接跑(用于在 GPU 盒子上调试 harness)。"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from klab.targets.base import Target


class LocalTarget(Target):
    def _bash(self, script: str, check: bool, capture: bool) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", script], check=check, text=True, capture_output=capture)

    def sync(self, local_root: Path) -> None:
        repo = Path(os.path.expanduser(self.repo_dir))
        if repo.resolve() == local_root.resolve():
            return
        repo.parent.mkdir(parents=True, exist_ok=True)
        self._bash(
            "rsync -a --delete " + " ".join(f"--exclude {e}" for e in ["'.git'", "'.venv'", "'runs'", "'__pycache__'"])
            + f" {local_root}/ {repo}/",
            True, False,
        )

    def run(self, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        return self._bash(self.wrap(script), check, capture)

    def fetch(self, remote_path: str, local_dir: Path) -> None:
        src = Path(os.path.expanduser(remote_path))
        local_dir.mkdir(parents=True, exist_ok=True)
        dst = local_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    def shell(self) -> None:
        os.execvp("bash", ["bash", "-c", self.env_prefix() + f"cd {self.repo_dir}; exec bash"])
