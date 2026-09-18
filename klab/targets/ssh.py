"""SSH 后端:rsync 送代码,ssh 跑命令,rsync 拉结果。ControlMaster 复用连接。"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from klab.targets.base import SYNC_EXCLUDES, Target, q

CM_OPTS = [
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=~/.ssh/klab-cm-%C",
    "-o", "ControlPersist=600",
]


class SshTarget(Target):
    @property
    def host(self) -> str:
        return self.cfg.host

    def _ssh(self, *opts: str) -> list[str]:
        """ssh <控制选项> <额外选项> <host> --;命令跟在后面。"""
        return ["ssh", *CM_OPTS, *opts, self.host, "--"]

    def sync(self, local_root: Path) -> None:
        # 远端目录先建好;rsync 目标用相对 HOME 的路径,避开 ~ 展开差异
        self.run_raw(f"mkdir -p {self.repo_dir} {self.runs_dir} {self.root}/envs")
        dest = self.repo_dir.replace("~/", "")
        cmd = ["rsync", "-az", "--delete", "-e", "ssh " + " ".join(CM_OPTS)]
        for e in SYNC_EXCLUDES:
            cmd += ["--exclude", e]
        cmd += [str(local_root) + "/", f"{self.host}:{dest}/"]
        subprocess.run(cmd, check=True)

    def run_raw(self, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._ssh() + ["bash", "-c", q(script)],
            check=check,
            text=True,
            capture_output=capture,
        )

    def run(self, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        return self.run_raw(self.wrap(script), check=check, capture=capture)

    def fetch(self, remote_path: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        src = remote_path.replace("~/", "")
        subprocess.run(
            ["rsync", "-az", "-e", "ssh " + " ".join(CM_OPTS), f"{self.host}:{src}", str(local_dir) + "/"],
            check=True,
        )

    def shell(self) -> None:
        os.execvp("ssh", self._ssh("-t") + ["bash", "-c", q(self.env_prefix() + f"cd {self.repo_dir}; exec bash")])
