"""Modal 后端:按 targets.toml 里的 gpu 起一个容器,仓库以本地目录挂载(不重建镜像),
脚本在容器里跑,runs/ 打包成 tar 传回来。凭据由 Modal SDK 自己读 ~/.modal.toml。

镜像 = nvidia/cuda devel 基础镜像 + envs/<工具链>/ 里的依赖;首次构建几分钟,之后命中缓存。
"""
from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

from klab.targets.base import SYNC_EXCLUDES, Target

BASE_IMAGE = "nvidia/cuda:13.0.1-devel-ubuntu24.04"
PYTHON = "3.12"


class ModalTarget(Target):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._local_root: Path | None = None
        self._last_tar: bytes = b""

    @property
    def gpu(self) -> str:
        return self.cfg.gpu or "H100"

    def env_python(self, toolchain: str) -> str:
        return "python"  # 依赖装在镜像的系统 Python 里,没有 per-toolchain venv

    def sync(self, local_root: Path) -> None:
        self._local_root = local_root  # 真正的“同步”发生在 add_local_dir 挂载时

    def _choose_route(self) -> None:
        """先试直连 api.modal.com:443;不通就走 targets.toml 的 api_proxy(默认 http://127.0.0.1:3213)。

        必须在 import modal 之前决定:SDK 在导入时读 MODAL_DISABLE_API_PROXY 与 HTTPS_PROXY。
        """
        import socket

        proxy = self.cfg.extra.get("api_proxy", "http://127.0.0.1:3213")
        try:
            with socket.create_connection(("api.modal.com", 443), timeout=4):
                pass
            os.environ["MODAL_DISABLE_API_PROXY"] = "1"
            print("[modal] 直连 api.modal.com", flush=True)
        except OSError as e:
            os.environ.pop("MODAL_DISABLE_API_PROXY", None)
            os.environ["HTTPS_PROXY"] = proxy
            os.environ["HTTP_PROXY"] = proxy
            print(f"[modal] 直连失败({e.__class__.__name__}),改走代理 {proxy}", flush=True)

    def _build(self, toolchain: str):
        self._choose_route()
        import modal

        root = self._local_root or Path.cwd()
        env_dir = root / "envs" / toolchain
        torch_index = (env_dir / "torch-index.txt").read_text().strip().splitlines()[0]
        from klab import toolchains

        tc = toolchains.get(toolchain)
        image = (
            modal.Image.from_registry(BASE_IMAGE, add_python=PYTHON)
            .apt_install("git", *getattr(tc, "APT", []))
            .pip_install("torch", index_url=torch_index)
            .pip_install_from_requirements(str(env_dir / "requirements.txt"))
        )
        for cmd in getattr(tc, "MODAL_RUN_COMMANDS", []):
            image = image.run_commands(cmd)
        image = (
            image.env({"PYTHONPATH": self.repo_dir, "KLAB_ROOT": self.root, **getattr(tc, "MODAL_ENV", {})})
            .add_local_dir(root, remote_path=self.repo_dir, ignore=[*SYNC_EXCLUDES, "**/__pycache__"])
        )
        app = modal.App(f"klab-{toolchain}")
        runs_dir = self.runs_dir
        # 编译缓存(cpp_extension、tilelang、triton)跨调用持久化,否则每次冷启动都重编
        cache = modal.Volume.from_name("klab-cache", create_if_missing=True)

        @app.function(image=image, gpu=self.gpu, timeout=3600, serialized=True, volumes={"/root/.cache": cache})
        def run_script(script: str) -> tuple[int, bytes]:
            import io as _io
            import os as _os
            import subprocess as _sp
            import tarfile as _tar

            _os.makedirs(runs_dir, exist_ok=True)
            p = _sp.run(["bash", "-c", script])
            buf = _io.BytesIO()
            with _tar.open(fileobj=buf, mode="w:gz") as t:
                t.add(runs_dir, arcname="runs")
            return p.returncode, buf.getvalue()

        return app, run_script

    def run(self, script: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
        toolchain = os.environ.get("KLAB_TOOLCHAIN", "triton")
        app, fn = self._build(toolchain)
        import modal

        # 程序化调用 app.run() 时,容器的 stdout 只有在 enable_output() 里才会流回本地终端
        with modal.enable_output(), app.run():
            rc, tar_bytes = fn.remote(self.wrap(script))
        self._last_tar = tar_bytes
        if check and rc != 0:
            raise subprocess.CalledProcessError(rc, script)
        return subprocess.CompletedProcess(script, rc)

    def fetch(self, remote_path: str, local_dir: Path) -> None:
        """从最近一次 run 传回的 tar 里取出 runs/<name>。"""
        name = remote_path.rstrip("/").split("/")[-1]
        local_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(self._last_tar), mode="r:gz") as t:
            members = [m for m in t.getmembers() if m.name.startswith(f"runs/{name}/") or m.name == f"runs/{name}"]
            for m in members:
                m.name = m.name[len("runs/"):]
            t.extractall(local_dir, members=members, filter="data")

    def shell(self) -> None:
        raise SystemExit(f"Modal 没有常驻机器可进;要交互调试用:modal shell --gpu {self.gpu} {BASE_IMAGE}")
