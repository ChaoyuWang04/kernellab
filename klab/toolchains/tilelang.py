"""TileLang 工具链:纯 pip,JIT 时调 nvcc(后端 PATH 里已有 cuda/bin)。"""
from klab.toolchains._pip import setup_script as _pip_setup

NAME = "tilelang"


def setup_script(root: str, python: str, repo_dir: str) -> str:
    return _pip_setup(NAME, root, python, repo_dir)
