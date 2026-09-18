"""CuTe DSL 工具链:纯 pip(nvidia-cutlass-dsl[cu13]),JIT 编译。"""
from klab.toolchains._pip import setup_script as _pip_setup

NAME = "cute"


def setup_script(root: str, python: str, repo_dir: str) -> str:
    return _pip_setup(NAME, root, python, repo_dir)
