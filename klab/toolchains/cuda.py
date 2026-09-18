"""裸 CUDA C++ 工具链:kernel.py 里用 torch.utils.cpp_extension.load() 把同目录的 .cu 编成扩展。

编译缓存在后端的 ~/.cache/klab/<算子名>(SSH 后端持久;Modal 用 Volume 持久化,见 targets/modal.py)。
"""
from klab.toolchains._pip import setup_script as _pip_setup

NAME = "cuda"
APT = ["build-essential"]  # Modal 镜像:nvidia/cuda devel 自带 gcc,这里只是兜底


def setup_script(root: str, python: str, repo_dir: str) -> str:
    return _pip_setup(NAME, root, python, repo_dir)
