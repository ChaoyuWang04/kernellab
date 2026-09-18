"""工具链层:每种 DSL 一个模块,回答「环境怎么装」与「这个算子怎么跑」。

triton / tilelang / cute 是纯 pip 环境、JIT 编译;cuda / tk 用 torch cpp_extension 现场 nvcc 编译。全部共用 _pip.py 的装法。
模块可选属性:APT(Modal 镜像额外 apt 包)、MODAL_RUN_COMMANDS(镜像额外命令)、MODAL_ENV(镜像环境变量)。
"""
from __future__ import annotations

from klab.toolchains import cuda as _cuda
from klab.toolchains import cute as _cute
from klab.toolchains import tilelang as _tilelang
from klab.toolchains import tk as _tk
from klab.toolchains import triton as _triton

REGISTRY = {"triton": _triton, "tilelang": _tilelang, "cute": _cute, "cuda": _cuda, "tk": _tk}


def get(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"未知工具链 {name!r};已有:{list(REGISTRY)}")
    return REGISTRY[name]
