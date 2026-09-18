"""工具链层:每种 DSL 一个模块,回答「环境怎么装」与「这个算子怎么跑」。

目前只有 triton(纯 pip 环境,JIT 编译,没有独立 build 步骤)。
后续 tilelang / cute / thunderkittens / cuda 各加一个模块;需要 nvcc 版本隔离的走 Docker 镜像。
"""
from __future__ import annotations

from klab.toolchains import triton as _triton

REGISTRY = {"triton": _triton}


def get(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"未知工具链 {name!r};已有:{list(REGISTRY)}")
    return REGISTRY[name]
