"""CuTe DSL dense GEMM 的薄包装:按当前卡挑 _vendor/ 里 NVIDIA 官方示例的 kernel 类,编译一次按形状缓存。

约定:A 是 (m, k) 行主序,B 以 (n, k) 行主序传入(即 B^T,"TN" GEMM),C 是 (m, n) 行主序。
三者对 CuTe 来说都是 k-major / n-major 的 3D 张量(末维 l = 1),这是两份示例的默认布局。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import from_dlpack  # noqa: E402

TILE_SM120 = (128, 128, 64)   # 官方示例默认
TILE_SM90 = (128, 128)        # 官方示例默认;cluster (1, 1)
CLUSTER_SM90 = (1, 1)

_cache: dict = {}


def _cute(t: torch.Tensor, dtype, leading_dim: int):
    """torch (rows, cols) → cute (rows, cols, 1),按 leading_dim 标记动态布局。"""
    t3 = t.unsqueeze(-1)
    ct = from_dlpack(t3, assumed_align=16)
    ct.element_type = dtype
    return ct.mark_layout_dynamic(leading_dim=leading_dim)


def _build(a, bt, c):
    major, minor = torch.cuda.get_device_capability(0)
    dt = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[a.dtype]
    mA, mB, mC = _cute(a, dt, 1), _cute(bt, dt, 1), _cute(c, dt, 1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if major == 12:
        from sm120_dense_gemm import Sm120GemmKernel

        gemm = Sm120GemmKernel(cutlass.Float32, TILE_SM120)
        max_active = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        compiled = cute.compile(gemm, mA, mB, mC, max_active, stream)
        return lambda A, B, C: compiled(A, B, C, stream), "sm120"
    if major == 9:
        from hopper_dense_gemm import HopperWgmmaGemmKernel

        gemm = HopperWgmmaGemmKernel(cutlass.Float32, TILE_SM90, CLUSTER_SM90)
        compiled = cute.compile(gemm, mA, mB, mC, stream)
        return lambda A, B, C: compiled(A, B, C, stream), "sm90"
    raise SystemExit(f"matmul_cute 只带了 sm_120 与 sm_90 两份实现,当前卡 cc {major}.{minor}")


def make_inputs(case, device):
    m, n, k = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "float16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, device=device, dtype=torch.float32, generator=g).to(dtype)
    bt = torch.randn(n, k, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "bt": bt}


def run(a, bt):
    m, k = a.shape
    n = bt.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=a.dtype)
    key = (m, n, k, str(a.dtype))
    if key not in _cache:
        _cache[key] = _build(a, bt, c)
    fn, _ = _cache[key]
    dt = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[a.dtype]
    fn(_cute(a, dt, 1), _cute(bt, dt, 1), _cute(c, dt, 1))
    return c


def reference(a, bt):
    return (a.float() @ bt.float().t()).to(a.dtype)


def workload(case, a, bt):
    m, k = a.shape
    n = bt.shape[0]
    e = a.element_size()
    return {"flops": 2 * m * n * k, "bytes": (m * k + n * k + m * n) * e}
