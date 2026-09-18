"""接线:kernels/matmul_cute/kernel.py 的 matmul_tn(a, bt)(CuTe DSL,B 以 (n, k) 传入)。"""
import torch

from klab.harness.spec import kernel_module

k = kernel_module(__file__)


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "float16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    bt = torch.randn(n, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "bt": bt}


def run(a, bt):
    return k.matmul_tn(a, bt)


def reference(a, bt):
    return a @ bt.t()  # 同 dtype 的 cuBLAS,公平标尺


def workload(case, a, bt):
    m, kk = a.shape
    n = bt.shape[0]
    e = a.element_size()
    return {"flops": 2 * m * n * kk, "bytes": (m * kk + n * kk + m * n) * e}
