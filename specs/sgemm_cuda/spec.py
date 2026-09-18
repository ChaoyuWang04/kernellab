"""接线:kernels/sgemm_cuda/kernel.cu 现场 nvcc 编成 torch 扩展,导出 sgemm(A, B)。"""
import torch

from klab.harness.cppext import load_extension

_ext = None


def _mod():
    global _ext
    if _ext is None:
        _ext = load_extension("sgemm_cuda", __file__, sources=["kernel.cu"])
    return _ext


def make_inputs(case, device):
    m, n, k = int(case["m"]), int(case["n"]), int(case["k"])
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, device=device, dtype=torch.float32, generator=g)
    b = torch.randn(k, n, device=device, dtype=torch.float32, generator=g)
    return {"a": a, "b": b}


def run(a, b):
    return _mod().sgemm(a, b)


def reference(a, b):
    return a @ b


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * 4}
