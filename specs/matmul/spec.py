"""接线:kernels/matmul/kernel.py 的 matmul(a, b)(Triton)。"""
import torch

from klab.harness.spec import kernel_module

k = kernel_module(__file__)


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "float16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(kk, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def run(a, b):
    return k.matmul(a, b)


def reference(a, b):
    return a @ b  # 同 dtype 的 cuBLAS(fp16 输入、fp32 累加),这才是公平的标尺;不要先 .float()


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}


def configure(**p):
    k.configure(**p)
