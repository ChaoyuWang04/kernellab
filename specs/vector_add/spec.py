"""接线:kernels/vector_add/kernel.py 的 add(x, y)。"""
import torch

from klab.harness.spec import kernel_module

k = kernel_module(__file__)


def make_inputs(case, device):
    n = int(case["n"])
    dtype = getattr(torch, case.get("dtype", "float32"))
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(n, device=device, dtype=torch.float32, generator=g).to(dtype)
    y = torch.randn(n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"x": x, "y": y}


def run(x, y):
    return k.add(x, y)


def reference(x, y):
    return x + y


def workload(case, x, y):
    n = x.numel()
    return {"flops": n, "bytes": 3 * n * x.element_size()}
