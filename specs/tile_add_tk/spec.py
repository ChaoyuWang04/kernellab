"""接线:kernels/tile_add_tk/kernel.cu(ThunderKittens)现场 nvcc 编成 torch 扩展,导出 tile_add(A, B)。"""
import torch

from klab.harness.cppext import load_extension

_ext = None


def _mod():
    global _ext
    if _ext is None:
        _ext = load_extension("tile_add_tk", __file__, sources=["kernel.cu"], tk=True)
    return _ext


def make_inputs(case, device):
    rows, cols = int(case["rows"]), int(case["cols"])
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(rows, cols, device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    b = torch.randn(rows, cols, device=device, dtype=torch.float32, generator=g).to(torch.bfloat16)
    return {"a": a, "b": b}


def run(a, b):
    return _mod().tile_add(a, b)


def reference(a, b):
    return a + b


def workload(case, a, b):
    n = a.numel()
    return {"flops": n, "bytes": 3 * n * a.element_size()}
