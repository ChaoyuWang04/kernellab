"""接线:kernels/softmax/kernel.py 的 softmax(x)。"""
import torch

from klab.harness.spec import kernel_module

k = kernel_module(__file__)


def make_inputs(case, device):
    rows, cols = int(case["rows"]), int(case["cols"])
    dtype = getattr(torch, case.get("dtype", "float32"))
    g = torch.Generator(device=device).manual_seed(0)
    return {"x": torch.randn(rows, cols, device=device, dtype=torch.float32, generator=g).to(dtype)}


def run(x):
    return k.softmax(x)


def reference(x):
    return torch.softmax(x, dim=-1)  # torch 对 half 输入内部用 fp32 算,再写回 half;不要先 .float(),那会多一次拷贝、标尺不公平


def workload(case, x):
    rows, cols = x.shape
    return {"flops": 5 * rows * cols, "bytes": 2 * rows * cols * x.element_size()}
