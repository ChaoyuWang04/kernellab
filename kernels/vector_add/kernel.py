"""最小 Triton 算子:逐元素加。用来验证整条链路(check → bench → ncu)。"""
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def make_inputs(case, device):
    n = int(case["n"])
    dtype = getattr(torch, case.get("dtype", "float32"))
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(n, device=device, dtype=torch.float32, generator=g).to(dtype)
    y = torch.randn(n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"x": x, "y": y}


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    return out


def reference(x, y):
    return x + y


def workload(case, x, y):
    n = x.numel()
    return {"flops": n, "bytes": 3 * n * x.element_size()}
