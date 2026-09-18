"""行 softmax:每个 program 负责一行,整行读进寄存器一次算完(融合 max / exp / sum / 归一化)。"""
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride_row, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * stride_row + offs, (num / den).to(out_ptr.dtype.element_ty), mask=mask)


def make_inputs(case, device):
    rows, cols = int(case["rows"]), int(case["cols"])
    dtype = getattr(torch, case.get("dtype", "float32"))
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(rows, cols, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"x": x}


def run(x):
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)
    num_warps = 4 if BLOCK < 2048 else (8 if BLOCK < 8192 else 16)
    softmax_kernel[(rows,)](x, out, x.stride(0), cols, BLOCK=BLOCK, num_warps=num_warps)
    return out


def reference(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def workload(case, x):
    rows, cols = x.shape
    return {"flops": 5 * rows * cols, "bytes": 2 * rows * cols * x.element_size()}
