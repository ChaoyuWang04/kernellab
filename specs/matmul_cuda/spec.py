"""接线:kernels/matmul_cuda/kernel.cu 现场 nvcc 编成 torch 扩展。

用户的 .cu 只写 CUDA(__global__ kernel + 启动它的 matmul_launch);
torch 张量检查与 pybind 导出在同目录的 binding.cu 里,一起编进来。
"""
import torch

from klab.harness.cppext import load_extension

_ext = None

# 与 Triton / TileLang 版同样的理由,见 specs/matmul_triton_ampere/spec.py。
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _mod():
    global _ext
    if _ext is None:
        _ext = load_extension("matmul_cuda", __file__, sources=["kernel.cu"], spec_sources=["binding.cu"])
    return _ext


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(kk, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def run(a, b):
    return _mod().matmul(a, b)


def reference(a, b):
    return a @ b


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
