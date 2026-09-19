"""接线:kernels/matmul_tilelang/kernel.py 的 gemm(TileLang,bf16)。

用户只写 T.prim_func;jit 编译、按形状缓存、分配输出、启动都在这里。
TileLang 按形状特化,所以缓存键带上 M/N/K 与 dtype,一个 case 只编一次。
"""
import tilelang
import torch

from klab.harness.spec import kernel_module

k = kernel_module(__file__)

# 与 Triton 版同样的理由:cuBLAS 在块数喂不满 GPU 时走 split-K 并用 bf16 归约部分和,
# 会给接近 0 的输出带来约 0.1 的绝对误差,让 check 误判成算子写错。见 specs/matmul_triton_ampere/spec.py。
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

_cache: dict = {}


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(kk, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def _compiled(M, N, K, dtype):
    key = (M, N, K, dtype, k.BLOCK_M, k.BLOCK_N, k.BLOCK_K, k.NUM_STAGES, k.THREADS)
    if key not in _cache:
        @tilelang.jit(out_idx=[-1])          # 最后一个张量参数是输出,由 TileLang 自己分配
        def build():
            return k.gemm(M, N, K, dtype, "float")
        _cache[key] = build()
    return _cache[key]


def run(a, b):
    M, K = a.shape
    N = b.shape[1]
    return _compiled(M, N, K, str(a.dtype).replace("torch.", ""))(a, b)


def reference(a, b):
    return a @ b  # 同 dtype 的 cuBLAS;不要先 .float(),否则标尺虚高


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
