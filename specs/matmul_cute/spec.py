"""接线:kernels/matmul_cute/kernel.py 的 matmul_kernel(CuTe DSL,bf16)。

用户只写 @cute.kernel 本体与分块常量;把 torch 张量包成 cute.Tensor、写 @cute.jit
的启动壳、cute.compile 并按形状缓存,全在这里。

CuTe 的 compile 很贵(每个形状一次),所以缓存键带上 M/N/K 与全部分块常量。
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from klab.harness.spec import kernel_module

k = kernel_module(__file__)

# 与其他三种语言同样的理由,见 specs/matmul_triton_ampere/spec.py。
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

_cache: dict = {}


def _t(x: torch.Tensor):
    """torch 的二维行主序张量 -> cute.Tensor。leading_dim=1 表示最后一维连续。"""
    return from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)


@cute.jit
def _launch(mA, mB, mC, M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
            gx: cutlass.Constexpr, gy: cutlass.Constexpr, block: cutlass.Constexpr):
    k.matmul_kernel(mA, mB, mC, M, N, K).launch(grid=(gx, gy, 1), block=block)


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(kk, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def run(a, b):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    gx = (N + k.BLOCK_N - 1) // k.BLOCK_N
    gy = (M + k.BLOCK_M - 1) // k.BLOCK_M
    key = (M, N, K, str(a.dtype), k.BLOCK_M, k.BLOCK_N, k.BLOCK_THREADS)
    if key not in _cache:
        _cache[key] = cute.compile(_launch, _t(a), _t(b), _t(c),
                                   cutlass.Int32(M), cutlass.Int32(N), cutlass.Int32(K),
                                   gx, gy, k.BLOCK_THREADS)
    _cache[key](_t(a), _t(b), _t(c), cutlass.Int32(M), cutlass.Int32(N), cutlass.Int32(K))
    return c


def reference(a, b):
    return a @ b


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
