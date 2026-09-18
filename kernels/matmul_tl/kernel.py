"""TileLang 版分块 matmul(fp16 输入、fp32 累加),与 kernels/matmul 的 Triton 版同一分块尺寸,便于并排比。

TileLang 把「搬到共享内存 → 流水 → 用 tensor core 乘」写成三句;具体降到 mma.sync 还是 wgmma 由它按架构决定。
编译结果按形状缓存在进程内,一个 case 只编一次。
"""
import torch
import tilelang
import tilelang.language as T

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
NUM_STAGES, THREADS = 3, 128

_cache: dict = {}


def configure(**p):
    """klab sweep 用:改分块常量并清掉已编译缓存。"""
    globals().update({k: int(v) for k, v in p.items()})
    _cache.clear()


def _build(M, N, K, dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=[-1])
    def matmul(M, N, K, block_M, block_N, block_K):
        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=THREADS) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=NUM_STAGES):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    return matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)


def _kernel(a, b):
    M, K = a.shape
    N = b.shape[1]
    key = (M, N, K, str(a.dtype))
    if key not in _cache:
        _cache[key] = _build(M, N, K, dtype=str(a.dtype).replace("torch.", ""))
    return _cache[key]


def make_inputs(case, device):
    m, n, k = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "float16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, k, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(k, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def run(a, b):
    return _kernel(a, b)(a, b)


def reference(a, b):
    return (a.float() @ b.float()).to(a.dtype)


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
