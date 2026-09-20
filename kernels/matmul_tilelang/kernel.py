"""TileLang × Ampere v0 — 最朴素的矩阵乘基线。

TileLang 把「搬到共享内存 → 开流水 → 用 tensor core 乘」写成三句话。
降到 mma.sync 还是 wgmma 由它按架构自己决定 —— 用 klab ptx 可以确认。

只写 kernel 本体:jit 编译、按形状缓存、启动都由接线做。
"""

import tilelang.language as T


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_STAGES = 3     # T.Pipelined 的流水级数
THREADS = 128      # 一个 block 多少线程


def gemm(M, N, K, dtype, accum_dtype, out_dtype=None):
    """返回一个 T.prim_func。M/N/K 是编译期形状:TileLang 按形状特化,一个 case 编一次。

    out_dtype 由接线传:普通情况与输入同 dtype;split-K 时是 fp32(见参考答案 3)。
    """
    out_dtype = out_dtype or dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        # 二维 grid:(列块, 行块)。TileLang 自己管线程到数据的映射,不用手写指针
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (bx, by):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)   # 累加器住在寄存器

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=NUM_STAGES):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)   # 全局 -> 共享(自动 cp.async)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)              # 块乘块,走 tensor core
            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])       # 写回,自动转 dtype

    return kernel
