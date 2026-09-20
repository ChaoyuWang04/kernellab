"""调分块 —— 手拧旋钮能拧到哪

代码一行没动,只改四个常量:

    BLOCK_K    32 → 64     每步吃更深,循环次数减半
    THREADS   128 → 256    分块变大要更多线程来搬

**这一级最值钱的是我先撞的那面墙。** 我一开始写的是 BLOCK_N=256 + NUM_STAGES=4,
上机直接报:

    InternalError: Failed to set the allowed dynamic shared memory size to 196608

共享内存用量 = (BLOCK_M×BLOCK_K + BLOCK_K×BLOCK_N) × 2 字节 × NUM_STAGES:

    128×64 + 64×256 = 24576 元素 × 2B = 49152 B  × 4 级 = 196608 B  ✗ 超
                                                   × 3 级 = 147456 B  ✗ 还超
    128×64 + 64×128 = 16384 元素 × 2B = 32768 B  × 3 级 =  98304 B  ✓ 刚好

5090 的上限是 **101376 字节**。这就是优化路线第 2 级说的那个天花板 ——
**它是硬的,不是建议值**,而且每张卡不一样(H100 是 228 KB)。

所以最终这一级只动了 BLOCK_K 和 THREADS。5090 实测 4096³:

    基线(BLOCK_K=32, THREADS=128)   0.7332 ms   187.5 TFLOPS   0.88×
    这一级(BLOCK_K=64, THREADS=256)  0.6574 ms   209.1 TFLOPS   0.97×   +11%

两个常量换来 11%,而且**超过了 Triton 手调那版的 0.95×**。

但别高兴太早:这两个数是我试出来的,不是算出来的。手拧旋钮会撞墙,
墙在哪只有卡知道(5090 是 101376 字节,H100 是 228 KB)——
这正是该交给 autotune 的理由。
"""

import tilelang.language as T


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
NUM_STAGES = 3     # 想开 4 级?见上面的说明 —— 共享内存不够
THREADS = 256      # 128 → 256:分块变大,要更多线程来搬


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
