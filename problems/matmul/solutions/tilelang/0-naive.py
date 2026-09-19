"""朴素分块 —— TileLang 基线

TileLang 把「搬到共享内存 → 开流水 → 用 tensor core 乘」写成三句话:
T.copy / T.Pipelined / T.gemm。不写指针、不写掩码 —— 边界它自己处理,
降到 mma.sync 还是 wgmma 也由它按架构决定。

5090 实测,与同分块的 Triton 版并排:

    case                Triton      TileLang
    4096³               0.95×       0.88×      (205 vs 187 TFLOPS)
    8192³               1.01×       0.96×
    1000×999×777        0.37×       0.57×

大方阵 Triton 赢,喂不满 GPU 的小奇怪形状 TileLang 赢。同样的分块常量、
同样的 case 与容差,差的是两个编译器各自的取舍 —— 这正是一道题写多种语言的意义。
"""

import tilelang.language as T


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_STAGES = 3     # T.Pipelined 的流水级数
THREADS = 128      # 一个 block 多少线程


def gemm(M, N, K, dtype, accum_dtype):
    """返回一个 T.prim_func。M/N/K 是编译期形状:TileLang 按形状特化,一个 case 编一次。"""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
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
