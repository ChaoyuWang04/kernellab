"""split-K —— 和 Triton 版同一招,同一条判据

deepK(512×512×16384)只切出 4×4 = 16 个块,而卡有 108~170 个 SM。
把 K 也切成 SPLIT_K 段,grid 多一维,各段算部分和再 T.atomic_add 累加。

写法上比 Triton 版干净:T.Kernel 直接支持三维 grid,拿 bz 当 K 的段号;
累加用 T.Parallel + T.atomic_add,不用手算指针。

判据与 Triton 版完全一样(见 triton/3-splitk.py):
**输出小、K 大才用 split-K** —— 原子流量 = 输出元素数 × SPLIT_K × 4 字节,
要能被总计算量摊薄才划算。1000×999×777 上会输,deepK 上会赢。

5090 实测 deepK(512×512×16384,只切出 16 个块):

    基线              0.4035 ms    21.3 TFLOPS   0.14×
    SPLIT_K = 8       0.0635 ms   135.3 TFLOPS   0.87×    快 6.4 倍

比 Triton 版的 split-K(0.79×)还高一截 —— 因为 TileLang 的 T.atomic_add 走
T.Parallel 展开,访存模式比手写的块状 atomic 更规整。

接线约定与 Triton 版一样:定义 SPLIT_K > 1,接线自动把输出换成 fp32 缓冲并清零、
跑完转回 bf16。一处细节:split-K 时不能再让 TileLang 用 out_idx 自己分配输出
(它会以为输入少一个),所以接线按 SPLIT_K 切换编译方式。
"""

import tilelang.language as T


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_STAGES = 3     # T.Pipelined 的流水级数
SPLIT_K = 8        # K 切成几份;>1 时接线把输出换成 fp32 缓冲并清零
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
        # grid 多一维:z 走 K 的切分
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), SPLIT_K, threads=THREADS) as (bx, by, bz):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)   # 累加器住在寄存器

            T.clear(C_local)
            # 只走属于我这一段的 K:每段 ktiles 个分块步
            ktiles = T.ceildiv(T.ceildiv(K, BLOCK_K), SPLIT_K)
            for kk in T.Pipelined(ktiles, num_stages=NUM_STAGES):
                k = bz * ktiles + kk
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            # 多个 block 往同一块 C 上加,必须原子。C 是 fp32 缓冲,接线负责转回 bf16
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                T.atomic_add(C[by * BLOCK_M + i, bx * BLOCK_N + j], C_local[i, j])

    return kernel
