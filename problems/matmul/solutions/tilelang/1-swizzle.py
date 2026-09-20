"""L2 swizzle —— 一句话 vs 十行

Triton 版的这一级要自己写十行 pid 换算(group_id / first_pid_m / group_size_m ...),
TileLang 一句 T.use_swizzle(panel_size) 就完了 —— block 到 tile 的映射是调度层的事,
不是算法的事,TileLang 把它放在了调度层。

这是两种 DSL 设计哲学的直接对照:Triton 的 grid 是裸的一维 pid,重排归你;
TileLang 的 T.Kernel 是带语义的,重排是它的一个旋钮。

5090 实测 4096³:189.0 vs 基线 187.5 TFLOPS —— +0.8%,在噪声里。
和 Triton 版同样的结论:L2 命中率本来就高,没有可省的。
代码上省了十行,性能上没省什么 —— 这一级买的是可读性,不是速度。
"""

import tilelang.language as T


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_STAGES = 3     # T.Pipelined 的流水级数
SWIZZLE_PANEL = 10  # L2 swizzle 的面板宽度;0 关掉
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
            # 一句话开 L2 swizzle:把 block 按 panel 分组铺,同批 block 共用 tile。
            # Triton 里这一级要自己写十行 pid 换算(见 triton/1-swizzle.py)
            T.use_swizzle(panel_size=SWIZZLE_PANEL)
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
