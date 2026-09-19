"""TileLang 矩阵乘。C[M,N] = A[M,K] @ B[K,N],bf16 进、fp32 累加、bf16 出。

下面的常量和 gemm() 的签名是你与系统之间的契约:接线按这些常量编译、按这个签名
拿 prim_func,改了就跑不起来。prim_func 的内容是你的,从头写。

TileLang 不写指针:你声明共享内存 / 寄存器块,用 T.copy 搬、T.gemm 算,
线程到数据的映射它自己管。
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
        # ① 开 grid:一个 block 负责 C 的一块 BLOCK_M × BLOCK_N
        #    with T.Kernel(列块数, 行块数, threads=THREADS) as (bx, by):

        # ② 申请共享内存放 A、B 的 tile,申请 fragment 放累加器(累加器用 accum_dtype)

        # ③ 累加器清零

        # ④ 沿 K 开流水:T.Pipelined(段数, num_stages=NUM_STAGES)
        #    每段:T.copy 把 A、B 的 tile 搬进共享内存,T.gemm 乘加进累加器

        # ⑤ 写回:T.copy 把累加器搬回 C 的对应块(自动转 dtype)

        pass

    return kernel
