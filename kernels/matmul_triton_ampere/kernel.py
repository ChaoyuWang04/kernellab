"""Triton 矩阵乘。C[M,N] = A[M,K] @ B[K,N],bf16 进、fp32 累加、bf16 出。

下面的常量和函数签名是你与系统之间的契约:系统按这些常量算 grid、按这个签名启动,
改了就跑不起来。函数体是你的,从头写。
"""

import triton
import triton.language as tl


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 4      # 一个 CTA 里几个 warp
NUM_STAGES = 3     # 共享内存流水线深度(Ampere 的 cp.async 多级缓冲)


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,                 # 三个矩阵的起始地址
    M, N, K,                             # 形状,运行期传入
    stride_am, stride_ak,                # A 的行跨度、列跨度(行优先时 = K, 1)
    stride_bk, stride_bn,                # B 的(= N, 1)
    stride_cm, stride_cn,                # C 的(= N, 1)
    BLOCK_M: tl.constexpr,               # constexpr:编译期常量,决定 tile 形状
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ① 我是谁:grid 是一维的,把 pid 换算成「第几个行块、第几个列块」
    #    (换一种换算方式就是优化路线第 6 级的 L2 swizzle)

    # ② 我这块覆盖哪些行号、列号,以及 K 方向的块内偏移

    # ③ 把坐标换成地址:基址 + 行*行跨度 + 列*列跨度
    #    [:, None] / [None, :] 把一维向量广播成二维

    # ④ 开一个 [BLOCK_M, BLOCK_N] 的 fp32 累加器

    # ⑤ 沿 K 一段一段乘加。M/N/K 不保证是分块的整数倍,越界的位置要掩掉读 0

    # ⑥ 写回:累加器转回 bf16,越界位置不写

    pass
