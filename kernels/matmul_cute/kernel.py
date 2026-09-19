"""CuTe DSL × Ampere v0 — 最朴素的矩阵乘基线。

CuTe DSL 是 CUTLASS 的 Python 前端:写出来像 Python,编译后是真 CUDA kernel。
这一级只用最基础的部分 —— 线程索引 + 张量下标,和裸 CUDA 的第 0 级一一对应。

只写 kernel 本体与分块常量:cute.compile、按形状缓存、启动都由接线做。
"""

import cutlass
import cutlass.cute as cute


BLOCK_M = 16
BLOCK_N = 16
BLOCK_THREADS = (BLOCK_N, BLOCK_M, 1)    # 一个 block 的线程排布,x 走列


@cute.kernel
def matmul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                  M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    # ① 我是谁。x 决定列 —— 同一个 warp 落在连续的列上,访存才合并
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    col = bx * BLOCK_N + tx
    row = by * BLOCK_M + ty

    # ② 越界的直接退出:M / N 不保证是分块的整数倍
    if row < M and col < N:
        # ③ 累加器用 fp32,不要用 bf16
        acc = cutlass.Float32(0.0)
        # ④ 沿 K 累加。张量按 (行, 列) 下标取,布局由接线标注
        for k in cutlass.range(K, unroll=1):
            acc += mA[row, k].to(cutlass.Float32) * mB[k, col].to(cutlass.Float32)
        # ⑤ 写回,转回 bf16
        mC[row, col] = acc.to(cutlass.BFloat16)
