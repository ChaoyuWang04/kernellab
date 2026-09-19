"""朴素分块 —— CuTe DSL 基线

CuTe DSL 是 CUTLASS 的 Python 前端:写出来像 Python,编译后是真 CUDA kernel。
这一级只用最基础的部分 —— 线程索引 + 张量下标,和裸 CUDA 的第 0 级一一对应,
连合并访存都已经做了(x 决定列)。

5090 实测 4096³:35.6 ms,3.9 TFLOPS,0.02× torch。
大致相当于裸 CUDA 的第 1 级(合并访存,40.5 ms / 3.4 TFLOPS)—— 合理,
因为这两份代码在做同一件事。

**CuTe 真正的价值不在这一级。** 它的卖点是 layout 代数:用 TiledCopy / TiledMma
描述「数据怎么切、谁搬哪块、怎么喂给 tensor core」,让 swizzle、多级流水、
warp specialization 这些变成组合 atom 而不是手写指针。那一层还没写进这条阶梯 ——
官方的 dense GEMM 示例(几千行)是那个层次的参考,见 git 历史里的
kernels/matmul_cute/_vendor/。
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
