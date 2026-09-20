"""CuTe DSL 矩阵乘。C[M,N] = A[M,K] @ B[K,N],bf16 进、fp32 累加、bf16 出。

CuTe DSL 是 CUTLASS 的 Python 前端:写出来像 Python,编译后是真 CUDA kernel。
下面的常量与 matmul_kernel 的签名是你与系统之间的契约:接线负责把 torch 张量包成
cute.Tensor、写 @cute.jit 的启动壳、cute.compile 并按形状缓存。改了签名就跑不起来。

线程/块索引:cute.arch.thread_idx() 与 block_idx() 都返回三元组。
张量按 (行, 列) 下标取,元素用 .to(cutlass.Float32) 转类型。
"""

import cutlass
import cutlass.cute as cute


BLOCK_M = 16
BLOCK_N = 16
BLOCK_THREADS = (BLOCK_N, BLOCK_M, 1)    # 一个 block 的线程排布,x 走列


@cute.kernel
def matmul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                  M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    # ① 我是谁:从 thread_idx / block_idx 算出负责 C 的哪个元素
    #    (让 x 决定列 —— 同一个 warp 落在连续的列上,访存才合并)

    # ② 越界判断:M / N 不保证是分块的整数倍

    # ③ 开累加器:cutlass.Float32(0.0),不要用 bf16 累加

    # ④ 沿 K 累加:for k in cutlass.range(K, unroll=1)

    # ⑤ 写回:acc.to(cutlass.BFloat16)

    pass
