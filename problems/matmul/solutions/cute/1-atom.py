"""换 atom —— 把一条标量 FMA 换成 tensor core 的 mma.sync

上一级是「用了 CUTLASS 的皮,没用 CUTLASS 的肉」:线程索引 + 一个 for k 累加,
和裸 CUDA 的第 0 级一模一样。`klab ptx` 打出来只有 85 行,内层 **fma×1**,
判定「没有任何标志指令」。

CuTe 的卖点是 **atom**:一块预制的积木,它知道某条硬件指令要求数据怎么摆、哪个
线程拿哪一份。你挑中它,布局细节全归库管 —— 这一级的全部改动就是下面三行:

    op        = warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16))   # 就是 mma.sync
    tiled_mma = cute.make_tiled_mma(op, (WARPS_M, WARPS_N, 1))      # 铺满一个 block
    cute.gemm(tiled_mma, acc, rA, rB, acc)                          # 算

**这一级只动「怎么算」,不动「怎么搬」。** 搬运还是最笨的逐元素循环,这样变量只有
一个:标量 FMA → mma.sync。

5090 实测 4096³:

    0-naive   35.62 ms   3.86 TFLOPS   0.02× torch   PTX: fma×1,没有标志指令
    1-atom     3.74 ms  36.70 TFLOPS   0.17× torch   PTX: mma.sync×32,命中 Ampere

**快 9.5 倍。** 三行代码。

## 但它立刻告诉你下一级该干什么

体检单的判定不是「卡在算上」,而是**「两头都没跑满,时间花在等上」**:计算 18%、
内存 48%,tensor core 只有 15%。warp 的等待里 **76% 是「等显存把数据取回来」**。

这完全对得上 —— 我们只换了「算」,「搬」还是那个逐元素循环:没有向量化、没有
cp.async 流水、共享内存也没 swizzle。**tensor core 现在闲着等饭吃。** 另外两个数
也摆在那儿:

- 占用率只有 33%,**卡在寄存器上**(104 个/线程,每个 SM 只放得下 2 块)
- 1024 块要跑 3.01 波,**最后一波只用掉 1% 的位置,白扔约 25% 的时间**
  (优化路线第 7 节说的就是这个:3.01 波比 3.98 波糟得多)

所以「用上 tensor core」和「喂饱 tensor core」确实是两件事 —— 裸 CUDA 的
`5-tensorcore` 学到的是同一课,这里换成 CuTe 再撞一次。

## 一个意外的观察:check 报 max_abs_err 精确等于 0

4096³ 那档,这个 kernel 和 cuBLAS 的输出**逐位相同**,16,777,216 个元素无一例外。
不是比较写错了 —— 两者离 fp32 真值一样远(都是 1.0010)。

原因在 bf16 只有 8 位尾数:两数相乘得 16 位尾数,fp32 的 24 位**装得下**。只要求和
过程里量级跨度不超过 2^(24-16) = 256 倍,fp32 累加就是**精确的,与顺序无关**。
K=4096 刚好在边界内;K=16384 的 deepK 跨出去了,于是出现 2233/262144(0.85%)
个不同。所以「算子和 torch 位位相同」在这里不是巧合,是 bf16 的数值性质。
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.utils import SmemAllocator

# 启动参数。接线读 BLOCK_M / BLOCK_N 算 grid,读 BLOCK_THREADS 定 block。
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

WARPS_M = 2          # 一个 block 里 warp 怎么排:2×4 = 8 个 warp
WARPS_N = 4
NUM_THREADS = WARPS_M * WARPS_N * 32          # 256
BLOCK_THREADS = (NUM_THREADS, 1, 1)

MMA_MNK = (16, 8, 16)                          # 一条 mma.sync 算多大一块

# 每个线程在一次「搬一整块 tile」里负责几个元素
A_PER_THREAD = BLOCK_M * BLOCK_K // NUM_THREADS       # 16
B_PER_THREAD = BLOCK_N * BLOCK_K // NUM_THREADS       # 16


@cute.kernel
def matmul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                  M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # ① 共享内存。两块都要 K 连续(stride 的最后一维是 1)—— atom 的要求。
    #    SmemAllocator 会自己算总用量,启动时不用传 smem 大小。
    smem = SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16,
                              cute.make_layout((BLOCK_M, BLOCK_K), stride=(BLOCK_K, 1)),
                              byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.BFloat16,
                              cute.make_layout((BLOCK_N, BLOCK_K), stride=(BLOCK_K, 1)),
                              byte_alignment=16)

    # ② 挑积木,铺满这个 block。这两行就是这一级的全部改动。
    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, MMA_MNK),
        (WARPS_M, WARPS_N, 1),
    )
    thr_mma = tiled_mma.get_slice(tidx)          # 我这个线程该拿哪一份

    # ③ 累加器:fp32,住在寄存器里。形状由 atom 决定,不用自己算。
    acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((BLOCK_M, BLOCK_N)))
    acc.fill(0.0)

    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    rA = thr_mma.make_fragment_A(tCsA)
    rB = thr_mma.make_fragment_B(tCsB)

    zero = cutlass.BFloat16(0.0)
    num_k = (K + BLOCK_K - 1) // BLOCK_K

    for kt in cutlass.range(num_k, unroll=1):
        k0 = kt * BLOCK_K

        # ④ 搬。最笨的写法:每个线程扫自己的那几个元素,越界补 0。
        #    补 0 而不是跳过 —— 共享内存里的脏值会被 mma 当成真数据乘进去。
        for i in cutlass.range(A_PER_THREAD, unroll=1):
            e = tidx + i * NUM_THREADS
            r = e // BLOCK_K
            c = e % BLOCK_K
            gr = by * BLOCK_M + r
            gk = k0 + c
            if gr < M and gk < K:
                sA[r, c] = mA[gr, gk]
            else:
                sA[r, c] = zero

        for i in cutlass.range(B_PER_THREAD, unroll=1):
            e = tidx + i * NUM_THREADS
            n = e // BLOCK_K
            c = e % BLOCK_K
            gn = bx * BLOCK_N + n
            gk = k0 + c
            if gk < K and gn < N:
                sB[n, c] = mB[gk, gn]        # 注意下标换了位置:(K,N) 读成 (N,K)
            else:
                sB[n, c] = zero

        cute.arch.barrier()

        # ⑤ 共享内存 -> 寄存器 fragment -> 算。搬进 fragment 的布局也是 atom 说了算。
        cute.autovec_copy(tCsA, rA)
        cute.autovec_copy(tCsB, rB)
        cute.gemm(tiled_mma, acc, rA, rB, acc)

        cute.arch.barrier()                  # 算完才能覆盖共享内存

    # ⑥ 写回。用 identity tensor 拿到每个累加器元素对应的全局 (m,n),据此判越界。
    gC = cute.local_tile(mC, (BLOCK_M, BLOCK_N), (by, bx))
    tCgC = thr_mma.partition_C(gC)
    idC = cute.local_tile(cute.make_identity_tensor((M, N)), (BLOCK_M, BLOCK_N), (by, bx))
    tCidC = thr_mma.partition_C(idC)

    for i in cutlass.range(cute.size(acc), unroll=1):
        coord = tCidC[i]
        if coord[0] < M and coord[1] < N:
            tCgC[i] = acc[i].to(cutlass.BFloat16)
