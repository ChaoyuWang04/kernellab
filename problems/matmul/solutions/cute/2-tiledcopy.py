"""搬也有 atom —— 把逐元素循环换成 TiledCopy

上一级换了「算」的积木,体检单立刻说话:判定不是「卡在算上」,而是**「两头都没跑满」**,
tensor core 只有 15%,warp 的等待里 76% 在等显存。**tensor core 闲着等饭吃。**

为什么等?看上一级搬 B 的那两行:

    n = e // BLOCK_K        # 线程号变,n 变得慢
    c = e % BLOCK_K         # 线程号变,c(K 方向)变得快
    sB[n, c] = mB[gk, gn]   # 相邻线程读 mB[k, n] 与 mB[k+1, n] —— 地址相隔 N 个元素

相邻线程读的地址隔着一整行,**完全没法合并**。这和裸 CUDA 第 0 级犯的是同一个错,
只是藏在了 CuTe 的外衣底下。

CuTe 对「怎么搬」的回答和「怎么算」一样 —— **也是一块 atom**,外面套一层 TiledCopy
说明「哪个线程拿哪几个元素」:

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16)
    tiled     = cute.make_tiled_copy_tv(copy_atom, 线程怎么排, 每人拿几个)
    cute.copy(tiled, tAgA, tAsA, pred=tApA)

你不再算下标,只声明**线程布局**和**值布局**。上一级的 `make_tiled_mma` 是同一套
东西换个名字 —— 这就是 CuTe 的全部主张:把「谁拿哪块数据」写成可组合的类型。

## 一个绕不开的不对称:A 天生顺手,B 要转置

mma 的 atom 要求 A、B 都 **K 方向连续**。A 本来就是 (M,K) 行主序,K 连续。B 是
(K,N) 行主序,**N 才连续**,共享内存里却要摆成 (N,K)。

所以 B 走一个转置视图 —— 同一块内存,两个 layout:

    sB  = (BLOCK_N, BLOCK_K) stride (BLOCK_K, 1)   # mma 看到的
    sBt = (BLOCK_K, BLOCK_N) stride (1, BLOCK_K)   # 搬运看到的,和 gB 同形

于是全局那边可以合并读(连续线程吃连续的 N),跳跃被挪到了共享内存那边。
**转置的代价躲不掉,能选的只是让它落在哪儿** —— 显存一次跳跃几百个周期,
共享内存一次 bank 冲突几个周期。

## 5090 实测

    case            1-atom              2-tiledcopy          提升
    4096³            3.74 ms  36.7 TF    3.30 ms  41.7 TF    1.14×
    8192³           34.39 ms  32.0 TF   23.52 ms  46.6 TF    1.46×
    1000x999x777     0.27 ms   5.8 TF    0.13 ms  11.3 TF    2.06×
    smallK           0.17 ms  12.6 TF    0.17 ms  12.6 TF    1.00×   ← 没动
    tall             2.57 ms  26.8 TF    1.87 ms  36.6 TF    1.37×
    deepK            5.27 ms   1.6 TF    2.41 ms   3.6 TF    2.19×

**合并访存确实修好了** —— 体检单里「等显存把数据取回来」从 22.72 掉到 6.58,**降了 3.5 倍**。

**但整体只快 14%(4096³)。** 因为瓶颈换了个地方,判定从「两头都没跑满」变成
**「偏搬数据这一侧」**:

    L1 缓存        56%  →  82%     (命中率 87% → 72%)
    L2 缓存        14%  →  54%
    显存            1%  →   2%     ← 一直很闲,从来不是它的问题
    访存指令排不进队  7%  →  30%     (NCU 叫 mio_throttle)

**显存一直闲着。** 真正的墙是共享内存那一侧:每轮 K 都要把整块 tile 搬进共享内存、
再逐个读进寄存器 fragment,访存指令把队列塞满了。8192³ 那档判定已经直接变成
「卡在搬数据上,L1 86%」。

smallK 完全没动也说得通:K 只有 64,一共才 2 轮 K 循环,时间全花在首尾开销上,
搬运快不快无所谓。

## 这一级最值钱的发现:向量化和逐元素谓词是冲突的

PTX 打出来是 `ld.global.b16 × 32` —— **一次只搬 16 位,根本没向量化**。
`make_copy_atom` 不给 `num_bits_per_copy` 时那句「编译器尽力自动向量化」没有生效。

显式给 `num_bits_per_copy=128` 呢?编译直接失败:

    'cute.copy' op expects pred to have compatible shape with: (2,(1,1))
    but got actual predShape: (((2,4),2),(1,1))

**一旦向量化,谓词的粒度就从「每个元素」变成「每个向量」。** K=777 时最后一块
K 只有 9 个有效元素,而一个向量是 8 个 —— 第二个向量正好跨在边界上,一刀切
必然算错。

这不是 CuTe 的毛病,是所有向量化访存的共同约束。CUTLASS 自己的解法是把主循环和
**尾块(residue)拆成两条路**:主循环全程满载、向量化拉满,尾块单独用逐元素的慢路
处理。那是工业级实现的标准结构,也是这条阶梯下一步该走的方向 —— 但它已经不是
「改几行看数字怎么变」了,所以这一级停在这里,把约束讲清楚。
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp
from cutlass.utils import SmemAllocator

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

WARPS_M = 2
WARPS_N = 4
NUM_THREADS = WARPS_M * WARPS_N * 32          # 256
BLOCK_THREADS = (NUM_THREADS, 1, 1)

MMA_MNK = (16, 8, 16)


@cute.kernel
def matmul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                  M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    smem = SmemAllocator()
    sA = smem.allocate_tensor(cutlass.BFloat16,
                              cute.make_layout((BLOCK_M, BLOCK_K), stride=(BLOCK_K, 1)),
                              byte_alignment=16)
    sB = smem.allocate_tensor(cutlass.BFloat16,
                              cute.make_layout((BLOCK_N, BLOCK_K), stride=(BLOCK_K, 1)),
                              byte_alignment=16)
    # 同一块内存的转置视图:搬运按 (K,N) 看它,mma 按 (N,K) 看它
    sBt = cute.make_tensor(sB.iterator,
                           cute.make_layout((BLOCK_K, BLOCK_N), stride=(1, BLOCK_K)))

    # ① 算的 atom(和上一级一样,没动)
    tiled_mma = cute.make_tiled_mma(
        warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, MMA_MNK),
        (WARPS_M, WARPS_N, 1),
    )
    thr_mma = tiled_mma.get_slice(tidx)
    acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((BLOCK_M, BLOCK_N)))
    acc.fill(0.0)
    rA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
    rB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))

    # ② 搬的 atom。线程布局的最后一维走内存连续的那一维,合并访存就是这么来的。
    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
    #   A 块 (128,32):4 个线程管一行的 32 个 K(每人 8 个 = 128 bit),64 行一轮,2 轮盖满
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom,
                                           cute.make_layout((64, 4), stride=(4, 1)),
                                           cute.make_layout((2, 8)))
    #   B 块 (32,128):32 个线程管一行的 128 个 N(每人 4 个),8 行一轮,4 轮盖满
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom,
                                           cute.make_layout((8, 32), stride=(32, 1)),
                                           cute.make_layout((4, 4)))
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)

    tAsA = thr_copy_A.partition_D(sA)
    tBsB = thr_copy_B.partition_D(sBt)

    # identity tensor:每个位置装着它自己的全局坐标,用来判越界
    idA = cute.make_identity_tensor((M, K))
    idB = cute.make_identity_tensor((K, N))

    # ③ 先把共享内存清零。M / N 边上那几列永远搬不到东西,不清零就是脏数据被乘进去。
    tAsA.fill(0.0)
    tBsB.fill(0.0)

    num_k = (K + BLOCK_K - 1) // BLOCK_K

    for kt in cutlass.range(num_k, unroll=1):
        gA = cute.local_tile(mA, (BLOCK_M, BLOCK_K), (by, kt))
        gB = cute.local_tile(mB, (BLOCK_K, BLOCK_N), (kt, bx))
        tAgA = thr_copy_A.partition_S(gA)
        tBgB = thr_copy_B.partition_S(gB)

        cA = thr_copy_A.partition_S(cute.local_tile(idA, (BLOCK_M, BLOCK_K), (by, kt)))
        cB = thr_copy_B.partition_S(cute.local_tile(idB, (BLOCK_K, BLOCK_N), (kt, bx)))
        pA = cute.make_fragment_like(tAgA, cutlass.Boolean)
        pB = cute.make_fragment_like(tBgB, cutlass.Boolean)
        for i in cutlass.range(cute.size(pA), unroll=1):
            pA[i] = cA[i][0] < M and cA[i][1] < K
        for i in cutlass.range(cute.size(pB), unroll=1):
            pB[i] = cB[i][0] < K and cB[i][1] < N

        # K 不是 BLOCK_K 整数倍时,最后一块的尾巴搬不到;上一轮的数据还留在那儿,要擦掉
        if kt == num_k - 1:
            tAsA.fill(0.0)
            tBsB.fill(0.0)

        cute.copy(tiled_copy_A, tAgA, tAsA, pred=pA)
        cute.copy(tiled_copy_B, tBgB, tBsB, pred=pB)
        cute.arch.barrier()

        cute.autovec_copy(thr_mma.partition_A(sA), rA)
        cute.autovec_copy(thr_mma.partition_B(sB), rB)
        cute.gemm(tiled_mma, acc, rA, rB, acc)
        cute.arch.barrier()

    gC = cute.local_tile(mC, (BLOCK_M, BLOCK_N), (by, bx))
    tCgC = thr_mma.partition_C(gC)
    tCidC = thr_mma.partition_C(
        cute.local_tile(cute.make_identity_tensor((M, N)), (BLOCK_M, BLOCK_N), (by, bx)))
    for i in cutlass.range(cute.size(acc), unroll=1):
        coord = tCidC[i]
        if coord[0] < M and coord[1] < N:
            tCgC[i] = acc[i].to(cutlass.BFloat16)
