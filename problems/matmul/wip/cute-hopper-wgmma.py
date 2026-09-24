"""【未完成 —— 不在阶梯里,面板看不到】cute × Hopper:warp -> warpgroup(wgmma)

**状态(2026-09-20):编得过、跑得通、结果错 13%。** 按仓库规矩不上架。
放在 wip/ 是因为剩下的部分都验证过了,下次接手不用从零开始。

## 已经验证正确的(别重复查)

- **共享内存内容**:kernel 里 printf 抽查 sA0[r,r]、sA0[0,17]、sB0[70,9],与全局
  逐个吻合。搬运是对的。
- **wgmma 的计算结果**:线程 0 的 acc[0..3] 与 kernel 内标量手算 dot 完全一致
  (15.202614 vs 15.202615)。**MMA 本身没问题。**
- **累加器到 C 的坐标映射**:线程 0 的 64 个坐标是标准 wgmma 布局
  —— 行 {0,8} × 列 {0,1,8,9,...,120,121},互不重叠。
- **写回覆盖率**:把 epilogue 换成写常数 1.0,主机侧统计 **0 / 16777216 未写到**。
  所以「有一部分 C 没被写、读到 torch.empty 残留」这个假设是错的。
- **共享内存覆盖**:cosize(sA0)=4096 = 拷贝写入 32×128;cosize(sB0)=8192 = 64×128。
  逻辑上没有空洞。
- swizzle 确实挂到了指针上:`ptr<bf16, smem, align<1024>, S<3,4,3>>`。

## 还不知道为什么

- 错的元素占 **13.1%**,不是全错。每改一次代码错误值就变(13.89 / 15.91 / 18.88 /
  20.52),但同一份代码是确定的。
- `smallK`(K=64,只有一轮 K 循环)也错 —— 所以不是跨 K 累加的问题。
- 把两处 `fill(0.0)` 去掉,4096³ 直接变 NaN。**这说明 wgmma 读的共享内存区域比拷贝
  覆盖的大**,但上面的 cosize 又说没有空洞 —— 这两条对不上,是下一步的突破口。
- N=128 换成 N=64(避开描述符把 N 切成两块)也没用。

## 最可能的原因(2026-09-24 提出,**还没验过**)

**swizzle 只在一侧生效。**

1. wgmma 要求 swizzle 挂在**指针**上、layout 保持仿射(编译器会直接报
   `Expected affine layout ... use recast_ptr`),所以下面把 `make_smem_layout_a/b`
   返回的 ComposedLayout 拆成了 `allocate_tensor(dtype, lA.outer, swizzle=lA.inner)`。
2. 写 `sA0[m,k]` 时地址是 `swizzle(layout(m,k))`;而 wgmma 的描述符**自己在硬件里
   编码了一个 swizzle 模式**。两者若不一致(比如 `partition_D` 组合时把它吃掉了,
   或描述符是从**剥掉 swizzle 的那个 layout** 推出来的),两边就各读各的排列。
3. **swizzle 本质是对几个地址位做 XOR —— 那几位本来是 0 时,XOR 是空操作。**
   这解释了一个当时看着矛盾的现象:手算验证过的 acc[0..3] 对应
   C[0,0]/C[0,1]/C[8,0]/C[8,1],**全在 tile 左上角,正是 XOR 不生效的区域**。
   「抽查对了」和「整体错 13%」因此不冲突,反而互证。
4. 它同时能解释下面那条唯一对不上的矛盾:去掉 fill 就 NaN(wgmma 读到了没写过的地方),
   而 cosize 又说没有空洞 —— 两边地址函数不同的话,这两条同时成立。

**怎么验**:把完整的 ComposedLayout 直接交给拷贝的目的端(而不是拆开后的 `.outer`),
看错误率变不变。一次跑就能证伪 —— 变了说明方向对,一点不变就排除掉、把结论补写在这里。

## 下次接手的建议

1. 先验上面那个 swizzle 假设。它没中的话,再去解释「去掉 fill 就 NaN」与
   「cosize 没空洞」的矛盾 —— 多半仍在 `make_tiled_copy_tv` 的 partition_D
   在 swizzle 布局上的行为。
2. 找一份官方的 CuTe DSL Hopper dense GEMM 示例逐行比对。pip 包里不带示例,
   要去 NVIDIA/cutlass 仓库的 examples/python 下找。
3. 已经确认的 API 用法都在下面,照抄即可:
   - `hopper_helpers.make_trivial_tiled_mma(...)` 建 TiledMma
   - `make_smem_layout_a/b` 返回 ComposedLayout,要拆成
     `allocate_tensor(dtype, lA.outer, swizzle=lA.inner)` —— 否则报
     「Expected affine layout ... use recast_ptr」
   - 两个切片别混用:`get_slice(0)` 给 wgmma 的共享内存描述符,
     `get_slice(tidx)` 给累加器和写回。混用会让 128 个线程全往线程 0 的位置写。
   - 协议四步:`fence()` / `cute.gemm` / `commit_group()` / `wait_group(0)`
   - ACCUMULATE 字段:`tiled_mma.set(warpgroup.Field.ACCUMULATE, False/True)`
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode, warpgroup
from cutlass.cute.nvgpu.warpgroup import OperandSource
from cutlass.utils import SmemAllocator
from cutlass.utils.hopper_helpers import make_smem_layout_a, make_smem_layout_b, make_trivial_tiled_mma
from cutlass.utils.layout import LayoutEnum

# wgmma 的 M 固定 64;N 取 128;K 方向一次吃 64
BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
MMA_TILER = (BLOCK_M, BLOCK_N, BLOCK_K)

NUM_THREADS = 128                     # 一个 warpgroup = 4 个 warp
BLOCK_THREADS = (NUM_THREADS, 1, 1)

# A 是 (M,K) 行主序 -> K 连续 -> K-major;B 是 (K,N) 行主序 -> N 连续 -> MN-major
A_LAYOUT = LayoutEnum.ROW_MAJOR
B_LAYOUT = LayoutEnum.COL_MAJOR       # 对 B 而言 COL_MAJOR 就是「N 连续」


@cute.kernel
def matmul_kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                  M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()

    # ① 挑 Hopper 的积木。官方 helper 负责把 op 包成 atom 再铺成 TiledMma。
    tiled_mma = make_trivial_tiled_mma(
        cutlass.BFloat16, cutlass.BFloat16,
        A_LAYOUT.mma_major_mode(), OperandMajorMode.MN,
        cutlass.Float32, (1, 1, 1), (BLOCK_M, BLOCK_N),
        a_source=OperandSource.SMEM,          # 操作数走共享内存描述符,不进寄存器
    )

    # ② 共享内存的摆法必须由官方 helper 生成 —— 描述符里编了 swizzle。
    #    helper 返回的是 ComposedLayout(swizzle ∘ 仿射 layout),而 wgmma 要求
    #    swizzle 挂在指针上、layout 保持仿射,所以这里把两半拆开分别传。
    smem = SmemAllocator()
    lA = make_smem_layout_a(A_LAYOUT, MMA_TILER, cutlass.BFloat16, 1)
    lB = make_smem_layout_b(B_LAYOUT, MMA_TILER, cutlass.BFloat16, 1)
    sA = smem.allocate_tensor(cutlass.BFloat16, lA.outer, byte_alignment=1024, swizzle=lA.inner)
    sB = smem.allocate_tensor(cutlass.BFloat16, lB.outer, byte_alignment=1024, swizzle=lB.inner)
    sA0 = sA[None, None, 0]               # num_stages=1,把那一维切掉
    sB0 = sB[None, None, 0]

    # 两个切片,别混用:
    #   wg_mma  —— wgmma 的操作数是 warpgroup 级的共享内存描述符,与线程号无关,取 0
    #   thr_mma —— 累加器和写回是**每个线程各一份**,必须用真实的 tidx
    wg_mma = tiled_mma.get_slice(0)
    thr_mma = tiled_mma.get_slice(tidx)
    acc = thr_mma.make_fragment_C(thr_mma.partition_shape_C((BLOCK_M, BLOCK_N)))
    tCrA = tiled_mma.make_fragment_A(wg_mma.partition_A(sA0))
    tCrB = tiled_mma.make_fragment_B(wg_mma.partition_B(sB0))
    # wgmma 的规范写法:第一轮覆盖(acc = A·B),之后每轮累加。比「预先清零 + 一直累加」
    # 可靠 —— 不依赖累加器的初值,少一次对 64 个寄存器的写。
    tiled_mma.set(warpgroup.Field.ACCUMULATE, False)

    # ③ 搬运。B 不用转置了:global 与 smem 两边都是 N 连续,直接把 mB 看成 (N,K)。
    mBt = cute.make_tensor(mB.iterator, cute.make_layout((N, K), stride=(1, N)))

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16)
    #   A 块 (64,64):8 个线程管一行的 64 个 K(每人 8 个),16 行一轮,4 轮盖满
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom,
                                           cute.make_layout((16, 8), stride=(8, 1)),
                                           cute.make_layout((4, 8)))
    #   B 块 (128,64):mBt 里 N 才是连续维,所以连续线程沿 N 走(stride 反过来)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom,
                                           cute.make_layout((16, 8), stride=(1, 16)),
                                           cute.make_layout((8, 8)))
    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    tAsA = thr_copy_A.partition_D(sA0)
    tBsB = thr_copy_B.partition_D(sB0)

    idA = cute.make_identity_tensor((M, K))
    idB = cute.make_identity_tensor((N, K))

    tAsA.fill(0.0)
    tBsB.fill(0.0)

    num_k = (K + BLOCK_K - 1) // BLOCK_K

    for kt in cutlass.range(num_k, unroll=1):
        tAgA = thr_copy_A.partition_S(cute.local_tile(mA, (BLOCK_M, BLOCK_K), (by, kt)))
        tBgB = thr_copy_B.partition_S(cute.local_tile(mBt, (BLOCK_N, BLOCK_K), (bx, kt)))
        cA = thr_copy_A.partition_S(cute.local_tile(idA, (BLOCK_M, BLOCK_K), (by, kt)))
        cB = thr_copy_B.partition_S(cute.local_tile(idB, (BLOCK_N, BLOCK_K), (bx, kt)))
        pA = cute.make_fragment_like(tAgA, cutlass.Boolean)
        pB = cute.make_fragment_like(tBgB, cutlass.Boolean)
        for i in cutlass.range(cute.size(pA), unroll=1):
            pA[i] = cA[i][0] < M and cA[i][1] < K
        for i in cutlass.range(cute.size(pB), unroll=1):
            pB[i] = cB[i][0] < N and cB[i][1] < K

        if kt == num_k - 1:
            tAsA.fill(0.0)
            tBsB.fill(0.0)

        cute.copy(tiled_copy_A, tAgA, tAsA, pred=pA)
        cute.copy(tiled_copy_B, tBgB, tBsB, pred=pB)
        cute.arch.barrier()

        # ④ wgmma 的四步协议。少任何一步都会算错或者挂住。
        warpgroup.fence()
        cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        tiled_mma.set(warpgroup.Field.ACCUMULATE, True)      # 第一轮之后改成累加

        cute.arch.barrier()

    tCgC = thr_mma.partition_C(cute.local_tile(mC, (BLOCK_M, BLOCK_N), (by, bx)))
    tCidC = thr_mma.partition_C(
        cute.local_tile(cute.make_identity_tensor((M, N)), (BLOCK_M, BLOCK_N), (by, bx)))
    for i in cutlass.range(cute.size(acc), unroll=1):
        coord = tCidC[i]
        if coord[0] < M and coord[1] < N:
            tCgC[i] = acc[i].to(cutlass.BFloat16)