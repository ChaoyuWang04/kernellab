// 再换一代 —— 同一份 CUTLASS 代码,把 Sm90 改成 Sm100
//
// 需要: modal-b200
//
// 上一级用三行模板参数在 H100 上拿到了 482 TFLOPS。这一级验证那条路线的**可移植性**:
// 换到 B200,要改的是不是还只有那几行?
//
// 答案基本是。逐字比 `6-cutlass-hopper.cu`,差异只有四处:
//
//     cutlass::arch::Sm90          ->  cutlass::arch::Sm100
//     TileShape  <_128,_128,_64>   ->  <_256,_128,_64>     // tile 大一倍
//     ClusterShape <_2,_1,_1>      ->  <_2,_2,_1>
//     Shape<int,int,int>           ->  Shape<int,int,int,int> + void
//
// 最后一处值得说:**Blackwell 的 ProblemShape 多了一维**,而末尾那个 `void` 表示
// 用默认的 CLC(Cluster Launch Control)tile scheduler —— 硬件帮你把 block 派发到
// cluster 上。这是 Hopper 上没有的东西,也是「换代不只是换指令」的又一个例子。
//
// wgmma → tcgen05、tmem、更大的 tile、CLC 调度,这四样我们一行都没写。
//
// ## B200 实测:四行参数,基本追平 cuBLAS
//
//     case      5-tensorcore(手写 WMMA)   7-cutlass(改四行)     vs torch
//     4096³       52.37 TF  0.03×         1387.71 TF  0.92×     26.5×
//     8192³       (未跑)                  1199.71 TF  0.93×
//     tall        (未跑)                  1001.62 TF  0.88×
//     smallK      (未跑)                    99.86 TF  0.71×
//     deepK       (未跑)                   164.48 TF  0.41×
//     1000x999x777(未跑)                    12.90 TF  0.30×   <- pad 的代价
//
// **0.92× cuBLAS。** 六档全过。`klab ptx`:`tcgen05×14 + cp.async.bulk×21 + mbarrier×124`,
// PTX 十二万八千行 —— 全是 CollectiveBuilder 根据那四行推出来的。
//
// ## 和我们自己抄的 tcgen05 比一比
//
// 同一张 B200,同一代指令:
//
//     tk/3-blackwell-tcgen05(照抄 TK 官方 level_06)    295 TFLOPS
//     cuda/7-cutlass-blackwell(改四行参数)            1388 TFLOPS     4.7×
//
// **同样用上了 tcgen05,差 4.7 倍。** 差距不在指令,在怎么喂它 —— warp specialization、
// epilogue 流水、cluster 级并行,这些 TK 的官方阶梯要到 level_07/08/09 才逐级加上,
// 而 CUTLASS 里**本来就全在**。
//
// 这就是「用上新硬件」和「喂饱新硬件」那四倍差距的具体去处。想看它是一级一级怎么
// 补起来的,去读 tk 那一列;想直接拿到结果,就是这一级。**两条路都该走一遍:**
// 一条告诉你差距由什么构成,一条告诉你工业级实现长什么样。
//
// ## 两档不好看的
//
// - `1000x999x777` 只有 0.30× —— TMA 要求 16 字节对齐,这一档只能 pad 着算,
//   多分配三块显存、拷进拷出。真实框架把隐藏维度设成 128 的倍数,一半原因在这儿。
// - `smallK`(K=64)0.71× —— 一个 TileShape 的 K 就是 64,整个 GEMM 只有一轮 K 循环,
//   CUTLASS 那套深流水全没机会展开。**工业级实现也不是所有形状都赢。**
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

// 三个矩阵都是行主序(和前六级、和 torch 一致)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

// 对齐:TMA 要求 16 字节对齐,bf16 就是 8 个元素
// TMA 一次搬 16 字节,bf16 就是 8 个元素。这个数和下面的 pad 是同一件事的两半。
constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int AlignmentC = 8;

// ====================== 这三行就是这一级的全部内容 ======================
using TileShape      = Shape<_256, _128, _64>;   // Blackwell 的 tile 比 Hopper 大一倍
using ClusterShape   = Shape<_2, _2, _1>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
// =====================================================================

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

// Blackwell 的 ProblemShape 是四元的(多一个 batch 维),末尾的 void 表示用默认的
// CLC(Cluster Launch Control)tile scheduler —— 又一样 Hopper 上没有的东西
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

static inline int round_up8(int x) { return (x + 7) / 8 * 8; }

// TMA 硬性要求全局内存的行 stride 是 16 字节(bf16 = 8 个元素)的整数倍。
// 1000x999x777 那档 K=777、N=999 都不满足,直接跑会在 CUTLASS 内部断言失败:
//     Assertion `(gmem_prob_stride[1] & 0b1111) == 0' failed
//
// 真实代码遇到这种形状只有一条路:**pad 到对齐,算完再拷回来**。
// 这里把 K 和 N 都补齐到 8 的倍数,补出来的部分清零 —— K 方向补的零乘进去是 0,
// 不影响结果;N 方向多算出来的几列直接不拷回。
void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;

    const int Kp = round_up8(K);
    const int Np = round_up8(N);
    const bool aligned = (Kp == K) && (Np == N);

    const ElementA* pA = reinterpret_cast<const ElementA*>(A);
    const ElementB* pB = reinterpret_cast<const ElementB*>(B);
    ElementC* pC = reinterpret_cast<ElementC*>(C);
    void *bufA = nullptr, *bufB = nullptr, *bufC = nullptr;

    if (!aligned) {
        const size_t nA = (size_t)M * Kp, nB = (size_t)Kp * Np, nC = (size_t)M * Np;
        cudaMallocAsync(&bufA, nA * sizeof(ElementA), stream);
        cudaMallocAsync(&bufB, nB * sizeof(ElementB), stream);
        cudaMallocAsync(&bufC, nC * sizeof(ElementC), stream);
        cudaMemsetAsync(bufA, 0, nA * sizeof(ElementA), stream);   // 补出来的位置必须是 0
        cudaMemsetAsync(bufB, 0, nB * sizeof(ElementB), stream);
        cudaMemcpy2DAsync(bufA, (size_t)Kp * sizeof(ElementA), A, (size_t)K * sizeof(ElementA),
                          (size_t)K * sizeof(ElementA), M, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(bufB, (size_t)Np * sizeof(ElementB), B, (size_t)N * sizeof(ElementB),
                          (size_t)N * sizeof(ElementB), K, cudaMemcpyDeviceToDevice, stream);
        pA = reinterpret_cast<const ElementA*>(bufA);
        pB = reinterpret_cast<const ElementB*>(bufB);
        pC = reinterpret_cast<ElementC*>(bufC);
    }

    // 逻辑形状也一起 pad,于是 extent 与 stride 自洽,用的还是最普通的那套公式
    const int Ng = aligned ? N : Np;
    const int Kg = aligned ? K : Kp;
    auto sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, Kg, 1));
    auto sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(Ng, Kg, 1));
    auto sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, Ng, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, Ng, Kg, 1},
        {pA, sA, pB, sB},
        {{1.0f, 0.0f},                       // epilogue: C = alpha*(A@B) + beta*C
         nullptr, sC, pC, sC}};

    Gemm gemm;
    gemm.initialize(args, nullptr, stream);   // 这个配置不需要 workspace
    gemm.run(stream);

    if (!aligned) {
        cudaMemcpy2DAsync(C, (size_t)N * sizeof(ElementC), bufC, (size_t)Np * sizeof(ElementC),
                          (size_t)N * sizeof(ElementC), M, cudaMemcpyDeviceToDevice, stream);
        cudaFreeAsync(bufA, stream);
        cudaFreeAsync(bufB, stream);
        cudaFreeAsync(bufC, stream);
    }
}
