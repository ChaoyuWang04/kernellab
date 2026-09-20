// Hopper:不再手写 —— 改 CUTLASS 的参数
//
// 需要: modal-h100
//
// 前六级是一条完整的手写路线:合并访存 → 共享内存分块 → 寄存器分块 → 双缓冲 →
// WMMA。走到 `5-tensorcore` 就停了,因为**官方的封装到 Ampere 为止**:
// `nvcuda::wmma` 没有 Hopper 版,wgmma 在裸 CUDA 里没有对应的手写接口。
//
// 那 Hopper 上该怎么办?**NVIDIA 自己的答案是:用 CUTLASS。** 这不是偷懒 ——
// 去翻 CUDA 编程指南,Hopper 的 tensor core 那一章指向的就是 CUTLASS,
// 而不是一段该抄的内联汇编。
//
// 所以这一级的练法也换了:**不写 kernel,写配置。**
//
// ## 旋钮在这三行
//
//     using TileShape    = Shape<_128, _128, _64>;   // 一个 threadblock 算多大一块
//     using ClusterShape = Shape<_2, _1, _1>;        // 几个 block 组成一个 cluster(Hopper 新增)
//     using KernelSchedule = KernelScheduleAuto;     // 流水方式:让它自己挑,或点名
//
// 其余全部由 `CollectiveBuilder` 推出来:流水级数、swizzle、warp specialization、
// TMA 描述符、epilogue 怎么写回 —— 这些正是前面五级一件件手写过来的东西,
// 现在变成了「模板参数一填,库替你选」。
//
// **ClusterShape 是 Hopper 才有的东西**:同一个 cluster 里的 block 能互相读对方的
// 共享内存(分布式共享内存),于是 A 或 B 的 tile 可以只搬一份、多个 block 共用。
// 前六级里没有任何一级能表达这个概念 —— 不是我们没写,是 Ampere 上没有。
//
// ## 怎么用这一级
//
// 改上面那三行,Run 一次看数字。几个值得试的方向:
//
//   - TileShape 的 M/N 调大(128×256):每块算得更多,但共享内存和寄存器压力上升
//   - ClusterShape 从 <_1,_1,_1> 改到 <_2,_1,_1> 或 <_4,_2,_1>:看 cluster 能省多少搬运
//   - KernelSchedule 点名 `KernelTmaWarpSpecializedCooperative` / `...Pingpong`,
//     而不是 Auto,对比编译器的默认选择
//
// ## H100 实测:一行 kernel 代码没写,快 11 倍
//
//     case      5-tensorcore(手写 WMMA)   6-cutlass(改参数)      变化
//     4096³      43.72 TF  0.06× torch    482.15 TF  0.63×      11.0×
//     8192³      (未跑)                   486.29 TF  0.65×
//     1000x999x777                         20.92 TF  0.44×
//     smallK                               27.26 TF  0.25×
//     tall                                449.64 TF  0.64×
//     deepK       3.06 TF  0.01× torch     72.14 TF  0.24×      23.6×
//
// klab ptx 的对照说明了这 11 倍从哪来:
//
//     5-tensorcore   世代 Ampere   wmma.mma×8
//     6-cutlass      世代 Hopper   wgmma×14  cp.async.bulk×3  mbarrier×32  setmaxnreg×2
//
// **三样 Hopper 特性一次到齐**:wgmma(异步 tensor core)、TMA(`cp.async.bulk`)、
// warp specialization(`setmaxnreg` —— 搬运 warp 与计算 warp 分家,各自调整寄存器配额)。
// 这三样我们一个都没写,全是 `CollectiveBuilder` 根据那三行模板参数推出来的。
//
// ## 一个绕不开的坑:TMA 要求 16 字节对齐
//
// 1000x999x777 那档 K=777、N=999,行距不是 16 字节的整数倍,直接跑会在 CUTLASS
// 内部断言失败(`gmem_prob_stride[1] & 0b1111`)。**Hopper 上没有非 TMA 的通路** ——
// 把 AlignmentA 降到 1 也没用,CollectiveBuilder 仍然给你 TMA 的 mainloop。
//
// 唯一的办法是 pad(见 matmul_launch 里那段)。代价看得见:这一档只有 20.92 TFLOPS,
// 是六档里最低的 —— 每次调用都要多分配三块显存、拷进去再拷回来。
// **真实推理框架里,模型的隐藏维度被设计成 128 的倍数,一半原因就在这儿。**
//
// ## smallK 为什么也差
//
// K=64,一个 TileShape 的 K 就是 64 —— 整个 GEMM 只有一轮 K 循环,CUTLASS 那套
// 深流水、warp specialization 全没机会展开,纯粹在付启动开销。0.25× torch,
// 而手写的朴素版在这一档反而不算太难看。**工业级实现也不是所有形状都赢。**
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
using TileShape      = Shape<_128, _128, _64>;
using ClusterShape   = Shape<_2, _1, _1>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
// =====================================================================

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
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
        {M, Ng, Kg},
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
