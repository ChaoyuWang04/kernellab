// warp specialization —— 搬运 warp 和计算 warp 分家
//
// 需要: modal-b200
//
// 上一级用上了 tcgen05,B200 上 295 TFLOPS。而同一张卡上 cuBLAS 是 ~1510,
// CUTLASS 改四行参数就有 1388(见 cuda 那一列的 `7-cutlass-blackwell`)。
// **同样一条 tcgen05 指令,差 4.7 倍。** 差距不在指令,在怎么喂它。
//
// 官方的 educational_b200 把补这个差距的过程拆成了三级,这是第一级,也是最大的一跳:
// **731 TFLOPs(官方标注),比 level_06 的 293 快 2.5 倍。**
//
// ## 改了什么
//
// 上一级所有线程干同一件事:一起等数据 → 一起发 mma → 一起等算完。
// 这一级**按 warp 分工**:
//
//     if (warpid == 0 && laneid == 0) {          // 生产者:只管搬
//         for (每个 K 块) { 等这一格空出来; 发 TMA; }
//     }
//     else if (warpid == 1 && laneid == 0) {     // 消费者:只管算
//         for (每个 K 块) { 等这一格到货; 发 mma; }
//     }
//
// 两个循环**各跑各的**,靠每一格的两个信号量咬合:`inputs_arrived[stage]`(搬完了)
// 和 `inputs_finished[stage]`(算完了,可以覆盖)。生产者跑在前面预取,消费者在后面
// 追着算 —— 这就是「流水」两个字的真身。
//
// 配套的一步:**流水从 2 级加到 3 级**(`PIPE_STAGES = 3`)。双缓冲只能让生产者领先
// 一格;三格之后它可以领先两格,TMA 的延迟才藏得住。
//
// ## 为什么是 laneid == 0
//
// TMA 和 tcgen05 的 mma 都**只需要一个线程按按钮**,硬件自己干活。所以生产者、
// 消费者各只用一个线程,剩下 126 个线程在这段循环里什么都不做 —— 它们在等着做尾声
// (把累加器从 tensor memory 取回来、写回显存)。
//
// **这听起来很浪费,但算力本来就不是它们提供的** —— tcgen05 的算力来自 tensor core,
// 不来自 CUDA core。B200 上「谁在算」和「谁在发指令」已经彻底分开了。
//
// ## B200 实测:分工本身值 2.4 倍
//
//     case      3-blackwell(全员一起干)   4-warpspec(分工)      变化
//     4096³       294.98 TF  0.20×         710.32 TF  0.46×     2.41×
//     8192³       289.93 TF  0.17×         698.59 TF  0.42×     2.41×
//     smallK       72.32 TF  0.45×          72.32 TF  0.52×     1.00×  <- 没动
//     tall        235.47 TF  0.21×         489.62 TF  0.47×     2.08×
//     deepK        30.07 TF  0.07×          68.20 TF  0.19×     2.27×
//
// **官方 README 标 level_07 = 731 TFLOPs,我们量到 710 —— 对上了。** 五档全过。
//
// 指令一条没换(还是 tcgen05),分块一个没改。变的只有「谁干什么」:
// 上一级 128 个线程同步走一条直线,这一级两个线程各跑各的循环,靠信号量咬合。
//
// **这就是「用上新硬件」和「喂饱新硬件」之间那道坎的第一段。**
//
// ## smallK 为什么又是一动不动
//
// K=64,`ktiles = 1` —— 生产者循环只跑一次,消费者循环也只跑一次,**根本没有流水
// 可言**。从 `2-hopper-tma` 到这一级,这一档一直卡在 72 TFLOPS 上不动。
//
// 流水、预取、分工,这三样都要「有下一块」才有意义。K 只有一块的时候,它们全部失效,
// 时间被启动与收尾开销吃掉 —— 这是这六档 case 里最稳定的一条规律。
//
// ## 还差两级
//
// 官方阶梯:level_08 是 epilogue 流水(1050 TFLOPs),level_09 是 2-CTA cluster
// 加 warpgroup 级并行(1285)。而 CUTLASS 里这些**本来就全在** ——
// 同一张 B200 上 `cuda/7-cutlass-blackwell` 改四行参数就是 1388 TFLOPS。
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 64;
constexpr int PIPE_STAGES = 3;    // 2 级只能领先一格;3 级之后生产者能领先两格
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N>;
using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;      // tensor memory 里的累加器

__global__ __launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ a_gl gA,
                   const __grid_constant__ b_gl gB,
                   const __grid_constant__ d_gl gC,
                   int M, int N, int K) {
    const int warpid = threadIdx.x / WARP_THREADS;
    const int laneid = threadIdx.x % WARP_THREADS;
    const int wg_laneid = warpgroup::laneid();
    const int grid_n = N / TILE_N;
    const int bid_m = blockIdx.x / grid_n;
    const int bid_n = blockIdx.x % grid_n;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a_tile (&a_smem)[PIPE_STAGES] = al.allocate<a_tile, PIPE_STAGES>();
    b_tile (&b_smem)[PIPE_STAGES] = al.allocate<b_tile, PIPE_STAGES>();
    d_tile (&d_smem) = al.allocate<d_tile>();

    // 每一格流水各有一对信号量:搬完了 / 算完了可以覆盖
    __shared__ semaphore inputs_arrived[PIPE_STAGES], inputs_finished[PIPE_STAGES];
    __shared__ semaphore compute_done;
    if (threadIdx.x == 0) {
        for (int i = 0; i < PIPE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 1, 0);
        }
        init_semaphore(compute_done, 0, 1);
    }
    __syncthreads();

    // 在 tensor memory 里开累加器
    tensor_allocator<1, 1> tm_alloc{};
    d_tt_t accum;
    if (wg_laneid == 0) {
        accum = tm_alloc.allocate<d_tt_t>(0);
    }
    warpgroup::sync(1);

    const int ktiles = K / TILE_K;
    int phase = 0;

    if (warpid == 0 && laneid == 0) {
        // 生产者:只管搬。等这一格被消费者用完,就把下一块 TMA 进来。
        for (int kt = 0; kt < ktiles; ++kt) {
            const int stage = kt % PIPE_STAGES;
            wait(inputs_finished[stage], phase ^ 1);
            if (stage == PIPE_STAGES - 1) phase ^= 1;
            tma::expect_bytes(inputs_arrived[stage], sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(a_smem[stage], gA, {bid_m, kt}, inputs_arrived[stage]);
            tma::load_async(b_smem[stage], gB, {kt, bid_n}, inputs_arrived[stage]);
        }
    } else if (warpid == 1 && laneid == 0) {
        // 消费者:只管算。等这一格到货就发 mma;mma 算完自己去敲 inputs_finished。
        for (int kt = 0; kt < ktiles; ++kt) {
            const int stage = kt % PIPE_STAGES;
            wait(inputs_arrived[stage], phase);
            if (stage == PIPE_STAGES - 1) phase ^= 1;
            if (kt == 0) mm_AB(accum, a_smem[stage], b_smem[stage], inputs_finished[stage]);
            else         mma_AB(accum, a_smem[stage], b_smem[stage], inputs_finished[stage]);
        }
        detail::tcgen05::commit<1>(compute_done);
    }

    wait(compute_done, 0);

    // 把累加器从 tensor memory 取回寄存器,再经共享内存 TMA 写回显存
    rt_bf<TILE_M / 4, TILE_N> d_reg;
    warpgroup::load_async(d_reg, accum);
    tensor_load_wait();
    warpgroup::sync(1);
    warpgroup::store(d_smem, d_reg);
    warpgroup::sync(1);
    if (wg_laneid == 0) {
        tma::store_async(gC, d_smem, {bid_m, bid_n});
        tma::store_async_read_wait();
    }
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    a_gl gA = make_gl<a_gl>(reinterpret_cast<uint64_t>(A), 1, 1, M, K);
    b_gl gB = make_gl<b_gl>(reinterpret_cast<uint64_t>(B), 1, 1, K, N);
    d_gl gC = make_gl<d_gl>(reinterpret_cast<uint64_t>(C), 1, 1, M, N);
    int grid = (M / TILE_M) * (N / TILE_N);
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_kernel<<<grid, NUM_THREADS, smem, stream>>>(gA, gB, gC, M, N, K);
}
