// 再换一代 —— tcgen05,累加器搬出寄存器,住进 tensor memory
//
// 需要: modal-b200
//
// 前两级的路线是「换命名空间」:`warp::` → `warpgroup::`。到了数据中心 Blackwell,
// 换的东西更多一层 —— **累加器不再住在寄存器里**。
//
//     Ampere / Hopper:  rt_fl<16, BN> acc;                    // 寄存器 tile
//     Blackwell:        tt<float, TILE_M, TILE_N> accum;      // tensor memory(tmem)
//
// tmem 是 Blackwell 新增的一块专供 tensor core 的存储。累加器搬进去之后,寄存器
// 全让给别的用途,一个 block 能吃下的分块也就跟着变大(这里 128×128,前两级是 64×64)。
//
// ## 协议又长了一截
//
// Ampere 一句 `warp::mma_AB` 就完;Hopper 要 `mma_async_wait`;到 Blackwell:
//
//     mm_AB (accum, a_smem, b_smem, inputs_finished);   // 第一轮:覆盖
//     mma_AB(accum, a_smem, b_smem, inputs_finished);   // 之后:累加
//     ...
//     detail::tcgen05::commit<1>(compute_done);         // 全部发完
//     wait(compute_done, 0);                            // 等算完
//     warpgroup::load_async(d_reg, accum);              // 从 tmem 取回寄存器
//     tensor_load_wait();                               // 取回也是异步的
//
// 还要三个信号量而不是一个:`inputs_arrived`(TMA 搬完了)、`inputs_finished`
// (tensor core 用完这块了,可以覆盖)、`compute_done`(全部算完)。
// **注意 mma 指令自己收一个信号量参数** —— 它算完会去敲,搬运方等这个才能覆盖共享内存。
//
// 而且 **mma 由单个线程发起**(`wg_laneid == 0`),和 TMA 一样。到这一代,
// 「搬」和「算」都变成了「一个线程按一下按钮,硬件自己干」。
//
// ## 这些都不是我们写的
//
// 整套协议照抄 `envs/tk/ThunderKittens/kernels/gemm/educational_b200/level_06.cu`。
// TK 自带 level_01..09 的 Blackwell 阶梯,官方标注的成绩是:level_06 = 293 TFLOPs,
// level_07(warp specialization)= 731,level_08(epilogue 流水)= 1050,
// level_09(2-CTA cluster)= 1285。我们停在 level_06 —— 再往上是工业级实现的形态,
// 不是「改几行看数字怎么变」了。
//
// ## B200 实测
//
//     case      0-tiles(mma.sync)   3-blackwell(tcgen05)    变化
//     4096³     170.98 TFLOPS       294.98 TFLOPS          1.73×
//     8192³     (未跑)              289.93 TFLOPS
//     smallK    (未跑)               72.32 TFLOPS
//     tall      (未跑)              235.47 TFLOPS
//     deepK      23.63 TFLOPS        30.07 TFLOPS          1.27×
//
// **294.98 TFLOPS 和官方 README 标的「Level 06 (293 TFLOPs)」几乎一模一样** ——
// 照抄对了。五档全过。
//
// klab ptx:`tcgen05×46`、`cp.async.bulk×13`、`mbarrier×19`,PTX 从 1376 行涨到 7219 行。
// 判定是「Blackwell + Hopper」—— tcgen05 是 Blackwell 的标志,TMA(cp.async.bulk)
// 是 Hopper 就有的,两代特性叠着用。
//
// ## 但相对 torch 只有 0.20×
//
// B200 上 cuBLAS 跑 4096³ 只要 0.0911 ms,**约 1510 TFLOPS,几乎顶满这张卡的峰值**。
// 我们的 295 只是它的五分之一。官方阶梯剩下的三级就是补这个差距的:
//
//     level_07  warp specialization(搬运 warp 与计算 warp 分家)    731 TFLOPs
//     level_08  epilogue 流水                                      1050 TFLOPs
//     level_09  2-CTA cluster + warpgroup 级并行                   1285 TFLOPs
//
// **一条指令换代只值 1.7 倍;把这条指令喂饱值 4 倍以上。** 这和裸 CUDA 阶梯
// 第 5 级学到的是同一课,只是换了一代硬件重演一遍:
// 「用上 tensor core」和「喂饱 tensor core」永远是两件事。
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int TILE_M = 128;       // 累加器搬进 tmem 之后,分块能开到 128×128
constexpr int TILE_N = 128;
constexpr int TILE_K = 64;
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
    const int wg_laneid = warpgroup::laneid();
    const int grid_n = N / TILE_N;
    const int bid_m = blockIdx.x / grid_n;
    const int bid_n = blockIdx.x % grid_n;

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    a_tile (&a_smem) = al.allocate<a_tile>();
    b_tile (&b_smem) = al.allocate<b_tile>();
    d_tile (&d_smem) = al.allocate<d_tile>();

    // 三个信号量:数据到了 / tensor core 用完了 / 全部算完了
    __shared__ semaphore inputs_arrived, inputs_finished, compute_done;
    if (threadIdx.x == 0) {
        init_semaphore(inputs_arrived, 0, 1);
        init_semaphore(inputs_finished, 1, 0);
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

    for (int kt = 0; kt < ktiles; ++kt) {
        if (threadIdx.x == 0) {
            wait(inputs_finished, phase ^ 1);      // 上一块 tensor core 用完了才能覆盖
            tma::expect_bytes(inputs_arrived, sizeof(a_tile) + sizeof(b_tile));
            tma::load_async(a_smem, gA, {bid_m, kt}, inputs_arrived);
            tma::load_async(b_smem, gB, {kt, bid_n}, inputs_arrived);
        }
        wait(inputs_arrived, phase);
        phase ^= 1;

        // 一个线程按按钮:mm_AB 覆盖、mma_AB 累加;算完自己去敲 inputs_finished
        if (wg_laneid == 0) {
            if (kt == 0) mm_AB(accum, a_smem, b_smem, inputs_finished);
            else         mma_AB(accum, a_smem, b_smem, inputs_finished);
        }
    }

    if (wg_laneid == 0) {
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
