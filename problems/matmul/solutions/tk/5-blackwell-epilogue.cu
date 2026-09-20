// epilogue 也开流水 —— 写回不再是一锤子买卖
//
// 需要: modal-b200
//
// 上一级把「搬」和「算」分了家,295 → 710 TFLOPS。但**写回那一段还是串的**:
// 等全部算完 → 从 tensor memory 把 128×128 个累加器一次性取回寄存器 → 塞进共享内存
// → 一次 TMA 写出去。这段时间里 tensor core 完全闲着。
//
// 官方 level_08 的做法:**把 epilogue 切成 4 段流水**(1050 TFLOPs)。
//
//     rt_bf<TILE_M/4, TILE_N/4> d_regs[4];
//     for (i : 0..3) load_async(d_regs[i], accum.subtile(...));   // 4 次读一次性发出去
//     tensor_load_wait();                                         // 只等一次
//     for (i : 0..3) { store(d_smem[i], d_regs[i]); tma::store_async(...); }
//
// 两处讲究:
//
// 1. **4 次 tmem 读先全发出去,再统一等一次。** tensor memory 的读也是异步的,
//    一次一等就白白串起来了 —— 和上一级 wgmma 的教训一模一样。
// 2. **第 i 段的 TMA 写出去之后不等**,直接去做第 i+1 段的共享内存写。
//    两者在时间上叠着走。
//
// 配套:**TILE_N 从 128 加到 256**。epilogue 能流水之后,更大的分块才划算 ——
// 输出块大了,写回的开销被摊薄,而摊薄它正是这一级干的事。
//
// ## B200 实测:1057 TFLOPS,官方标 1050
//
//     case      4-warpspec         5-epilogue          变化
//     4096³      710.32 TF 0.46×   1056.83 TF 0.70×   1.49×
//     8192³      698.59 TF 0.42×   1062.06 TF 0.66×   1.52×
//     smallK      72.32 TF 0.52×     83.89 TF 0.60×   1.16×
//     tall       489.62 TF 0.47×    721.60 TF 0.72×   1.47×
//     deepK       68.20 TF 0.19×     63.07 TF 0.16×   0.92×  <- 退了
//
// 五档全过。**写回那一段本来完全不参与计算,把它流水起来值 1.5 倍** ——
// 因为省下的不是写回本身的时间,是 tensor core 干等着的时间。
//
// ## deepK 为什么退了
//
// 这一级把 TILE_N 从 128 加到了 256。deepK 是 512×512×16384,N 只有 512 ——
// **列块数从 4 掉到 2**,整个 GEMM 只切出 4×2 = 8 个块,而 B200 有 148 个 SM。
// 块数本来就不够,再把分块开大就更不够了。
//
// 这是「大分块更好」这条直觉的反例,而且它一直都在:前面几级 deepK 也从来没跟上过
// 大盘的涨幅。**分块大小要配着形状看,没有一个值对所有 case 都好** ——
// 这正是 triton / tilelang 那两列的 `autotune` 那一级在解决的问题,
// 而 TK 的 tile 尺寸是模板参数,搜不了。
//
// ## 还差一级
//
// 官方 level_09 是 2-CTA cluster + warpgroup 级并行,标 1285 TFLOPs。
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;       // epilogue 能流水了,分块才值得开大
constexpr int TILE_K = 64;
constexpr int PIPE_STAGES = 3;    // 2 级只能领先一格;3 级之后生产者能领先两格
constexpr int EPI_DEPTH = 4;      // 写回切成几段流水
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N / EPI_DEPTH>;   // 输出按段切,每段单独 TMA 出去
using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;               // tensor memory 里的累加器
using d_tt_sub = tt<float, TILE_M, TILE_N / EPI_DEPTH>;  // 它的一段

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
    d_tile (&d_smem)[EPI_DEPTH] = al.allocate<d_tile, EPI_DEPTH>();

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

    // 写回流水:先把 4 段 tmem 读全发出去(异步),只等一次;
    // 然后逐段 store 到共享内存 + TMA 写出,第 i 段的 TMA 和第 i+1 段的 store 叠着走。
    rt_bf<TILE_M / 4, TILE_N / EPI_DEPTH> d_regs[EPI_DEPTH];
#pragma unroll
    for (int i = 0; i < EPI_DEPTH; ++i) {
        warpgroup::load_async(d_regs[i], accum.subtile<d_tt_sub>(0, i * (TILE_N / EPI_DEPTH)));
    }
    tensor_load_wait();

#pragma unroll
    for (int i = 0; i < EPI_DEPTH; ++i) {
        warpgroup::sync(1);
        warpgroup::store(d_smem[i], d_regs[i]);
        warpgroup::sync(1);
        if (wg_laneid == 0) {
            tma::store_async(gC, d_smem[i], {bid_m, bid_n * EPI_DEPTH + i});
        }
    }
    tma::store_async_read_wait();
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
