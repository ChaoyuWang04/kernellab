// 2-CTA cluster —— 两个 block 合伙,外加两个 consumer warpgroup
//
// 需要: modal-b200
//
// 官方 educational_b200 的最后一级(标 1285 TFLOPs)。前面每一级都只改一件事,
// 这一级**同时动了四样**,因为它们互相咬着,拆开单独上都不划算:
//
// 1. **`__cluster_dims__(2,1,1)`** —— 两个 block 组成一个 cluster。Hopper 起
//    cluster 内的 block 能互相访问共享内存;到 Blackwell,**一条 `mma2_AB` 可以
//    同时吃两个 CTA 的数据**。B 的 tile 因此只用搬一半(`st_bf<TILE_K, TILE_N/2>`),
//    另一半由伙伴 block 提供。少搬一半 B,这是这一级最大的一笔。
//
// 2. **两个 consumer warpgroup**(`NUM_CONSUMERS = 2`)。12 个 warp:
//    8 个分成两组各算各的一块 C,4 个当生产者。
//
// 3. **寄存器按角色重新分配**:
//        生产者 `warpgroup::decrease_registers<56>()`   —— 只发 TMA,不需要寄存器
//        消费者 `warpgroup::increase_registers<224>()`  —— 要装下整个 epilogue
//    这就是 PTX 里那条 `setmaxnreg`。**同一个 block 里不同 warp 拿不同数量的寄存器**,
//    前面几级从来没用过这个能力。
//
// 4. 流水加到 4 级、epilogue 切成 8 段、输出双缓冲(`NUM_D_TILES = 2`)。
//
// ## 形状约束变严了
//
// 一个 cluster 覆盖 `CLUSTER_SIZE × NUM_CONSUMERS × TILE_M` = 2×2×128 = **512 行**,
// 所以 M 必须是 512 的倍数、N 是 256 的倍数、K 是 64 的倍数。六档 case 里只有
// 1000x999x777 不满足 —— 它本来就不在 TK 的列表里(TK 要求形状是 tile 的整数倍)。
//
// ## 这一级基本是照抄
//
// 整个 kernel 逐字来自 `envs/tk/ThunderKittens/kernels/gemm/educational_b200/level_09.cu`,
// 只改了签名(官方假设方阵,只传一个 N;我们要 M/N/K)和启动壳。
// **这是本仓库定位的边界**:到这个复杂度,价值在「读懂 + 量出它值多少」,
// 不在「自己想出来」。想直接拿结果,看 cuda 那一列的 `7-cutlass-blackwell` ——
// 改四行参数就有 1388 TFLOPS,CUTLASS 里这些本来就全在。
//
// ## B200 实测:8192³ 上 0.99× cuBLAS
//
//     case      5-epilogue          6-cluster            变化
//     4096³     1056.83 TF 0.70×   1355.73 TF 0.91×    1.28×
//     8192³     1062.06 TF 0.66×   1614.65 TF 0.99×    1.52×   <- 追平 cuBLAS
//     smallK      83.89 TF 0.60×     83.89 TF 0.68×    1.00×
//     tall       721.60 TF 0.72×    849.48 TF 0.97×    1.18×
//     deepK       63.07 TF 0.16×     47.99 TF 0.12×    0.76×   <- 又退了
//
// 官方 README 标 level_09 = 1285 TFLOPs,我们在 4096³ 量到 1356、8192³ 量到 1615。
// 五档全过。
//
// ## 整条 B200 阶梯回头看
//
//     0-tiles                mma.sync          170.98 TF   0.11× torch
//     3-blackwell-tcgen05    换上 tcgen05      294.98 TF   0.20×      1.7×
//     4-blackwell-warpspec   搬算分家          710.32 TF   0.46×      2.4×
//     5-blackwell-epilogue   写回也流水       1056.83 TF   0.70×      1.5×
//     6-blackwell-cluster    两个 block 合伙  1355.73 TF   0.91×      1.3×
//
// **换指令只值 1.7 倍,后面「怎么喂」的三级合起来值 4.6 倍。**
// 这就是「用上新硬件」和「喂饱新硬件」那道坎的完整分解。
//
// ## deepK 的反方向:30 → 68 → 63 → 48
//
// 它是唯一一档越优化越慢的。原因一以贯之:**这几级都在把分块开大**,而 deepK 是
// 512×512×16384 —— M 和 N 本来就小。到这一级,一个 cluster 覆盖 512 行、256 列,
// 整个 GEMM 只切出 **1×2 个 cluster = 4 个 block**,而 B200 有 148 个 SM。
// **一百多个 SM 里只有 4 个在干活。**
//
// 它需要的是相反的东西:更小的分块、或者 split-K(见 triton/tilelang 那两列的
// `3-splitk`)。**没有一组参数对所有形状都好** —— 这是六档 case 从头到尾在讲的事。
//
// ## 和 CUTLASS 比
//
// 同一张 B200:这一级 1356 TFLOPS,`cuda/7-cutlass-blackwell` 改四行参数是 1388。
// **五级手工阶梯走到的地方,CUTLASS 的默认配置直接就在那儿。** 两条路都值得走一遍 ——
// 这一条告诉你那 4.6 倍由哪三件事构成,那一条告诉你工业级实现长什么样。
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int TILE_M = 128;
constexpr int TILE_N = 256;
constexpr int TILE_K = 64;
constexpr int PIPE_STAGES = 4;
constexpr int CLUSTER_SIZE = 2;      // 两个 block 合伙
constexpr int EPI_PIPE_DEPTH = 8;
constexpr int NUM_D_TILES = 2;
constexpr int NUM_CONSUMERS = 2;     // 两组消费者各算一块 C
constexpr int NUM_WARPS = (NUM_CONSUMERS + 1) * 4;    // 12 个 warp
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 384

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N / 2>;             // 只搬一半,另一半在伙伴 block 上
using d_tile = st_bf<TILE_M, TILE_N / EPI_PIPE_DEPTH>;
using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;
using d_tt_t = tt<float, TILE_M, TILE_N>;

__global__
__cluster_dims__(CLUSTER_SIZE, 1, 1)
__launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(
    const __grid_constant__ a_gl gA,
    const __grid_constant__ b_gl gB,
    const __grid_constant__ d_gl gC,
    int M, int N, int K
) {
    if (threadIdx.x == 0) {
        gA.template prefetch_tma<a_tile>();
        gB.template prefetch_tma<b_tile>();
        gC.template prefetch_tma<d_tile>();
    }

    const int cta_rank = cluster_ctarank();
    const int iters_per_task = K / TILE_K;

    const int cluster_idx = blockIdx.x / CLUSTER_SIZE;
    const int grid_n = N / TILE_N;
    const int2 tile_coord = { cluster_idx / grid_n, cluster_idx % grid_n };

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    a_tile (&a_smem)[PIPE_STAGES][NUM_CONSUMERS] = al.allocate<a_tile, PIPE_STAGES, NUM_CONSUMERS>();
    b_tile (&b_smem)[PIPE_STAGES]                = al.allocate<b_tile, PIPE_STAGES>();
    d_tile (&d_smem)[NUM_CONSUMERS][NUM_D_TILES]  = al.allocate<d_tile, NUM_CONSUMERS, NUM_D_TILES>();

    tensor_allocator<1, 2> tm_alloc{};

    __shared__ semaphore inputs_arrived[PIPE_STAGES], inputs_finished[PIPE_STAGES];
    __shared__ semaphore outputs_arrived[NUM_CONSUMERS];
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < PIPE_STAGES; i++) {
            init_semaphore(inputs_arrived[i], 0, NUM_CONSUMERS);
            init_semaphore(inputs_finished[i], 0, NUM_CONSUMERS);
        }
        #pragma unroll
        for (int i = 0; i < NUM_CONSUMERS; i++) {
            init_semaphore(outputs_arrived[i], 0, 1);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == NUM_CONSUMERS) {
        warpgroup::decrease_registers<56>();

        if (warp::laneid() == 0 && warpgroup::warpid() == 3) {
            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; idx++) {
                tma::cluster::wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                update_phasebit<1>(bitfield, input_ring);

                #pragma unroll
                for (int i = 0; i < NUM_CONSUMERS; i++)
                    tma::cluster::load_async(a_smem[input_ring][i], gA,
                        {(tile_coord.x * 2 + cta_rank) * NUM_CONSUMERS + i, idx},
                        inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);

                tma::cluster::load_async(b_smem[input_ring], gB,
                    {idx, tile_coord.y * 2 + cta_rank},
                    inputs_arrived[input_ring], (uint16_t)(1 << cta_rank), 0);

                input_ring = ring_advance<PIPE_STAGES>(input_ring);
            }
        }
        else if (cta_rank == 0 && warp::laneid() == 0 && warpgroup::warpid() < NUM_CONSUMERS) {
            d_tt_t accum = tm_alloc.allocate<d_tt_t>(warpgroup::warpid() * TILE_N);

            int input_ring = 0;
            for (int idx = 0; idx < iters_per_task; idx++) {
                tma::cluster::expect_bytes(inputs_arrived[input_ring],
                    (CLUSTER_SIZE * NUM_CONSUMERS * sizeof(a_tile) + 2 * sizeof(b_tile)) / NUM_CONSUMERS);
                tma::cluster::wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);

                if (idx == 0) mm2_AB (accum, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);
                else          mma2_AB(accum, a_smem[input_ring][warpgroup::warpid()], b_smem[input_ring], inputs_finished[input_ring]);

                input_ring = ring_advance<PIPE_STAGES>(input_ring);
            }
            detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived[warpgroup::warpid()]);
        }
    }
    else {
        warpgroup::increase_registers<224>();

        d_tt_t accum = tm_alloc.allocate<d_tt_t>(warpgroup::groupid() * TILE_N);

        wait(outputs_arrived[warpgroup::groupid()], 0);

        rt_bf<TILE_M / 4, TILE_N / EPI_PIPE_DEPTH> d_reg[EPI_PIPE_DEPTH];
        #pragma unroll
        for (int i = 0; i < EPI_PIPE_DEPTH; i++)
            warpgroup::load_async(d_reg[i],
                accum.template subtile<tt<float, TILE_M, TILE_N / EPI_PIPE_DEPTH>>(
                    0, (TILE_N / EPI_PIPE_DEPTH) * i));
        tensor_load_wait();

        warpgroup::sync(warpgroup::groupid() + 1);

        #pragma unroll
        for (int i = 0; i < EPI_PIPE_DEPTH; i++) {
            warpgroup::tma::store_async_read_wait<NUM_D_TILES - 1>();
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::store(d_smem[warpgroup::groupid()][i % NUM_D_TILES], d_reg[i]);
            warpgroup::sync(warpgroup::groupid() + 1);
            warpgroup::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                gC, d_smem[warpgroup::groupid()][i % NUM_D_TILES],
                {(2 * tile_coord.x + cta_rank) * NUM_CONSUMERS + warpgroup::groupid(),
                 EPI_PIPE_DEPTH * tile_coord.y + i});
        }
    }
}


void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    a_gl gA = make_gl<a_gl>(reinterpret_cast<uint64_t>(A), 1, 1, M, K);
    b_gl gB = make_gl<b_gl>(reinterpret_cast<uint64_t>(B), 1, 1, K, N);
    d_gl gC = make_gl<d_gl>(reinterpret_cast<uint64_t>(C), 1, 1, M, N);
    // 一个 cluster 覆盖 CLUSTER_SIZE × NUM_CONSUMERS × TILE_M = 512 行
    int grid = (M / (CLUSTER_SIZE * NUM_CONSUMERS * TILE_M)) * (N / TILE_N) * CLUSTER_SIZE;
    int smem = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_kernel<<<grid, NUM_THREADS, smem, stream>>>(gA, gB, gC, M, N, K);
}
