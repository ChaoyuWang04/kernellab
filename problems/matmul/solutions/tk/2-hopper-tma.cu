// TMA + 双缓冲 —— 让上一级的异步真正开始还钱
//
// 需要: modal-h100
//
// 上一级把 `warp::mma_AB` 换成了 `warpgroup::mma_AB`,指令确实变成了 wgmma,
// 但速度只涨 3%~21%,deepK 还倒退 30%。原因写在那一级里:
//
//     warpgroup::mma_AB(acc, As, Bs);
//     warpgroup::mma_async_wait();      // 发完立刻等 —— 异步被硬用成了同步
//
// 异步指令的价值在于「发出去之后还能干别的」。这一级就是去干那件别的事:
// **一边算当前这块,一边把下一块搬进来。**
//
// ## 两个新东西
//
// **一、双缓冲。** 共享内存开两份,一份在算,另一份在收数据:
//
//     st_bf<BM, BK> (&As)[2] = al.allocate<st_bf<BM, BK>, 2>();
//     int tic = 0, toc = 1;            // 每轮 tic^=1, toc^=1 换手
//
// **二、TMA(Tensor Memory Accelerator)。** Hopper 加的搬运硬件:你给它一个
// 描述符和一个信号量,它自己把整块 tile 从显存搬进共享内存,**不占线程**。
// 上一级的 `warpgroup::load` 要 128 个线程一起搬;TMA 只要**一个线程发起**:
//
//     if (threadIdx.x == 0) {
//         tma::expect_bytes(bar, ...);                       // 告诉信号量要等多少字节
//         tma::load_async(As[toc], gA, {0, 0, brow, kt+1}, bar);
//     }
//     wait(bar, tic);                                        // 到货了再用
//
// 省下来的 127 个线程去干什么?去算 —— 这正是「搬」和「算」重叠的实现方式。
//
// ## 循环的形状变了
//
// 上一级是「搬 → 同步 → 算 → 同步」,一条直线。这一级要**先把第 0 块预取进来**,
// 然后每轮:等当前块到货 → **立刻发起下一块的搬运** → 算当前块。
// 发起下一块的那一步不阻塞,于是它和 wgmma 在时间上叠在了一起。
//
// ## H100 实测:异步终于还钱了
//
//     case      0-tiles      1-hopper     2-hopper-tma      本级 vs 上一级
//     4096³     132.05 TF    136.28 TF    319.54 TF         2.3×
//     8192³     134.69 TF    162.58 TF    310.52 TF         1.9×
//     smallK     45.84 TF     50.16 TF     50.99 TF         1.0×   <- 没动
//     tall      114.28 TF    118.99 TF    313.96 TF         2.6×
//     deepK      23.68 TF     16.52 TF     47.58 TF         2.9×
//
// **deepK 那一列最值得看**:mma.sync 23.68 → wgmma 16.52(倒退)→ 加上流水 47.58。
// 上一级的倒退不是 wgmma 不行,是**异步指令不配流水就是负担**。补上流水之后,
// 它比 Ampere 基线快了一倍。
//
// 相对 torch 从 0.17× 涨到 0.41×,这是这条阶梯上最大的一跳。
//
// ## klab ptx:三级并排
//
//     0-tiles        Ampere   mma.sync×32   ldmatrix×20   ld.global×8
//     1-hopper       Hopper   wgmma×7       ldmatrix×0    ld.global×8
//     2-hopper-tma   Hopper   wgmma×21      cp.async.bulk×8   mbarrier×8   ld.global×0
//
// 两族指令先后消失,各对应一次「把活交给硬件」:
//
// - `ldmatrix` 归零(上一级):操作数不再经寄存器往返,wgmma 直接读共享内存。
// - **`ld.global` 归零(本级):普通线程不再搬数据,全交给 TMA(`cp.async.bulk`)。**
//   `mbarrier` 是配套的信号量 —— 硬件搬完敲它,消费方等它。
//
// ## smallK 为什么一动不动
//
// K 只有 64,`ktiles = 1` —— **整个循环只跑一轮,压根没有「下一块」可以预取**。
// 流水线要有东西可流才有意义。这一档从头到尾被首尾开销主导,三级都是 50 TFLOPS 左右。
//
// ## 还差什么
//
// 官方的 educational_h100 到 level_08 为止,后面还有两级我们没做:
// level_07 是 work partitioning,level_08 是多 consumer warpgroup(生产者 warp 专职
// 搬数据、消费者 warpgroup 专职算)。那才是 H100 上真正逼近 cuBLAS 的形态。
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int NUM_WARPS = 4;                          // 一个 warpgroup
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 128

using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<BM, BK>>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ tile_gl gA,
                   const __grid_constant__ tile_gl gB,
                   const __grid_constant__ tile_gl gC,
                   int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // 每块开两份:一份在算,一份在收
    st_bf<BM, BK> (&As)[2] = al.allocate<st_bf<BM, BK>, 2>();
    st_bf<BK, BN> (&Bs)[2] = al.allocate<st_bf<BK, BN>, 2>();

    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;
    const int ktiles = K / BK;

    rt_fl<16, BN> acc;
    warp::zero(acc);

    // 信号量:TMA 搬完会敲它,消费方 wait 到对应的相位才往下走
    __shared__ semaphore bar;
    if (threadIdx.x == 0) {
        init_semaphore(bar, 0, 1);
        tma::expect_bytes(bar, size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>);
        tma::load_async(As[0], gA, {0, 0, brow, 0}, bar);      // 预取第 0 块
        tma::load_async(Bs[0], gB, {0, 0, 0, bcol}, bar);
    }
    __syncthreads();

    int tic = 0, toc = 1;
    for (int kt = 0; kt < ktiles; ++kt, tic ^= 1, toc ^= 1) {
        wait(bar, tic);                 // 等当前这块到货
        __syncthreads();

        // 立刻发起下一块的搬运。这一步不阻塞 —— 它就是和下面的 wgmma 重叠的那部分
        if (threadIdx.x == 0 && kt + 1 < ktiles) {
            tma::expect_bytes(bar, size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>);
            tma::load_async(As[toc], gA, {0, 0, brow, kt + 1}, bar);
            tma::load_async(Bs[toc], gB, {0, 0, kt + 1, bcol}, bar);
        }

        warpgroup::mma_AB(acc, As[tic], Bs[tic]);
        warpgroup::mma_async_wait();

        __syncthreads();
    }

    warpgroup::store(gC, acc, {0, 0, brow, bcol});
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    tile_gl gA = make_gl<tile_gl>(reinterpret_cast<uint64_t>(A), 1, 1, M, K);
    tile_gl gB = make_gl<tile_gl>(reinterpret_cast<uint64_t>(B), 1, 1, K, N);
    tile_gl gC = make_gl<tile_gl>(reinterpret_cast<uint64_t>(C), 1, 1, M, N);
    dim3 block(NUM_THREADS), grid(N / BN, M / BM);
    // 双缓冲:两块 A + 两块 B
    size_t smem = 2 * (sizeof(st_bf<BM, BK>) + sizeof(st_bf<BK, BN>)) + 1024;
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_kernel<<<grid, block, smem, stream>>>(gA, gB, gC, M, N, K);
}
