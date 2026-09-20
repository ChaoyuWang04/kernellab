// tile 当一等公民 —— ThunderKittens 基线
//
// 另外四种语言里你都在算指针:A_ptr + row*stride + col。TK 把这一层整个拿掉 ——
// 你声明 st_bf(共享内存 tile)、rt_bf / rt_fl(寄存器 tile),用 load / store / mma_AB
// 在它们之间搬,线程到数据的映射、swizzle、fragment 布局全由库管。整个 kernel
// 一个指针都没有,mma_AB 那一句就是「tile 乘 tile」。
//
// 5090 实测,与另外四种语言并排(4096³):
//
//     triton    0.95×   206.5 TFLOPS
//     tilelang  0.88×   187.5
//     tk        0.86×   181.4     ← 这份
//     cuda      0.15×    31.6     (手写 WMMA)
//     cute      0.02×     3.9     (朴素)
//
// 一份 60 行、零指针的代码打到 0.86×,比手写 WMMA 的 CUDA 版快 5.7 倍 ——
// 这就是 tile 抽象的价值。klab ptx 验:mma.sync×32 + ldmatrix×20,命中 Ampere。
//
// 五档 case 的表现(5090):
//     4096³        0.86×    8192³  0.71×    smallK  1.07×
//     tall         0.87×    deepK  0.16×
//
// === 这份的代价 ===
//
// TK 的 tile 操作要求 M / N / K 都是 tile 尺寸(这里 64)的整数倍,所以这道题的
// 1000x999x777 那档不在 TK 的 case 列表里 —— 另外四种语言都能跑。这不是 bug,
// 是 TK 的设计取舍:它是给 ML 里那些规整形状准备的,不规整的边界由调用方去 pad。
//
// 另一个代价:BM=BN=BK=64 固定死了。TK 的 tile 尺寸是模板参数,要调分块得改类型,
// 不像 Triton 那样加个 autotune 装饰器就能搜。

#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BM = 64;          // 一个 block 算 C 的 BM×BN
constexpr int BN = 64;
constexpr int BK = 64;          // K 方向每步吃多少
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WM = BM / NUM_WARPS;   // 每个 warp 负责 16 行

using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<BM, BK>>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ tile_gl gA,
                   const __grid_constant__ tile_gl gB,
                   const __grid_constant__ tile_gl gC,
                   int M, int N, int K) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BM, BK> &As = al.allocate<st_bf<BM, BK>>();
    st_bf<BK, BN> &Bs = al.allocate<st_bf<BK, BN>>();

    const int warp = warpid();
    const int brow = blockIdx.y;     // 第几个行块
    const int bcol = blockIdx.x;     // 第几个列块

    rt_fl<WM, BN> acc;               // 累加器住寄存器,fp32
    warp::zero(acc);

    const int ktiles = K / BK;
    for (int kt = 0; kt < ktiles; ++kt) {
        // 整个 block 合作把两块搬进共享内存(TK 自己排线程、自己 swizzle)
        group<NUM_WARPS>::load(As, gA, {0, 0, brow, kt});
        group<NUM_WARPS>::load(Bs, gB, {0, 0, kt, bcol});
        group<NUM_WARPS>::sync(0);

        // 每个 warp 取自己那 16 行;B 的 fragment 要列布局(mma_AB 的约定)
        rt_bf<WM, BK> a;
        rt_bf<BK, BN, ducks::rt_layout::col> b;
        warp::load(a, As.template subtile<WM, BK>({warp, 0}));
        warp::load(b, Bs);
        warp::mma_AB(acc, a, b, acc);     // 一句话:tile 乘 tile,降到 mma.sync

        group<NUM_WARPS>::sync(0);
    }

    rt_bf<WM, BN> out;
    warp::copy(out, acc);                 // fp32 累加器 -> bf16
    warp::store(gC, out, {0, 0, brow * NUM_WARPS + warp, bcol});
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    tile_gl gA = make_gl<tile_gl>(reinterpret_cast<uint64_t>(A), 1, 1, M, K);
    tile_gl gB = make_gl<tile_gl>(reinterpret_cast<uint64_t>(B), 1, 1, K, N);
    tile_gl gC = make_gl<tile_gl>(reinterpret_cast<uint64_t>(C), 1, 1, M, N);
    dim3 block(NUM_THREADS), grid(N / BN, M / BM);
    size_t smem = sizeof(st_bf<BM, BK>) + sizeof(st_bf<BK, BN>) + 1024;
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_kernel<<<grid, block, smem, stream>>>(gA, gB, gC, M, N, K);
}
