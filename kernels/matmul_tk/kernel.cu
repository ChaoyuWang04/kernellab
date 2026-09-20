// ThunderKittens × 矩阵乘 —— 用 tile 当基本单位,而不是指针
//
// TK 的主张:把「16×16 的一块」当成语言里的一等公民。你声明 st_bf(共享内存 tile)、
// rt_bf / rt_fl(寄存器 tile),用 load / store / mma_AB 在它们之间搬,
// 线程到数据的映射、swizzle、fragment 布局全由库管。
//
// 代价:形状必须是 tile 尺寸的整数倍。TK 是给 ML 里那些规整形状设计的。
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
