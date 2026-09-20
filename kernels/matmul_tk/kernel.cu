// ThunderKittens 矩阵乘。C[M,N] = A[M,K] @ B[K,N],bf16 进、fp32 累加、bf16 出。
//
// TK 的主张:把「一块 tile」当成语言里的一等公民。你声明 st_bf(共享内存 tile)、
// rt_bf / rt_fl(寄存器 tile),用 load / store / mma_AB 在它们之间搬 ——
// 线程到数据的映射、swizzle、fragment 布局全由库管,你一个指针都不写。
//
// 契约:下面的常量、tile_gl 类型与 matmul_launch 的签名。接线的 binding.cu 声明了
// matmul_launch,负责 torch 张量检查与 pybind 导出,改了签名就链接不上。
//
// 注意 TK 要求 M / N / K 都是 tile 尺寸的整数倍 —— 这是它的设计取舍,
// 所以这道题的 1000x999x777 那档不在 TK 的 case 列表里。
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BM = 64;          // 一个 block 算 C 的 BM×BN
constexpr int BN = 64;
constexpr int BK = 64;          // K 方向每步吃多少
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WM = BM / NUM_WARPS;   // 每个 warp 负责几行

using tile_gl = gl<bf16, 1, 1, -1, -1, st_bf<BM, BK>>;

__global__ __launch_bounds__(NUM_THREADS, 1)
void matmul_kernel(const __grid_constant__ tile_gl gA,
                   const __grid_constant__ tile_gl gB,
                   const __grid_constant__ tile_gl gC,
                   int M, int N, int K) {
    // ① 在动态共享内存里开两块 tile:
    //    extern __shared__ alignment_dummy __shm[];
    //    shared_allocator al((int*)&__shm[0]);
    //    st_bf<BM, BK> &As = al.allocate<st_bf<BM, BK>>();

    // ② 开累加器:rt_fl<WM, BN>,用 warp::zero 清零

    // ③ 沿 K 一段一段:
    //    group<NUM_WARPS>::load(As, gA, {0, 0, 第几行块, 第几个 K 块})   整个 block 合作搬
    //    group<NUM_WARPS>::sync(0)
    //    每个 warp 取自己那几行:As.template subtile<WM, BK>({warpid(), 0})
    //    B 的 fragment 要列布局:rt_bf<BK, BN, ducks::rt_layout::col>
    //    warp::mma_AB(acc, a, b, acc)                                   tile 乘 tile
    //    再 sync 一次才能覆盖

    // ④ 写回:warp::copy 把 fp32 累加器转成 rt_bf,再 warp::store 到 gC
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    // 用 make_gl<tile_gl>(ptr, b, d, rows, cols) 把裸指针包成 TK 的全局张量。
    // 共享内存超过默认 carveout 时要先 cudaFuncSetAttribute 抬上限。
    tile_gl gA = make_gl<tile_gl>(reinterpret_cast<uint64_t>(A), 1, 1, M, K);
    tile_gl gB = make_gl<tile_gl>(reinterpret_cast<uint64_t>(B), 1, 1, K, N);
    tile_gl gC = make_gl<tile_gl>(reinterpret_cast<uint64_t>(C), 1, 1, M, N);
    dim3 block(NUM_THREADS), grid(N / BN, M / BM);
    size_t smem = sizeof(st_bf<BM, BK>) + sizeof(st_bf<BK, BN>) + 1024;
    cudaFuncSetAttribute(matmul_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    matmul_kernel<<<grid, block, smem, stream>>>(gA, gB, gC, M, N, K);
}
