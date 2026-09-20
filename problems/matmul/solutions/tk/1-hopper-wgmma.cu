// 换代 —— warp:: 换成 warpgroup::,mma.sync 变 wgmma
//
// 需要: modal-h100
//
// 这是整个仓库里「换代」最干净的一格:**换个命名空间,删掉四行**。
//
// 上一级(0-tiles)要先把共享内存里的 tile 搬进寄存器 fragment,再乘:
//
//     rt_bf<WM, BK> a;                             // 寄存器 fragment
//     rt_bf<BK, BN, ducks::rt_layout::col> b;      // B 还得是列布局
//     warp::load(a, As.template subtile<WM, BK>({warp, 0}));
//     warp::load(b, Bs);
//     warp::mma_AB(acc, a, b, acc);
//
// 这一级四行变一行:
//
//     warpgroup::mma_AB(acc, As, Bs);              // 直接吃共享内存 tile
//
// **wgmma 的操作数走共享内存描述符,不进寄存器。** 省掉的不只是四行代码,
// 是每轮 K 循环一整趟「共享内存 → 寄存器」的往返 —— 上一级的体检单里,
// 那趟往返正是 ld.shared 的大头。顺带 B 也不用再摆成列布局了。
//
// ## 代价:多了一条异步协议
//
// mma.sync 是同步的,发出去就算完了。wgmma 是**异步**的,发出去只是排进队列,
// 要自己等:
//
//     warpgroup::mma_AB(acc, As, Bs);
//     warpgroup::mma_async_wait();     // 不等就去读 acc,读到的是半成品
//
// TK 把围栏(fence)和提交(commit)都藏在 mma_AB 里了,你只需要记得 wait。
// 裸写 PTX 的话这三步都得自己来 —— 这就是「用官方封装」和「手搓协议」的差别。
//
// ## 约束
//
// - **只有 sm_90 有 wgmma。** 5090(sm_120)数字更大,但没有这条指令,编不过。
//   载入这一级之前先把右上角的后端切到 modal-h100。
// - 一个 warpgroup 固定是 4 个 warp = 128 线程,累加器高度固定 16
//   (4 个 warp 各 16 行,合起来正好 64)。这些不是 TK 的规定,是硬件的。
//
// ## H100 实测:指令真的换了,但快得不多,还有一档变慢了
//
// klab ptx 的对照最能说明问题:
//
//     0-tiles          世代 Ampere   mma.sync×32   ldmatrix×20
//     1-hopper-wgmma   世代 Hopper   wgmma×7       ldmatrix×0   <- 整整消失了
//
// **ldmatrix 归零,就是「操作数不再过寄存器」的直接证据** —— 那 20 条指令干的正是
// 「共享内存 → 寄存器 fragment」这趟往返,换成 wgmma 之后它整族不见了。
//
// 速度(H100,同一张卡上两级并排):
//
//     case      0-tiles(mma.sync)    1-hopper(wgmma)      变化
//     4096³     132.05 TFLOPS        136.28 TFLOPS       +3%
//     8192³     134.69 TFLOPS        162.58 TFLOPS      +21%
//     smallK     45.84 TFLOPS         50.16 TFLOPS       +9%
//     tall      114.28 TFLOPS        118.99 TFLOPS       +4%
//     deepK      23.68 TFLOPS         16.52 TFLOPS      -30%   <- 变慢了
//
// ## 为什么只快了一点,deepK 还倒退
//
// 看循环里这两行:
//
//     warpgroup::mma_AB(acc, As, Bs);
//     warpgroup::mma_async_wait();      // <- 发完立刻等
//
// **wgmma 是异步的,而我们每发一条就等一次 —— 等于把异步硬用成了同步。**
// 省下来的只有 ldmatrix 那趟往返,指令本身的并行能力一点没用上。
//
// deepK(512×512×16384)把这件事放大成了倒退:它只切出 64 个块,而 H100 有 132 个
// SM,一多半 SM 空着,没有别的 warp 能盖住等待;同时 K=16384 意味着 **256 轮**循环,
// 每轮都付一次异步指令的启动与等待开销。**并行度不够的时候,异步比同步更贵。**
//
// 这正好是下一级的题目:官方的 educational_h100 走到 level_06 才引入 TMA + 双缓冲,
// 让「搬下一块」和「算当前块」真正重叠起来 —— 到那时 wgmma 的异步才开始还钱。
// 本级停在这里,是为了让「换了指令」和「用好了指令」分成两件事来量。
//
// (对照:同一份 0-tiles 在 5090 上是 181.4 TFLOPS/0.86× torch,在 H100 上却只有
//  0.17× torch —— 不是代码退步了,是 H100 的 cuBLAS 基线高得多。跨卡比倍数没有意义,
//  只能同卡比。)
#include <cuda_bf16.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int NUM_WARPS = 4;                          // 一个 warpgroup,不能改
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 128

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

    const int brow = blockIdx.y;
    const int bcol = blockIdx.x;

    // 累加器高度固定 16:warpgroup 的 4 个 warp 各持 16 行,合起来 64
    rt_fl<16, BN> acc;
    warp::zero(acc);

    const int ktiles = K / BK;
    for (int kt = 0; kt < ktiles; ++kt) {
        warpgroup::load(As, gA, {0, 0, brow, kt});
        warpgroup::load(Bs, gB, {0, 0, kt, bcol});
        __syncthreads();

        warpgroup::mma_AB(acc, As, Bs);   // 共享内存 tile 直接进 wgmma,不过寄存器
        warpgroup::mma_async_wait();      // 异步指令,不等就读到半成品

        __syncthreads();                  // 算完才能覆盖共享内存
    }

    warpgroup::store(gC, acc, {0, 0, brow, bcol});
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
