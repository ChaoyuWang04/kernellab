// Tensor core —— 用矩阵指令替掉标量 FMA
//
// 前四级都在解决「喂得够不够快」,这一级换「算得够不够快」。
// 一条 mma 指令算一整块 16×16×16,取代 4096 次标量 FMA。
//
// 这里用 WMMA(nvcuda::wmma):声明 fragment、load_matrix_sync 装载、mma_sync 乘加。
// 它编译后就是 mma.sync —— 用 klab ptx 可以直接数出来。生产级实现会手写
// mma.sync + ldmatrix 以拿到更细的布局控制,但那是另一个量级的复杂度。
//
// 分块:一个 block 算 128×128,8 个 warp 排成 4×2,每个 warp 算 32×64
//
// 5090 实测 4096³:4.33 ms,31.75 TFLOPS —— 比上一级快 1.4 倍。
//
// 但只有这张卡峰值的 14%。手写到这一步就停了,而同一道题的 Triton 版是
// 205 TFLOPS(89% 峰值)—— 快 6.5 倍。这个差距就是这条阶梯最终要给你的答案:
//
//     「用上 tensor core」和「喂饱 tensor core」是两件事。
//
// 还差的几级(Triton 的 tl.dot 背后替你做了的):BK 加大到 32/64 并让每个 warp
// 持有更多累加器 fragment 来摊薄同步;共享内存 swizzle 彻底消掉 bank 冲突;
// cp.async 把搬运交给 DMA;最后是手写 mma.sync + ldmatrix,省掉 fragment
// 经共享内存往返的那一趟。
//
// 顺带一个已经修过的坑:B 的 fragment 在 i 循环里是不变的,最初写成在内层装载,
// 白装了 WM/16 倍 —— 改完只快了 0.7%,说明瓶颈另有其人,别凭直觉优化。
// (= 2×4 个 16×16 的 fragment)。
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

#define BM 128
#define BN 128
#define BK 16          // WMMA 的 K 也是 16,一段正好一次 mma
#define WM 32          // 一个 warp 算 WM×WN
#define WN 64
#define WARPS_M (BM / WM)                    // 4
#define WARPS_N (BN / WN)                    // 2
#define NWARPS (WARPS_M * WARPS_N)           // 8
#define NTHREADS (NWARPS * 32)               // 256

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
#define APAD 8         // 给 As 的行加 padding,错开共享内存的 bank,避免装载 fragment 时冲突
    __shared__ __nv_bfloat16 As[BM][BK + APAD];
    __shared__ __nv_bfloat16 Bs[BK][BN];
    __shared__ float Cs[NWARPS][16][16];     // 写回中转:M/N 不是 16 的整数倍时要逐元素判越界

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warpM = warp / WARPS_N;        // 0..3
    const int warpN = warp % WARPS_N;        // 0..1
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.f);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM / 16][WN / 16];
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += NTHREADS) {          // 搬 A 的 128×16
            int r = i / BK, c = i % BK;
            As[r][c] = (row0 + r < M && k0 + c < K) ? A[(size_t)(row0 + r) * K + k0 + c] : zero;
        }
        for (int i = tid; i < BK * BN; i += NTHREADS) {          // 搬 B 的 16×128
            int r = i / BN, c = i % BN;
            Bs[r][c] = (k0 + r < K && col0 + c < N) ? B[(size_t)(k0 + r) * N + col0 + c] : zero;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf[WN / 16];
        // B 的 fragment 在 i 循环里是不变的,装一次就够 —— 放进内层会白装 WM/16 倍
#pragma unroll
        for (int j = 0; j < WN / 16; ++j)
            wmma::load_matrix_sync(bf[j], &Bs[0][warpN * WN + j * 16], BN);
#pragma unroll
        for (int i = 0; i < WM / 16; ++i) {
            wmma::load_matrix_sync(af, &As[warpM * WM + i * 16][0], BK + APAD);
#pragma unroll
            for (int j = 0; j < WN / 16; ++j)
                wmma::mma_sync(acc[i][j], af, bf[j], acc[i][j]);   // 一条指令算 16×16×16
        }
        __syncthreads();
    }

    // 每个 fragment 先落到共享内存,再逐元素判越界写回
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 16; ++j) {
            wmma::store_matrix_sync(&Cs[warp][0][0], acc[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 16 * 16; e += 32) {
                int r = row0 + warpM * WM + i * 16 + e / 16;
                int c = col0 + warpN * WN + j * 16 + e % 16;
                if (r < M && c < N) C[(size_t)r * N + c] = __float2bfloat16(Cs[warp][e / 16][e % 16]);
            }
            __syncwarp();
        }
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
