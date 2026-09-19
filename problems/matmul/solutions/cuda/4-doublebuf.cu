// 双缓冲 —— 让「搬下一块」和「算当前块」重叠
//
// 上一级的循环是串的:搬 → 同步 → 算 → 同步 → 再搬。搬的时候计算单元闲着。
// 开两块共享内存轮流用:算第 i 块的同时,把第 i+1 块搬进另一块缓冲,
// 一轮只需要一次同步。体检单里「等显存把数据取回来」的占比应该明显下降。
//
// 注意:Ampere 真正的硬件手段是 cp.async(把搬运交给 DMA,不占寄存器),
// Triton 的 num_stages 就是编译器替你做这件事。这里用普通 load + 双缓冲演示
// 同一个概念 —— cp.async 要求 16 字节对齐,而本题有 K=777 这档过不去,
//
// 5090 实测 4096³:5.97 ms,23.04 TFLOPS —— 与上一级持平,没有收益。
//
// 原因:到这一级瓶颈已经不是「等数据」了。上一级的体检单里主要等待不是
// 「等显存把数据取回来」,而是标量 FMA 本身的吞吐 —— 128×128 的输出块靠
// CUDA core 一次一次乘加,算力就是不够。搬得再快也没用。
//
// 这和 Triton 版的 2-occupancy 是同一课:先看判定,再决定动哪个旋钮。
// 双缓冲要在「卡在等数据」时才值钱。
// 为它写一套对齐兜底会把这一级的重点冲淡。想看 cp.async 真身:klab ptx 打 Triton 版。
#include <cuda_bf16.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))     // 256

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    __shared__ __nv_bfloat16 As[2][BK][BM];   // 两块缓冲轮流用
    __shared__ __nv_bfloat16 Bs[2][BK][BN];

    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);
    const int tCol = tid % (BN / TN);
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.f);

    float acc[TM][TN] = {};

    // 搬第 k0 段到缓冲 buf
    auto load_tile = [&](int k0, int buf) {
#pragma unroll
        for (int i = 0; i < (BM * BK) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int ar = idx / BK, ak = idx % BK;
            As[buf][ak][ar] = (row0 + ar < M && k0 + ak < K)
                            ? A[(size_t)(row0 + ar) * K + k0 + ak] : zero;
        }
#pragma unroll
        for (int i = 0; i < (BK * BN) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int bk = idx / BN, bc = idx % BN;
            Bs[buf][bk][bc] = (k0 + bk < K && col0 + bc < N)
                            ? B[(size_t)(k0 + bk) * N + col0 + bc] : zero;
        }
    };

    load_tile(0, 0);                 // 先把第 0 段搬进来
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) load_tile(k0 + BK, buf ^ 1);   // 搬下一段到另一块,不等它

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {               // 同时算当前这块
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = __bfloat162float(As[buf][kk][tRow * TM + i]);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = __bfloat162float(Bs[buf][kk][tCol * TN + j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();             // 一轮只要一次同步
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = row0 + tRow * TM + i, c = col0 + tCol * TN + j;
            if (r < M && c < N) C[(size_t)r * N + c] = __float2bfloat16(acc[i][j]);
        }
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    dim3 block(NTHREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
