// 寄存器分块 —— 把复用再往上提一层
//
// 共享内存也有带宽上限:上一级里每个线程每次乘加都要读两次共享内存。
// 让一个线程算 TM×TN 个输出:从共享内存读 TM+TN 个值,却能做 TM×TN 次乘加,
// 共享内存的读写次数除以了 (TM×TN)/(TM+TN) 倍。复用发生在寄存器里。
//
// 代价是寄存器压力:光累加器就占 TM×TN 个。这是本级唯一的权衡,
//
// 5090 实测 4096³:5.90 ms,23.29 TFLOPS —— 比上一级快 7.6 倍,整条阶梯最大的一跳。
// 每个线程从共享内存读 8+8 个值,却做 64 次乘加;共享内存的访问次数除以了 4 倍。
// 体检单的「每个线程占几个寄存器」会直接告诉你付了多少。
#include <cuda_bf16.h>

#define BM 128    // 一个 block 算 C 的 BM×BN
#define BN 128
#define BK 8      // K 方向每步吃多少
#define TM 8      // 一个线程算 TM×TN
#define TN 8
#define NTHREADS ((BM / TM) * (BN / TN))     // 16×16 = 256

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    __shared__ __nv_bfloat16 As[BK][BM];   // 转置存:内层循环按 As[kk][...] 连着读
    __shared__ __nv_bfloat16 Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tRow = tid / (BN / TN);      // 0..15
    const int tCol = tid % (BN / TN);      // 0..15
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;
    const __nv_bfloat16 zero = __float2bfloat16(0.f);

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 256 个线程搬 A 的 BM×BK 与 B 的 BK×BN,各 1024 个元素,每线程 4 个
#pragma unroll
        for (int i = 0; i < (BM * BK) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int ar = idx / BK, ak = idx % BK;
            As[ak][ar] = (row0 + ar < M && k0 + ak < K)
                       ? A[(size_t)(row0 + ar) * K + k0 + ak] : zero;
        }
#pragma unroll
        for (int i = 0; i < (BK * BN) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int bk = idx / BN, bc = idx % BN;
            Bs[bk][bc] = (k0 + bk < K && col0 + bc < N)
                       ? B[(size_t)(k0 + bk) * N + col0 + bc] : zero;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = __bfloat162float(As[kk][tRow * TM + i]);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = __bfloat162float(Bs[kk][tCol * TN + j]);
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
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
