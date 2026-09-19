// 合并访存 —— 只把两行下标换个个儿
//
// 第 0 级里 threadIdx.x 决定「行」:同一个 warp 的 32 个线程读 A 的 32 个不同行,
// 地址彼此相隔 K 个元素,硬件没法合并,一次 128 字节的事务只用上 2 个字节。
// 把 threadIdx.x 改成决定「列」,同一 warp 读的就是 B 的连续 32 列、C 的连续 32 列,
// 一次事务全用上。代码只动了两行,grid 的 x/y 跟着换。
//
//
// 5090 实测 4096³:40.5 ms,3.40 TFLOPS —— 比上一级快 4.2 倍。
// 两行下标换个个儿,四倍。这是整条阶梯上性价比最高的一次改动。
// 这一级不需要理解任何算法,只需要知道「一个 warp 的 32 个线程应该读连续地址」。
#include <cuda_bf16.h>

#define BLOCK 16

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    const int col = blockIdx.x * BLOCK + threadIdx.x;   // x -> 列(连续)
    const int row = blockIdx.y * BLOCK + threadIdx.y;   // y -> 行
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(A[(size_t)row * K + k]) *
               __bfloat162float(B[(size_t)k * N + col]);
    }
    C[(size_t)row * N + col] = __float2bfloat16(acc);
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
