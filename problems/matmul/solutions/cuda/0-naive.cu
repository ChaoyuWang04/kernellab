// CUDA × Ampere v0 — 最朴素的矩阵乘:一个线程算 C 的一个元素。
//
// 没有共享内存、没有寄存器复用、没有 tensor core。A 的一行被 N 个线程各读一遍,
// B 的一列被 M 个线程各读一遍。这是优化路线的第 0 级。
//
//
// 5090 实测 4096³:168.7 ms,0.81 TFLOPS,0.00× torch。
// 慢在哪一看便知 —— 同一个 warp 的 32 个线程读的是 A 的 32 个不同行。
// 只写 CUDA:torch 张量、pybind 导出都在接线的 binding.cu 里。
#include <cuda_bf16.h>

#define BLOCK 16     // 每个 block 边长(BLOCK × BLOCK 个线程)

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    // 注意这里的映射:threadIdx.x 决定的是「行」。
    // 同一个 warp 里 threadIdx.x 连着变,于是 32 个线程读的是 A 的 32 个不同行 ——
    // 地址相隔 K 个元素,完全没法合并。这就是第 1 级要修的东西。
    const int row = blockIdx.x * BLOCK + threadIdx.x;
    const int col = blockIdx.y * BLOCK + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.f;                                   // 累加用 fp32
    for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(A[(size_t)row * K + k]) *
               __bfloat162float(B[(size_t)k * N + col]);
    }
    C[(size_t)row * N + col] = __float2bfloat16(acc);
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
