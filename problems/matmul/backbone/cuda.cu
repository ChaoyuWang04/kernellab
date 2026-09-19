// CUDA 矩阵乘。C[M,N] = A[M,K] @ B[K,N],bf16 进、fp32 累加、bf16 出。
//
// 下面 matmul_launch 的签名是你与系统之间的契约:接线的 binding.cu 声明了它、
// 负责 torch 张量检查与 pybind 导出,改了签名就链接不上。
// kernel 本体与启动配置是你的,从头写。
//
// 你只写 CUDA —— 这个文件不该出现 torch/extension.h 或 PYBIND11_MODULE。
#include <cuda_bf16.h>

#define BLOCK 16     // 分块参数由你定;grid / block 在 matmul_launch 里算

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    // ① 我是谁:从 blockIdx / threadIdx 算出这个线程负责 C 的哪些元素
    //    (让同一个 warp 的 32 个线程落在连续的列上 —— 这是优化路线第 1 级)

    // ② 越界判断:M / N 不保证是分块的整数倍

    // ③ 开累加器。累加一律用 float,不要用 bf16

    // ④ 沿 K 累加。要复用就先把 A、B 的 tile 搬进 __shared__,
    //    搬完 __syncthreads(),算完再 __syncthreads() 才能覆盖

    // ⑤ 写回:__float2bfloat16(acc),越界不写
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    // grid / block 怎么切由你定,记得把 stream 传进去
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
