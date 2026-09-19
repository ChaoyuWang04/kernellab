// 共享内存分块 —— 让搬进来的字节被用 BT 次
//
// 前两级里,A 的一行被 N 个线程各从显存读一遍,B 的一列被 M 个线程各读一遍。
// 把 A、B 的一小块先搬进共享内存,块内的 BT 个线程共用它 —— 每个字节的复用次数
// 从 1 变成 BT,算术强度提高 BT 倍。
//
//
// 5090 实测 4096³:45.1 ms,3.05 TFLOPS —— 比上一级「慢」了 10%。
//
// 这一级没有变快,而这正是它值得保留的原因:
//   · 每个线程仍然只算一个输出,于是每做一次乘加就要读两次共享内存 ——
//     瓶颈只是从全局带宽挪到了共享内存带宽,没消失。
//   · 32×32 = 1024 线程/block,占用率被压得很低。
//   · 上一级那种「合并的全局读」在 5090 的 96 MB L2 上命中率很高,
//     B 的同一列会被相邻 block 反复命中,并没有真的每次都去显存。
//
// 教训:分块本身不产生收益,产生收益的是「复用」。复用没提到寄存器那一层,
// 就只是把瓶颈搬了个家。下一级才是真正的跃迁。
// 代价是要手动同步:搬完要 __syncthreads() 才能读,算完要再同步一次才能覆盖。
#include <cuda_bf16.h>

#define BT 32     // 分块边长:一个 block 算 C 的 BT×BT,每步吃 K 方向 BT 个

__global__ void matmul_kernel(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              __nv_bfloat16* __restrict__ C,
                              int M, int N, int K) {
    __shared__ __nv_bfloat16 As[BT][BT];
    __shared__ __nv_bfloat16 Bs[BT][BT];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * BT + tx;
    const int row = blockIdx.y * BT + ty;
    const __nv_bfloat16 zero = __float2bfloat16(0.f);

    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += BT) {
        // 合作搬运:每个线程各搬 A、B 的一个元素,越界的填 0
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : zero;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : zero;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BT; ++kk) {
            acc += __bfloat162float(As[ty][kk]) * __bfloat162float(Bs[kk][tx]);
        }
        __syncthreads();   // 算完才能让下一轮覆盖
    }

    if (row < M && col < N) C[(size_t)row * N + col] = __float2bfloat16(acc);
}

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream) {
    dim3 block(BT, BT);
    dim3 grid((N + BT - 1) / BT, (M + BT - 1) / BT);
    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
