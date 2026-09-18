// 经典分块 SGEMM:64×64 输出块,16×16 线程,每线程算 4×4;K 方向每步 16。
// 只用 CUDA core,不碰 tensor core;用来做 cuda 工具链的烟测与 CUDA core 侧的对照基线。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void sgemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    const int tid = threadIdx.x;  // 256 线程搬 64×16 的 A 块与 16×64 的 B 块,各 4 元素/线程
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = 0; i < 4; ++i) {
            int idx = tid + i * 256;                      // 0..1023
            int ar = idx / BK, ak = idx % BK;             // A 块 [64][16]
            As[ak][ar] = (row0 + ar < M && k0 + ak < K) ? A[(size_t)(row0 + ar) * K + k0 + ak] : 0.f;
            int bk = idx / BN, bc = idx % BN;             // B 块 [16][64]
            Bs[bk][bc] = (k0 + bk < K && col0 + bc < N) ? B[(size_t)(k0 + bk) * N + col0 + bc] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
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
            int r = row0 + ty * TM + i, c = col0 + tx * TN + j;
            if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
}

torch::Tensor sgemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32);
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(256), grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("sgemm", &sgemm, "tiled sgemm (fp32, CUDA cores)"); }
