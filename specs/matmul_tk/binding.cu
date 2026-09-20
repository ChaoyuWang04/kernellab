// torch 与 pybind 的绑定样板。用户的 kernel.cu 不必 include torch —— 它只写 CUDA。
// 契约:kernel.cu 提供这个函数。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

void matmul_launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                   int M, int N, int K, cudaStream_t stream);

torch::Tensor matmul(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "输入要在 GPU 上");
    TORCH_CHECK(A.dtype() == torch::kBFloat16 && B.dtype() == torch::kBFloat16, "输入要是 bf16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "输入要连续");
    TORCH_CHECK(A.size(1) == B.size(0), "K 不匹配");
    const int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    matmul_launch(reinterpret_cast<const __nv_bfloat16*>(A.data_ptr()),
                  reinterpret_cast<const __nv_bfloat16*>(B.data_ptr()),
                  reinterpret_cast<__nv_bfloat16*>(C.data_ptr()),
                  M, N, K, at::cuda::getCurrentCUDAStream());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("matmul", &matmul, "bf16 matmul"); }
