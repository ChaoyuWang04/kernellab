// ThunderKittens 烟测:C = A + B,每个 warp 处理一块 16×64 的 bf16 寄存器 tile。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kittens.cuh"
using namespace kittens;

constexpr int TR = 16, TC = 64;                 // 寄存器 tile 尺寸
using gl_t = gl<bf16, 1, 1, -1, -1>;            // 全局张量布局:[rows, cols],行列运行时决定

__global__ void tile_add_kernel(const __grid_constant__ gl_t A, const __grid_constant__ gl_t B,
                                const __grid_constant__ gl_t C) {
    const int warp = kittens::warpid();         // block 内 warp 号
    const int tile_r = blockIdx.y * (blockDim.x / 32) + warp;
    const int tile_c = blockIdx.x;
    rt_bf<TR, TC> a, b;
    warp::load(a, A, {0, 0, tile_r, tile_c});
    warp::load(b, B, {0, 0, tile_r, tile_c});
    warp::add(a, a, b);
    warp::store(C, a, {0, 0, tile_r, tile_c});
}

torch::Tensor tile_add(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kBFloat16 && A.is_contiguous() && B.sizes() == A.sizes());
    const int rows = A.size(0), cols = A.size(1);
    TORCH_CHECK(rows % (TR * 4) == 0 && cols % TC == 0, "rows 需是 64 的倍数,cols 需是 64 的倍数");
    auto C = torch::empty_like(A);
    // gl 的构造函数对编译期固定的维度要求传 nullptr,运行期维度传 size_t;make_gl 替我们做这个转换
    gl_t ga = make_gl<gl_t>(reinterpret_cast<uint64_t>(A.data_ptr()), 1, 1, rows, cols);
    gl_t gb = make_gl<gl_t>(reinterpret_cast<uint64_t>(B.data_ptr()), 1, 1, rows, cols);
    gl_t gc = make_gl<gl_t>(reinterpret_cast<uint64_t>(C.data_ptr()), 1, 1, rows, cols);
    dim3 block(128), grid(cols / TC, rows / (TR * 4));  // 4 warp/block,每 warp 一块 16×64
    tile_add_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(ga, gb, gc);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tile_add", &tile_add, "TK tile add"); }
