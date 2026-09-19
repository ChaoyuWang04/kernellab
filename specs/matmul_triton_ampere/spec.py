"""接线:kernels/matmul_triton_ampere/kernel.py 的 matmul_kernel(Triton,bf16)。

用户只写 @triton.jit 的 kernel 本体与 tile 常量;分配输出、算 grid、传 stride
这些样板都在这里(run()),不占用户的代码空间。
"""
import torch
import triton

from klab.harness.spec import kernel_module

k = kernel_module(__file__)

# PyTorch 默认允许 cuBLAS 把 split-K 的分段部分和用 bf16 归约。块数喂不满 GPU 时(本算子的
# 1000x999x777 只有 64 块,5090 有 170 个 SM)cuBLAS 会选 split-K,于是部分和的量级(~28)在 bf16 里
# 一舍入就带来约 0.1 的绝对误差,与最终结果多小无关 —— 接近 0 的输出因此顶爆 atol。
# 实测:关掉之后 torch 对 fp32 真值的最大误差 0.5938 → 0.2577,与本 kernel 持平,999000 个元素零越界。
# 关掉它同时也让「相对 torch」的速度标尺公平:torch 不能靠少算精度赢。
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(kk, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def run(a, b):
    """launcher:分配输出、按用户的 tile 常量算 grid、把 stride 传进去。"""
    assert a.shape[1] == b.shape[0], f"K 不匹配:{tuple(a.shape)} @ {tuple(b.shape)}"
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, k.BLOCK_M), triton.cdiv(N, k.BLOCK_N))
    k.matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=k.BLOCK_M, BLOCK_N=k.BLOCK_N, BLOCK_K=k.BLOCK_K,
        num_warps=k.NUM_WARPS,
        num_stages=k.NUM_STAGES,
    )
    return c


def reference(a, b):
    return a @ b  # 同 dtype 的 cuBLAS(bf16 输入、fp32 累加);不要先 .float(),否则标尺虚高


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
