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


def _autotuned() -> bool:
    """@triton.autotune 包过的 kernel:tile 常量与 num_warps/num_stages 由它自己填,接线不能再传。"""
    return hasattr(k.matmul_kernel, "configs")


def _constexprs():
    """kernel 签名里所有全大写的编译期参数,从用户模块的同名常量取值。

    这样加一个新旋钮(比如 swizzle 的 GROUP_M、split-K 的 SPLIT_K)只需要在 kernel.py 里
    声明参数 + 定义常量,launcher 不用跟着改 —— 「每一级只改 kernel」这条才站得住。
    """
    names = getattr(k.matmul_kernel, "arg_names", [])
    return {n: getattr(k, n) for n in names if n.isupper() and hasattr(k, n)}


def run(a, b):
    """launcher:分配输出、算 grid、把 stride 传进去。

    三种形态都要接住,靠读 kernel 模块里的约定,不靠改这里:
      · 普通       —— 从模块常量取 tile 与 num_warps/num_stages
      · SPLIT_K>1  —— K 也切开,多个 block 往同一块 C 上 atomic_add,
                      所以输出缓冲必须是 fp32 且预先清零,最后再转回 bf16
      · autotune   —— 常量由装饰器填,接线一个都不能传;grid 改成读 META 的回调
    """
    assert a.shape[1] == b.shape[0], f"K 不匹配:{tuple(a.shape)} @ {tuple(b.shape)}"
    M, K = a.shape
    N = b.shape[1]
    split_k = int(getattr(k, "SPLIT_K", 1))

    if split_k > 1:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)   # atomic_add 要 fp32,且要清零
    else:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        # 一维 grid:pid -> 哪一块 C 由 kernel 自己决定(按行铺 / swizzle 分组铺都只改 kernel)
        bm = meta.get("BLOCK_M", getattr(k, "BLOCK_M", 128))
        bn = meta.get("BLOCK_N", getattr(k, "BLOCK_N", 128))
        return (triton.cdiv(M, bm) * triton.cdiv(N, bn) * split_k,)

    extra = {} if _autotuned() else {**_constexprs(),
                                     "num_warps": k.NUM_WARPS, "num_stages": k.NUM_STAGES}
    k.matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **extra,
    )
    return c.to(a.dtype) if split_k > 1 else c


def reference(a, b):
    return a @ b  # 同 dtype 的 cuBLAS(bf16 输入、fp32 累加);不要先 .float(),否则标尺虚高


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
