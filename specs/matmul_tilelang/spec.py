"""接线:kernels/matmul_tilelang/kernel.py 的 gemm(TileLang,bf16)。

用户只写 T.prim_func;jit 编译、按形状缓存、分配输出、启动都在这里。
TileLang 按形状特化,所以缓存键带上 M/N/K 与 dtype,一个 case 只编一次。
"""
import tilelang
import torch

from klab.harness.spec import kernel_module

k = kernel_module(__file__)

# 与 Triton 版同样的理由:cuBLAS 在块数喂不满 GPU 时走 split-K 并用 bf16 归约部分和,
# 会给接近 0 的输出带来约 0.1 的绝对误差,让 check 误判成算子写错。见 specs/matmul_triton_ampere/spec.py。
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

_cache: dict = {}


def make_inputs(case, device):
    m, n, kk = int(case["m"]), int(case["n"]), int(case["k"])
    dtype = getattr(torch, case.get("dtype", "bfloat16"))
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(m, kk, device=device, dtype=torch.float32, generator=g).to(dtype)
    b = torch.randn(kk, n, device=device, dtype=torch.float32, generator=g).to(dtype)
    return {"a": a, "b": b}


def _split_k() -> int:
    return int(getattr(k, "SPLIT_K", 1))


def _compiled(M, N, K, dtype, out_dtype):
    tunables = tuple(getattr(k, n, None) for n in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "NUM_STAGES", "THREADS"))
    key = (M, N, K, dtype, out_dtype, *tunables, _split_k())
    if key not in _cache:
        if hasattr(k, "build"):
            # 算子自己拥有编译过程。autotune 需要这个 —— 它要在候选之间反复编译与计时,
            # 光返回一个 prim_func 不够。契约:build(M, N, K, dtype, accum, out_dtype) -> 可调用的 kernel。
            _cache[key] = k.build(M, N, K, dtype, "float", out_dtype)
        else:
            # 普通情况让 TileLang 自己分配输出(out_idx=[-1]);
            # split-K 时输出必须由我们预先清零后传进去,所以不能标 out_idx。
            jit = tilelang.jit if _split_k() > 1 else (lambda f: tilelang.jit(out_idx=[-1])(f))

            @jit
            def build():
                return k.gemm(M, N, K, dtype, "float", out_dtype)
            _cache[key] = build()
    return _cache[key]


def run(a, b):
    """launcher。SPLIT_K > 1 时:多个 block 往同一块 C 上 atomic_add,输出必须是
    fp32 且预先清零 —— 与 Triton 版同一套约定,kernel 里定义常量就自动生效。"""
    M, K = a.shape
    N = b.shape[1]
    dtype = str(a.dtype).replace("torch.", "")
    if _split_k() > 1:
        out = torch.zeros((M, N), device=a.device, dtype=torch.float32)
        _compiled(M, N, K, dtype, "float")(a, b, out)
        return out.to(a.dtype)
    return _compiled(M, N, K, dtype, dtype)(a, b)


def reference(a, b):
    return a @ b  # 同 dtype 的 cuBLAS;不要先 .float(),否则标尺虚高


def workload(case, a, b):
    M, K = a.shape
    N = b.shape[1]
    e = a.element_size()
    return {"flops": 2 * M * N * K, "bytes": (M * K + K * N + M * N) * e}
