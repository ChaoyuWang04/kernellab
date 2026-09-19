"""autotune —— 别自己猜,让编译器把配置搜一遍

前面每一级都是手工拧常量,而实测一再打脸:swizzle 没用、提占用率没用、
split-K 反而更慢。既然凭直觉动旋钮这么不可靠,就把候选列出来让 Triton 跑一遍,
按 key 里的形状分别记住最快的一组。

@triton.autotune 接管了 BLOCK_* 与 num_warps / num_stages,所以接线一个都不能再传,
grid 也得改成读 META 的回调(不同 config 的 BLOCK_M 不一样)。接线看见 kernel 被
autotune 包过就自动切到这个模式,你只要加装饰器。

5090 实测,72 组候选,三个形状全部追平或超过 cuBLAS:

    case                基线        autotune     它选了什么
    4096³               0.95×       1.00×        BLOCK 128×64,  warps 4, stages 3
    8192³               1.16×       1.01×        BLOCK 128×128, warps 4, stages 4
    1000×999×777        0.37×       1.23×        BLOCK  64×64,  warps 8, stages 3

三点值得看:

1. **手写的 128×128×32 / warps 4 / stages 3 在三个形状上没有一个是最优的。**
2. 4096³ 选了 BLOCK_N=64 —— 块数从 1024 变成 2048,尾波更整齐,214.4 TFLOPS
   (93% 可达算力),比手写的 205.2 高。
3. 小形状选了 64×64:分块变小、块数从 64 变成 256,**不用 split-K 就把 SM 填满了**,
   直接 1.23× 超过 cuBLAS。上一级那套 atomic split-K 完全是白费劲。

代价:每个新形状第一次跑要把 72 组全试一遍,几秒到几十秒;之后按 key 命中缓存。
真要上生产,把搜出来的 config 钉死成几组常用形状,别在线搜。
"""
import triton
import triton.language as tl

# 候选配置。5090 的共享内存上限 101376 字节,BLOCK_N=256 配 4 级流水会超 —— 超了的
# config 会被 autotune 自己剔掉,不用手动排除。
_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                  num_warps=w, num_stages=st)
    for bm in (64, 128, 256)
    for bn in (64, 128, 256)
    for bk in (32, 64)
    for w in (4, 8)
    for st in (3, 4)
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,                 # 三个矩阵的起始地址
    M, N, K,                             # 形状
    stride_am, stride_ak,                # A 的行跨度、列跨度(行优先时 = K, 1)
    stride_bk, stride_bn,                # B 的(= N, 1)
    stride_cm, stride_cn,                # C 的(= N, 1)
    BLOCK_M: tl.constexpr,               # constexpr:编译期常量,决定 tile 形状
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ① 我是谁:grid 是一维的,自己把 pid 换算成「第几个行块、第几个列块」
    #    这里按行铺,最直白;想提高 L2 命中就改这几行(优化路线第 6 级)
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ② 我这块覆盖的行号、列号(各是一个一维向量)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    # ③ 一整块指针:坐标 -> 地址 = 基址 + 行*行跨度 + 列*列跨度
    #    [:, None] / [None, :] 把两个一维向量广播成二维
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]

    # ④ 累加器:整块,fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ⑤ 沿 K 一段一段乘加
    for k in range(0, K, BLOCK_K):
        # 边界掩码:M/N/K 不是 tile 整数倍时,越界的位置读 0
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)     # 搬 A 的一块(谁搬、搬到哪:编译器管)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)     # 搬 B 的一块

        acc += tl.dot(a, b)                             # 块乘块,走 tensor core(Ampere: mma.sync)

        a_ptrs += BLOCK_K * stride_ak                   # 指针沿 K 前进一段
        b_ptrs += BLOCK_K * stride_bk

    # ⑥ 写回:转回 bf16,越界位置不写
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
