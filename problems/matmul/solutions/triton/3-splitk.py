"""split-K —— 教科书答案,在这张卡上是错的

前面几级的并行度只有 (M/BLOCK_M) × (N/BLOCK_N) 块。1000×999×777 只切出 8×8 = 64 块,
而 5090 有 170 个 SM —— 一多半 SM 从头到尾空转,所以基线只有 0.37× torch。
「块数不够就把 K 也切开」是教科书答案,cuBLAS 也确实这么做。这一级把它写出来。

做法:K 再切 SPLIT_K 份,并行度乘以 SPLIT_K;每块只算自己那一段的部分和,
用 tl.atomic_add 累加到同一块 C 上。

这是整条阶梯里唯一需要接线配合的一级:atomic_add 要求输出缓冲是 fp32 且预先清零,
最后再转回 bf16。接线看见模块里 SPLIT_K > 1 就自动这么做,你只要定义这个常量。
(cuBLAS 把这件事藏在库里,所以你看不见它换了输出缓冲。)

5090 实测 1000×999×777,越切越慢:

    基线(无 split-K)   0.0880 ms   0.37× torch
    SPLIT_K = 4         0.1003 ms   0.33×
    SPLIT_K = 8         0.1618 ms   0.20×
    SPLIT_K = 16        0.2652 ms   0.12×

4096³ 也从 0.95× 掉到 0.86% —— 大方阵本来就喂得满,split-K 纯属添乱。

为什么会输:全局 atomic 的流量把收益吃光了。输出 1M 个元素 × 4 字节,被原子累加
SPLIT_K 遍,再加最后一趟 fp32 → bf16 的转换。而这个形状总共才 1.55 GFLOP。
cuBLAS 的 split-K 不走全局原子,它用 workspace 存部分和再跑一个规约 kernel。

**真正的修法在下一级**:autotune 给这个形状选的是 BLOCK_M=64, BLOCK_N=64 ——
分块变小,块数从 64 变成 256,不用 split-K 就把 SM 填满了,结果 1.23× torch。
「块数不够就 split-K」是想当然;先试更小的分块。
"""
import triton
import triton.language as tl

# 启动参数。接线读这几个常量去算 grid 并启动,你只管调它们。
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 4      # 一个 CTA 里几个 warp
NUM_STAGES = 3     # 共享内存流水线深度(Ampere 的 cp.async 多级缓冲)
SPLIT_K = 4        # K 切成几份;>1 时接线会把输出换成 fp32 缓冲并清零


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
    SPLIT_K: tl.constexpr,
):
    # ① 我是谁:grid 多了一维 SPLIT_K,先把它剥出来
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_k = pid % SPLIT_K                  # 我负责 K 的第几段
    pid_mn = pid // SPLIT_K
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # ② 我这块覆盖的行号、列号(各是一个一维向量)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)    # 从我这段的起点开始

    # ③ 一整块指针:坐标 -> 地址 = 基址 + 行*行跨度 + 列*列跨度
    #    [:, None] / [None, :] 把两个一维向量广播成二维
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]

    # ④ 累加器:整块,fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ⑤ 只走属于我的那些 K 段:步长是 SPLIT_K * BLOCK_K
    for k in range(pid_k * BLOCK_K, K, SPLIT_K * BLOCK_K):
        # 边界掩码:M/N/K 不是 tile 整数倍时,越界的位置读 0
        offs = k + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs[None, :] < K)
        b_mask = (offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)     # 搬 A 的一块(谁搬、搬到哪:编译器管)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)     # 搬 B 的一块

        acc += tl.dot(a, b)                             # 块乘块,走 tensor core(Ampere: mma.sync)

        a_ptrs += SPLIT_K * BLOCK_K * stride_ak         # 跳过别人负责的那几段
        b_ptrs += SPLIT_K * BLOCK_K * stride_bk

    # ⑥ 写回:多个 block 往同一块 C 上加,必须用 atomic。C 是 fp32 缓冲,接线负责转回 bf16
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask)
