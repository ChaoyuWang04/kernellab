"""朴素分块 —— 基线

一个 CTA 负责 C 的一块 BLOCK_M × BLOCK_N,沿 K 一段一段累加,pid 按行铺。
没有 autotune、没有 program 重排、没有 split-K。后面每一级只在这份上改一处。

5090 实测 4096³:0.95× torch,205.2 TFLOPS(89% 可达算力)。
判定「卡在算上」,tensor core 85% —— 已经是好状态。
体检单另外指出两处:warp 位置只用了 16%(卡在寄存器),尾波浪费约 25%。
下面两级就分别去动这两处,看看动了有没有用。
"""

import triton
import triton.language as tl

# 启动参数。接线读这几个常量去算 grid 并启动,你只管调它们。
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 4      # 一个 CTA 里几个 warp
NUM_STAGES = 3     # 共享内存流水线深度(Ampere 的 cp.async 多级缓冲)


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
