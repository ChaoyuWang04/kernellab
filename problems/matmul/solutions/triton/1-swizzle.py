"""L2 swizzle —— 换一种 pid 到 tile 的铺法

按行铺时,同时在跑的那一批 block 横跨很宽的一条,它们要的 B 列几乎没有交集,
L2 白白浪费。改成按 GROUP_M 行分组再铺,同批 block 就会共用 tile。

只改了 ① 那几行:grid 还是一维,launcher 一个字没动。这正是 grid 设计成一维的
理由 —— 换映射方式是 kernel 自己的事。GROUP_M 是新加的 constexpr,
只要在模块里定义同名常量,接线会自动把它传进去。

5090 实测 4096³:206.5 vs 基线 205.2 TFLOPS —— +0.6%,在噪声里。
原因看体检单就知道:L2 命中率本来就有 95%。4096³ 的 B 只有 32 MB,整个装得进
5090 的 96 MB L2,没有可省的。这一级要到 L2 装不下的尺寸才见效。
先看 L2 命中率再决定做不做这一级。
"""

import triton
import triton.language as tl

# 启动参数。接线读这几个常量去算 grid 并启动,你只管调它们。
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 4      # 一个 CTA 里几个 warp
NUM_STAGES = 3     # 共享内存流水线深度(Ampere 的 cp.async 多级缓冲)
GROUP_M = 8        # 每组塞几个行块;越大 L2 复用越好,但尾波越不齐


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
    GROUP_M: tl.constexpr,
):
    # ① 我是谁:按 GROUP_M 行块分组铺,同一组内先竖着走完再换列
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # 最后一组可能不满
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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
