"""提占用率 —— 旋钮拧对了,但一点没变快

基线的体检单说「卡在寄存器上」:128×128 的累加器有 16384 个 fp32,NUM_WARPS=4
只有 128 个线程,光累加器就占 128 个寄存器/线程,每个 SM 只放得下 2 块,
warp 位置只用了 16%。

改一个常量:NUM_WARPS 4 → 8。代码一行没动。

5090 实测,旋钮确实生效了:

    每块线程数     128 → 256
    寄存器 / 线程  210 → 128
    warp 位置      16% → 31%(翻倍,和预期分毫不差)
    速度           205.2 → 205.2 TFLOPS(纹丝不动)

这一级真正要教的是:体检单说的「卡在谁身上」,说的是谁限制了占用率,
不是谁限制了性能。这里判定已经是「卡在算上、tensor core 85%」—— tensor core
喂饱了,再多 warp 也没活干。占用率只有在判定是「两头都没跑满、时间花在等上」
时才值得拧。

先看判定,再决定动哪个旋钮。这是整套体检单最容易读错的一处。
"""

import triton
import triton.language as tl

# 启动参数。接线读这几个常量去算 grid 并启动,你只管调它们。
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32
NUM_WARPS = 8      # 4 → 8:累加器摊到 256 个线程,每线程寄存器压力减半
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
