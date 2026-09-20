"""split-K —— 同一招,一个形状上越切越慢,另一个形状上快 5.6 倍

`(M/BLOCK_M) × (N/BLOCK_N)` 块要是比 SM 数还少,一多半 SM 从头空转。
把 K 也切成 SPLIT_K 段,并行度乘以 SPLIT_K,各段算部分和再用 tl.atomic_add 累加。

这是整条阶梯里唯一需要接线配合的一级:atomic_add 要求输出缓冲是 fp32 且预先清零,
最后再转回 bf16。接线看见模块里 SPLIT_K > 1 就自动这么做,你只要定义这个常量。
(cuBLAS 把这件事藏在库里,所以你看不见它换了输出缓冲。)

=== 5090 实测:两个 case,结论完全相反 ===

    SPLIT_K   1000×999×777          deepK 512×512×16384
    ------------------------------------------------------
    1(基线)  0.0880 ms  0.37×      0.3932 ms  0.14×
    4         0.1003 ms  0.33× ↓    0.1167 ms  0.47× ↑
    8         0.1618 ms  0.20× ↓    0.0737 ms  0.75× ↑
    16        0.2652 ms  0.12× ↓    0.0696 ms  0.79× ↑
                    越切越慢              越切越快,比基线快 5.6 倍

同一份代码、同一张卡、同一个旋钮,一边输得一塌糊涂,一边是整条阶梯最大的一次提升。

=== 判据 ===

split-K 的成本是**全局原子流量 = 输出元素数 × SPLIT_K × 4 字节**,收益是把闲置的
SM 用起来。所以要看这两者的比:

    case                输出元素   K        总计算量    原子代价  结论
    1000×999×777        999 K      777      1.6 GFLOP   大       输
    deepK 512×512×16384 262 K      16384    8.6 GFLOP   小       赢

deepK 的输出少 3.8 倍、计算多 5.5 倍,原子流量摊薄了 20 倍 —— 这才是 split-K 的主场。

**一句话:输出小、K 大,才用 split-K。** 光看「块数不够」就上是错的 ——
1000×999×777 也块数不够,但它的真正修法是更小的分块(见下一级 autotune,
它给那个形状选了 BLOCK 64×64,不用 split-K 就到了 1.23×)。

顺带一提 cuBLAS 的 split-K 不走全局原子,它用 workspace 存部分和再跑一个规约 kernel ——
这就是为什么它在 1000×999×777 上能赢而我们这版会输。
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
