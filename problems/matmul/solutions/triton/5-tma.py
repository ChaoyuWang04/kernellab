"""TMA —— Hopper 起的批量异步拷贝,以及「用了 API」不等于「用了硬件」

前面每一级都在手工算指针、手工写掩码:

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

TMA 把这三件事全交给硬件:描述一次「这个张量长什么样、我要多大的块」,之后只报坐标,
地址计算、越界补零、异步搬运都是 DMA 引擎的事。指针算术和掩码从 kernel 里彻底消失。

    desc = tl.make_tensor_descriptor(A_ptr, shape=[M, K], strides=[stride_am, 1],
                                     block_shape=[BLOCK_M, BLOCK_K])
    a = tl.load_tensor_descriptor(desc, [pid_m * BLOCK_M, k])     # 越界自动补 0

=== 这一级真正要教的事 ===

同一份源码,四张卡,klab ptx 实测:

    卡      架构      搬运指令              tensor core      TFLOPS
    A100    sm_80     cp.async×35  ← 普通!  mma.sync×64       163.8
    5090    sm_120    cp.async.bulk×15      mma.sync×64       187.5
    H100    sm_90     cp.async.bulk×15      wgmma×8           558.4
    B200    sm_100    cp.async.bulk×17      tcgen05×17        906.9

**A100 上它没报错,也没变慢,它只是不是 TMA。** sm_80 没有这个硬件,Triton
把描述符 API 静默降级成了普通的 cp.async —— 代码照跑、结果照对,你完全看不出来。

这就是 klab ptx 存在的唯一理由:**体检单会告诉你 tensor core 忙到几成,
但它不会告诉你走的是 mma.sync 还是 wgmma 还是 tcgen05;
「我写了 TMA」和「硬件真的用了 TMA」是两件必须分开验的事。**

顺带看清了三代的分界线:
  · TMA(cp.async.bulk):sm_90 起。消费级 Blackwell(5090)也有,虽然它没有 wgmma
  · wgmma:只有 Hopper。一条顶 8 条 mma.sync —— warpgroup 一次算一大块
  · tcgen05:只有数据中心 Blackwell。连 ldmatrix 都基本消失了(累加器住进 tmem,
    不再经寄存器和共享内存往返)

=== 关于速度 ===

别被上表的 TFLOPS 误导成「TMA 让它变快了」:这一级顺手把 BLOCK_K 改成 64、
NUM_WARPS 改成 8(TMA 的 box 内维要凑 16 字节整数倍),不是 TMA 的纯 A/B。
相对 torch 反而都比基线低(5090 0.87× vs 0.95×)—— TMA 省的是整数指令,
而基线本来就卡在 tensor core 上,省整数指令没用。

TMA 真正值钱的地方在 B200:906.9 vs 基线 729.3 TFLOPS,+24%。那张卡的
tcgen05 太快,喂数据才是瓶颈,这时候 DMA 引擎才派上用场。
"""
import triton
import triton.language as tl


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64       # TMA 的一个 box 内维要凑够 16 字节的整数倍;bf16 下 64 个元素 = 128 字节
NUM_WARPS = 8
NUM_STAGES = 3


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ① 我是谁:还是一维 grid 自己换算
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # ② 描述三个张量各是什么形状、我要多大的块。之后只报坐标,不碰指针
    a_desc = tl.make_tensor_descriptor(A_ptr, shape=[M, K], strides=[stride_am, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(B_ptr, shape=[K, N], strides=[stride_bk, 1],
                                       block_shape=[BLOCK_K, BLOCK_N])
    c_desc = tl.make_tensor_descriptor(C_ptr, shape=[M, N], strides=[stride_cm, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ③ 沿 K 累加。没有指针递增,没有掩码 —— 越界由 TMA 自己补 0
    for k in range(0, K, BLOCK_K):
        a = tl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_M, k])
        b = tl.load_tensor_descriptor(b_desc, [k, pid_n * BLOCK_N])
        acc += tl.dot(a, b)

    # ④ 写回也走 TMA
    tl.store_tensor_descriptor(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.bfloat16))
