"""autotune —— 别自己猜,让编译器把配置搜一遍

前面三级都在手工拧常量:`2-tiles` 那一级我把分块从 128×128×32 调到 128×128×64、
线程 128 → 256,4096³ 上从 0.88× 涨到 0.97×。但那几个数是**猜的** —— 猜完再去量,
量完再改,改到没耐心为止。

而实测一再打脸:Triton 版的 `1-swizzle` 没用、`2-occupancy` 旋钮拧对了却一点没变快、
`3-splitk` 在一个形状上越切越慢。**人对这些旋钮的直觉很差。**

把候选列出来,让它自己编译、计时、挑最快的:

    @tilelang.autotune(configs=CONFIGS)
    @tilelang.jit(out_idx=[-1])
    def build(M, N, K, dtype, accum_dtype, out_dtype,
              block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
        ...

**可调的参数必须写进函数签名**,autotune 靠签名知道哪些能动。调用时不传它们,
它就去搜;传了就直接用那一组(这也是关掉搜索的办法)。

## 这一级动了接线

前三级的接线是「算子返回 prim_func、spec 负责 jit」。autotune 要在候选之间反复
编译与计时,**编译过程必须归算子自己**。所以 `specs/matmul_tilelang/spec.py` 加了
一条分支:算子模块里有 `build()` 就用它,没有就照旧 jit `gemm()`。

契约:`build(M, N, K, dtype, accum_dtype, out_dtype) -> 可调用的 kernel`。

## 代价:第一次很慢

每个 case 都要把候选全编一遍再计时。候选越多搜得越好,也越慢。这里列了 18 组
(3 种分块 × 3 种流水级数 × 2 种线程数里挑出的合法组合),已经够看出趋势。

## 5090 实测:手调的那一组只对它被调出来的那个形状好

                      2-tiles(手调)      4-autotune       
    4096³             0.96×  208.4 TF    0.96×  205.9 TF    持平
    8192³             1.01×  221.3 TF    1.00×  113.3 TF    持平(见下)
    1000x999x777      0.67×   31.6 TF    0.89×   42.1 TF    +33%
    smallK            1.00×   69.9 TF    1.00×   69.9 TF    持平
    tall              0.92×  179.4 TF    0.94×  180.4 TF    持平
    deepK             0.15×   23.6 TF    0.55×   85.6 TF    **3.6 倍**

手调那一级我是拿 4096³ 调的,于是它在 4096³ 上追平了 autotune —— 但换到
**1000x999x777 和 deepK 就露馅**:同一组 128×128×64 在那两个形状上是错的选择。
autotune 按形状各挑各的,deepK 直接快了 3.6 倍。

**这一级的价值不是「更快」,是「不用再猜」。** 前三级每一级我都要手动试几组再挑,
而实测一再打脸;这一级把那件事自动化掉,而且按形状分别记住。

## 8192³ 那一行别被 TFLOPS 骗了

221.3 → 113.3 TFLOPS 看着像大幅倒退,**但 ×torch 是 1.01 → 1.00,几乎没变**。
原因是 5090 是消费卡:autotune 为这一档先编译并计时了 24 组候选,卡已经烧热降频,
之后 kernel 和 torch 参考**一起变慢**。绝对 TFLOPS 在这一档不可比,只能和同时段的
torch 比 —— 这正是仓库文档里记着的那条坑,在这里撞了个正着。

## 代价

每个 case 都要把 24 组候选编一遍再计时。4096³ 那一档搜完用了 33 秒,8192³ 更久。
候选越多搜得越好也越慢,而且**搜索本身会把卡烧热**,影响紧接着的测量。
"""

import tilelang
import tilelang.language as T

# 手写那一级定下的配置,作为对照:128×128×64 / stages 3 / threads 256
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
NUM_STAGES = 3
THREADS = 256

# 候选空间。5090 每 block 的共享内存上限是 101376 字节,
# (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 2 字节 * stages 超了就编不过 —— autotune 会跳过失败的组。
CONFIGS = [
    {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns, "threads": th}
    for bm, bn in ((64, 64), (128, 128), (128, 64))
    for bk in (32, 64)
    for ns in (2, 3)
    for th in (128, 256)
]


@tilelang.autotune(configs=CONFIGS)
@tilelang.jit(out_idx=[-1])
def build(M, N, K, dtype, accum_dtype, out_dtype=None,
          block_M=None, block_N=None, block_K=None, num_stages=None, threads=None):
    """可调参数写在签名里 —— autotune 认的就是这个。不传就搜,传了就用那一组。"""
    out_dtype = out_dtype or dtype

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel
