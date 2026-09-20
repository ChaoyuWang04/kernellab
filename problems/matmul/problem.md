# 矩阵乘 (GEMM)

计算 `C = A @ B`,其中 `A` 是 `[M, K]`、`B` 是 `[K, N]`、`C` 是 `[M, N]`,全部行优先连续。

输入 `bf16`,**累加用 fp32**,输出转回 `bf16`。这是深度学习里 GEMM 的标准做法:bf16 只有 8 位尾数,直接用 bf16 累加几百上千项会把有效位吃光。

## 你只写 kernel 本体

造输入、对答案、计时、抓计数器都由系统做好了,**分配输出、算 grid、传 stride 这些样板也由系统做**,就像 LeetCode 不用你写读输入。你只写两样:

```python
BLOCK_M, BLOCK_N, BLOCK_K = ...   # tile 形状与启动参数,系统读它们去算 grid
NUM_WARPS, NUM_STAGES = ...

@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

系统按 `grid = (cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)` 启动一维 grid,所以改 tile 形状 grid 会自动跟着变;**pid 怎么换算成「第几块」是 kernel 自己的事**(换一种换算方式就是优化路线第 6 级的 L2 swizzle,不用动任何接线)。

其他语言的契约见各自的骨架:TileLang 写 `T.prim_func`,CUDA 写 `__global__` + `matmul_launch`,CuTe DSL 写 `@cute.kernel`。

`M`、`N`、`K` 不保证是分块大小的整数倍 —— 越界的位置必须处理掉,否则会读到别人的内存或写坏结果。

## 评判

| 维度 | 标准 |
|---|---|
| 正确性 | 与同 dtype 的 `torch.matmul` 逐元素比对,`atol=1e-1, rtol=1e-2` |
| 速度 | 相对 `torch.matmul`(背后是 cuBLAS)的倍数,以及对本卡可达峰值的百分比 |
| 剖析 | NCU 体检单:瓶颈判定、各单元利用率、占用率与限制因子、stall 归因 |

Submit 之后左边的「结果」页会出体检单。先看**判定**那一行,再看它指向的那一段。

## 测试用例

六档,每档暴露一种不同的瓶颈画像 —— 同尺寸方阵重复没有意义。算术强度从 62 到 2731 横跨两个数量级。

| case | 形状 | 算术强度 | 考什么 | 基线表现 |
|---|---|---|---|---|
| `4096-bf16` | 4096³ | 1365 | 主战场,块数喂得满,NCU 默认抓这一档 | 0.95× |
| `8192-bf16` | 8192³ | 2731 | 工作集 100 MB 装不进 L2;消费卡持续满载会降频 | 1.03× |
| `1000x999x777-bf16` | 非整数倍 | 304 | 边界掩码;块数只有 64,喂不满 | 0.37× |
| `smallK-...x64` | 4096×4096×64 | **62** | K 方向只有 2 步,时间被首尾开销主导;工作集住在 L2 里 | 1.55× |
| `tall-16384x512x4096` | 瘦长 | 443 | N 只有 512,列块少。真实模型的 QKV 投影就是这形状 | 0.90× |
| `deepK-512x512x16384` | 大 K 小 MN | 252 | **只切出 16 个块**,六档里最惨;split-K 唯一该赢的场景 | **0.14×** |

注意 `smallK` 那档:bench 报 1689 GB/s(94% 带宽),但 NCU 看**显存只有 3%**。两个数差 35 倍,因为它的工作集 34.6 MB 整个住在 5090 的 96 MB L2 里 —— bench 的 GB/s 是按 `workload()` 的**数学最小字节数**算的「有效带宽」,不是真的从显存搬了那么多。

## 一道题,五种语言

同样的数学、同样的 case 与容差,分别用 **Triton / TileLang / 裸 CUDA / CuTe DSL / ThunderKittens** 写一遍,数字可以直接横着比。编辑器头部的下拉切语言,右上角切后端。

5090 实测 4096³(各自的基线):

| 语言 | vs torch | TFLOPS |
|---|---|---|
| triton | **0.95×** | 206.5 |
| tilelang | 0.88×(调完分块 **0.97×**) | 187.5 → 209.1 |
| tk | 0.86× | 181.4 |
| cuda(手写 WMMA) | 0.15× | 31.6 |
| cute(朴素标量) | 0.02× | 3.9 |

差距来自算法、分块、访存模式,还是这门语言在这张卡上降到了哪一代指令 —— 用 `klab ptx` 一比就知道,见优化路线第 8 节那张四卡对照表。

(tk 少一档非整数倍的 case:ThunderKittens 的 tile 操作要求形状是 tile 尺寸的整数倍。)
