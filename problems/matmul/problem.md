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

系统按 `grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))` 启动,所以改 tile 形状 grid 会自动跟着变。

`M`、`N`、`K` 不保证是分块大小的整数倍 —— 越界的位置必须处理掉,否则会读到别人的内存或写坏结果。

## 评判

| 维度 | 标准 |
|---|---|
| 正确性 | 与同 dtype 的 `torch.matmul` 逐元素比对,`atol=1e-1, rtol=1e-2` |
| 速度 | 相对 `torch.matmul`(背后是 cuBLAS)的倍数,以及对本卡可达峰值的百分比 |
| 剖析 | NCU 体检单:瓶颈判定、各单元利用率、占用率与限制因子、stall 归因 |

Submit 之后左边的「结果」页会出体检单。先看**判定**那一行,再看它指向的那一段。

## 测试用例

| case | 形状 | 考什么 |
|---|---|---|
| `4096-bf16` | 4096³ | 主战场。块数喂得满 GPU,数字最稳,NCU 默认抓这一档 |
| `8192-bf16` | 8192³ | 规模上去以后 L2 复用垮掉的样子。注意 5090 是消费卡,持续满载会降频,这一档的绝对值不可比 |
| `1000x999x777-bf16` | 非整数倍 | 专考边界掩码。这一档时间没有意义,只看对不对 |

## 一道题,四种语言

同样的数学、同样的 case 与容差,分别用 Triton / TileLang / CuTe DSL / 裸 CUDA 写一遍,数字可以直接横着比 —— 差距来自算法、分块、访存模式,还是 DSL 本身在这张卡上的降级,一比就知道。右上角的下拉切语言。
