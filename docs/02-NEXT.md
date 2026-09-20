# 接下来做什么

> 交接文档。**只写「还没做的」和「怎么做」**;现在的样子在 [00-START.md](00-START.md),操作流程在 [01-AGENT-PLAYBOOK.md](01-AGENT-PLAYBOOK.md),某条规矩为什么这么定在 [03-DECISIONS.md](03-DECISIONS.md)。

## 一、现在到哪了

一道题(matmul)、五种语言、四张卡、六档形状,全部上机验证过。参考答案 18 份。

这个矩阵是**两根轴**,完成度差很远:

**轴一 · 主线阶梯**(与架构无关的旋钮)

| 语言 | 级数 | 内容 |
|---|---|---|
| triton | 6 | naive → swizzle → occupancy → splitk → autotune → tma |
| cuda | 6 | naive → coalesce → smem → regtile → doublebuf → tensorcore(WMMA) |
| tilelang | 4 | naive → swizzle → tiles → splitk |
| cute | **3** | naive → atom → tiledcopy(2026-09-20) |
| tk | **2** | tiles → hopper-wgmma(2026-09-20) |

**轴二 · 换代**(Ampere → Hopper → Blackwell)

| 语言 | Ampere | Hopper | Blackwell | 性质 |
|---|---|---|---|---|
| triton | ✅ | ✅ | ✅ | 自动,已四卡实测 |
| tilelang | ✅ | ✅ | ⚠️ | 自动,但 0.1.14 在 B200 上退回 `mma.sync` |
| tk | ✅ | ✅ | ❌ | 换命名空间(`warp::` → `warpgroup::` → `tcgen05::`) |
| cute | ✅ | ❌ | ❌ | 换命名空间(`warp` → `warpgroup` → `tcgen05`) |
| cuda | ✅ | ❌ | ❌ | WMMA 只到 Ampere,再往上是 CUTLASS |

**轴二前两行是白送的** —— 代码一个字不改,我们的活只是去量。**后三行才要动手**,而且动手的量都不大:换名字、换积木、改参数。

## 二、待办清单

> 每一项都按「官方推荐的路」写,判定表见 [CLAUDE.md 定位节](../CLAUDE.md#定位写参考答案前必读)。
> **不手写 `wgmma` / `tcgen05` 的 PTX 协议,也不手写 `mma.sync` + `ldmatrix` 的 fragment 布局。**

### ~~1. cute × Ampere:从标量换成 atom~~ ✅ 2026-09-20

做完了,见 `problems/matmul/solutions/cute/1-atom.py`。三行代码(挑 op → `make_tiled_mma` → `cute.gemm`),
5090 上 **3.86 → 36.70 TFLOPS,快 9.5 倍**,`klab ptx` 从 `fma×1` 变成 `mma.sync×32`。

留下的结论供后面几级参考:

- **CuTe DSL 的 atom 不叫 `SM80_*`**(那是 C++ CuTe 的命名)。Python DSL 里是
  `cutlass.cute.nvgpu.<warp|warpgroup|tcgen05>.MmaF16BF16Op` —— **同一个类名,三个命名空间**,
  换代就是换一行 import。第 3、4 项照着改即可。
- 搬运在 `2-tiledcopy` 里换成了 TiledCopy(合并访存修好,等显存的 stall 降 3.5 倍),但整体只快 14%:
  瓶颈搬到了共享内存一侧(L1 56%→82%,mio_throttle 7%→30%),**显存全程只有 2%,从来不是它的问题**。
- **向量化与逐元素谓词冲突**:`num_bits_per_copy=128` 会让 `cute.copy` 要求按向量给谓词,
  而 K=777 的尾块跨在向量中间。CUTLASS 的解法是主循环 / 尾块拆两条路 —— 这是 cute 主线下一步。
- atom 对操作数布局有硬要求:A、B 都必须 K 连续。B 是行主序 (K,N),所以搬进共享内存时要转置成 (N,K)。

### ~~2. tk × Hopper~~ ✅ 2026-09-20

`problems/matmul/solutions/tk/1-hopper-wgmma.cu`。**整个仓库里换代最干净的一格**:
四行(两个 `rt_bf` fragment + 两次 `warp::load` + `warp::mma_AB`)变一行
`warpgroup::mma_AB(acc, As, Bs)`,操作数直接走共享内存描述符。

H100 实测:`mma.sync×32 / ldmatrix×20` → `wgmma×7 / ldmatrix×0`。**ldmatrix 整族消失**
就是「不再过寄存器」的直接证据。但速度只涨 3%~21%,**deepK 还倒退 30%** ——
因为 `mma_async_wait()` 紧跟在 `mma_AB` 后面,把异步硬用成了同步。

**照抄的官方源**:`envs/tk/ThunderKittens/kernels/gemm/educational_h100/`,
TK 自带 level_01..08 的 Hopper 阶梯,level_05 就是 WGMMA。**后面几级直接往下抄**:
level_06 = TMA + 双缓冲(正是本级欠的那一课),level_07 = work partitioning,
level_08 = 多 consumer warpgroup。

### 3. tk × Blackwell:`tcgen05::mma`

同上,入口在 `include/ops/group/mma/tcgen05.cuh`。只能在 `modal-b200` 上验。

### 4. cute × Hopper / Blackwell:换命名空间

**Hopper 那一级试过了,没成 —— 半成品在 `problems/matmul/wip/cute-hopper-wgmma.py`。**
编得过、跑得通、结果错 13%。文件头里列清了已验证正确的部分(共享内存内容、wgmma 的
计算值、坐标映射、写回覆盖率)与唯一对不上的那条线索(去掉 `fill(0.0)` 就 NaN,但
cosize 又说共享内存没空洞)。下次接手从那儿开始,不要从零重来。

**API 用法已经全部摸清**(见那个文件的头部):`make_trivial_tiled_mma`、
`make_smem_layout_a/b` 要拆 `outer`/`inner`、两个 slice 不能混用、fence/commit/wait
四步协议、ACCUMULATE 字段。

原计划的描述:

第 1 项做完之后,这两级就是把 atom 换掉:`SM80_*` → `SM90_*` → `SM100_*`。

atom 名字长这样 —— `SM90_64x128x16_F32BF16BF16_SS`:Hopper 的 / 一条指令算 64×128×16 / 累加 fp32 输入 bf16 / 两个输入都从共享内存来。**挑中它,发哪条 `wgmma`、描述符怎么编码、数据怎么摆,全在这块积木里。**

### 5. cuda × Hopper+:改 CUTLASS 的参数

**不手写 wgmma。** 裸 CUDA 到 Hopper 之后没有官方的手写封装(WMMA 只覆盖到 Ampere),NVIDIA 自己的答案就是 CUTLASS。这一格的练法是:拿一个 CUTLASS GEMM,改它的 **tile 形状 / schedule / atom**,然后量。

**这一项是唯一需要动 `envs/` 的** —— 现在 `envs/cuda/requirements.txt` 里只有 `numpy / ninja / pybind11`,CUTLASS 的 C++ 头还没弄到后端去。

现有的 cuda 阶梯到 `5-tensorcore`(WMMA)为止,那是 Ampere 的官方封装;**WMMA 不覆盖 Hopper 之后,这一点要在阶梯里写明** —— 这本身就是一条该学的结论。

### 6. 补厚主线阶梯

- **tilelang 还差 autotune**:`tilelang.autotune` 要求 kernel 自己拥有编译过程,和现有契约不同,要扩接线。
- **cute / tk 各还差几级**:swizzle、调分块、split-K 这类,照 triton 与 tilelang 的现成阶梯改。

### 7. 第二道题(新算子)

**用户明确说放在最后。** 建议 `softmax` 或 `layernorm`:

matmul 这道题**本质上触发不了**体检单「卡在搬数据上」那个判定分支(加了 `smallK` 那档想触发,实测是「两边吃得差不多」)。访存密集类算子的优化手段和 matmul 几乎没有重叠(融合、在线算法、warp shuffle 规约、persistent kernel),体检单有一半能力现在是闲置的。

再往后:flash attention(把前面全用上,含金量最高)。

## 三、已知的限制(别重复踩)

- **TK 要求形状是 tile 尺寸的整数倍**,所以 `specs/matmul_tk/` 的 case 比别人少一档(没有 1000x999x777)。这是 TK 的设计取舍,不是 bug。
- **TileLang 0.1.14 在 B200 上不用 tcgen05**,退回 `mma.sync`(Triton 3.8 会用)。想在 TileLang 上练 Blackwell,先确认新版本有没有支持。
- 其余的坑全在 [04-PITFALLS.md](04-PITFALLS.md),改相关代码前先看。

## 四、怎么验证一级写完了

1. `uv run pytest` 全绿
2. `klab run <算子> -s <序号-名字> --target <该级需要的卡>` —— check 全过、bench 有数
3. `klab ptx <算子> -s <序号-名字> --target <同上>` —— **确认指令真的换了**(换代类答案的唯一证据)

`--solution/-s` 让这两跑用 `problems/<题>/solutions/` 里的那一份,**`kernels/<名>/` 原样不动** —— 别再把答案拷进用户的目录验证了。
4. 把实测数字写进那一级的 docstring,**不编造单调递增的阶梯**。没收益就如实写没收益并解释为什么 —— 现成的反例见 [D8](03-DECISIONS.md#d8-参考答案的结论必须是实测的)
