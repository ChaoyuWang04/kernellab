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
| cute | **1** | 只有纯标量 naive |
| tk | **1** | 只有 `warp::mma_AB` |

**轴二 · 换代**(Ampere → Hopper → Blackwell)

| 语言 | Ampere | Hopper | Blackwell | 性质 |
|---|---|---|---|---|
| triton | ✅ | ✅ | ✅ | 自动,已四卡实测 |
| tilelang | ✅ | ✅ | ⚠️ | 自动,但 0.1.14 在 B200 上退回 `mma.sync` |
| tk | ✅ | ❌ | ❌ | 换命名空间 |
| cute | ❌ | ❌ | ❌ | 换 atom |
| cuda | ✅ | ❌ | ❌ | WMMA 只到 Ampere,再往上是 CUTLASS |

**轴二前两行是白送的** —— 代码一个字不改,我们的活只是去量。**后三行才要动手**,而且动手的量都不大:换名字、换积木、改参数。

## 二、待办清单

> 每一项都按「官方推荐的路」写,判定表见 [CLAUDE.md 定位节](../CLAUDE.md#定位写参考答案前必读)。
> **不手写 `wgmma` / `tcgen05` 的 PTX 协议,也不手写 `mma.sync` + `ldmatrix` 的 fragment 布局。**

### 1. cute × Ampere:从标量换成 atom ← 建议先做

**为什么先做它**:现在 `cute/0-naive.py` 是**用 CUTLASS 的皮、没用 CUTLASS 的肉** —— 线程索引 + `for k` 累加,和裸 CUDA 第 0 级一样,实测 3.9 TFLOPS(0.02× torch),五种语言垫底。而且 `SM80_*` atom 在默认后端 5090 上就能编能跑,**不用切卡**。这一步同时补两根轴。

**怎么做**:用 `TiledMma` / `TiledCopy` 描述「数据怎么切、谁搬哪块、怎么喂 tensor core」,atom 选 `SM80_16x8x16_F32BF16BF16F32_TN` 这一族。官方 dense GEMM 示例可以从 git 历史捞(见 [D13](03-DECISIONS.md#d13-历史里能捞的东西))。

**对照基线已经量好了**(2026-09-20,5090):现在这份 `0-naive` 的 PTX 只有 85 行,`fma×1 / ld.global×2 / st.global×1`,判定「没有任何标志指令」。atom 版跑出 `mma.sync` 就算这一级成了。

### 2. tk × Hopper:`warp::mma_AB` → `warpgroup::mma_AB`

**换代最直观的一格** —— 就是换个命名空间,TK 把 wgmma 的协议细节全封好了。

- 照抄官方示例:`envs/tk/ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`
- 关键行:`base_tile = st_bf<64,64>`;累加器 `rt_fl<16, N_BLOCK*64>`;`warpgroup::mma_AB(accum, A_shared, B_shared)` 之后要 `warpgroup::mma_async_wait()`
- `warpgroup` 就是 `group<4>`(`include/ops/group/group.cuh:115`)
- wgmma 的操作数**直接从共享内存来**,不用先 `warp::load` 进寄存器 —— 这正是它比 `warp::mma_AB` 强的地方
- 形状约束在 `include/ops/group/mma/warpgroup.cuh:148` 的 static_assert 里,写之前先读

**注意**:wgmma 只有 sm_90 有,**5090 上编不过**。docstring 里写明「载入这一级前先把后端切到 `modal-h100`」,并借这个机会讲清楚 `requires.features` 门禁为什么存在。验证:`klab ptx --target modal-h100`,应该看到 `wgmma`。

### 3. tk × Blackwell:`tcgen05::mma`

同上,入口在 `include/ops/group/mma/tcgen05.cuh`。只能在 `modal-b200` 上验。

### 4. cute × Hopper / Blackwell:换 atom

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
