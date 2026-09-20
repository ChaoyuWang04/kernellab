# 接下来做什么

> 这是交接文档。新会话按 [CLAUDE.md](../CLAUDE.md) → [00-START.md](00-START.md) → 本页的顺序读,就能接着干。
> **本页只写「还没做的」和「怎么做」**;已经做完的样子在 00-START,操作流程在 [01-AGENT-PLAYBOOK.md](01-AGENT-PLAYBOOK.md)。

## 一、现在到哪了(一句话)

一道题(matmul)、五种语言、四张卡、六档形状,全部上机验证过。参考答案 18 份。
面板 `klab web` 是日常入口。**缺的是:三条语言的阶梯还薄,以及「换代」那一层只做了 Triton 的 TMA 一级。**

## 二、最重要的一条原则(决定每一级怎么写)

见 CLAUDE.md 的「这个仓库是干什么的」。一句话复述:

> **学的是「如何把算子优化到工业级」,不是「从零手搓一个工业级算子」。**
> 每种语言走它**官方推荐**的路;一级的价值在于「改几行 → 看数字怎么变」,不在于写起来多难。

所以「裸 CUDA 手写 wgmma」这种**不做**;「改 CUTLASS 的参数」才是那一格的正确答案。

## 三、待办清单(按建议顺序)

### 1. tk × Hopper:`warp::mma_AB` → `warpgroup::mma_AB`

**为什么先做它**:换代最直观 —— 就是换个命名空间,而且 TK 把 wgmma 的协议细节全封好了。

**怎么做**:
- 照抄官方示例的 tile 形状:`envs/tk/ThunderKittens/kernels/gemm/bf16_h100/bf16_h100_gemm.cu`
  - 关键行:`base_tile = st_bf<64,64>`;累加器 `rt_fl<16, N_BLOCK*64>`;
    `warpgroup::mma_AB(accum, A_shared, B_shared)` 之后 `warpgroup::mma_async_wait()`
- `warpgroup` 就是 `group<4>`(见 `include/ops/group/group.cuh:115`)
- wgmma 的操作数**直接从共享内存来**,不用先 `warp::load` 进寄存器 —— 这正是它比 `warp::mma_AB` 强的地方
- 形状约束在 `include/ops/group/mma/warpgroup.cuh:148` 的 static_assert 里,写之前先读

**注意**:wgmma 只有 sm_90 有,**5090 上编不过**。在 docstring 里写明「载入这一级前先把后端切到 modal-h100」,并借这个机会讲清楚 `requires.features` 门禁为什么存在。验证用 `klab ptx --target modal-h100`,应该看到 `wgmma`。

### 2. tk × Blackwell:`tcgen05::mma`

同上,入口在 `include/ops/group/mma/tcgen05.cuh`。只能在 `modal-b200` 上验。

### 3. cute × 各代:换 atom

CuTe 的换代方式就是换一块积木(atom):`SM80_*` → `SM90_*` → `SM100_*`。
atom 名字长这样:`SM90_64x128x16_F32BF16BF16_SS`(Hopper 的 / 一条指令算 64×128×16 / 累加 fp32 输入 bf16 / 两个输入都从共享内存来)。

现有的 `cute/0-naive.py` 是纯标量版(没用 atom),所以这一步同时补主线和换代。

### 4. cuda × Hopper+:改 CUTLASS 的参数

**不手写 wgmma。** NVIDIA 自己的答案就是用 CUTLASS —— 这一格的练法是拿一个 CUTLASS GEMM,改它的 tile 形状 / schedule / atom,然后量。
现有的 `cuda` 阶梯到 `5-tensorcore`(WMMA)为止,那是 Ampere 的官方封装;WMMA 不覆盖 Hopper 之后,这一点要在阶梯里写明。

### 5. 补厚主线阶梯(与架构无关的旋钮)

tilelang 还差 autotune(它的 `tilelang.autotune` 要求 kernel 自己拥有编译过程,和现有契约不同,要扩接线);cute 和 tk 各还差 ~4 级(swizzle / 调分块 / split-K 这类,照 triton 与 tilelang 的现成阶梯改)。

### 6. 第二道题(新算子)

**用户明确说放在最后。** 建议 `softmax` 或 `layernorm`:

matmul 这道题**本质上触发不了**体检单「卡在搬数据上」那个判定分支(我加了 `smallK` 那档想触发,实测是「两边吃得差不多」)。访存密集类算子的优化手段和 matmul 几乎没有重叠(融合、在线算法、warp shuffle 规约、persistent kernel),体检单有一半能力现在是闲置的。

再往后:flash attention(把前面全用上,含金量最高)。

## 四、已知的坑与限制(别重复踩)

- **`klab ptx` 接不上 cute**:CUTLASS 4.7.1 的 `JitFunctionArtifacts.PTX` 字段在但填不上(`DeviceTarget` 选项打开也是 None),`dump_to_object` 出来的是宿主 ELF,`cuobjdump` 抠不出。用户已同意**先不接**。
- **TK 要求形状是 tile 尺寸的整数倍**,所以 `specs/matmul_tk/` 的 case 比别人少一档(没有 1000x999x777)。这是 TK 的设计取舍,不是 bug。
- **TileLang 0.1.14 在 B200 上不用 tcgen05**,退回 `mma.sync`(Triton 3.8 会用)。想在 TileLang 上练 Blackwell,先确认新版本有没有支持。
- 其余工具链/后端层面的坑全在 00-START 第六节,改相关代码前先看。

## 五、怎么验证一级写完了

1. `uv run pytest` 全绿
2. `klab run <算子> --target <该级需要的卡>` —— check 全过、bench 有数
3. `klab ptx <算子> --target <同上>` —— **确认指令真的换了**(这是换代类答案的唯一证据)
4. 把实测数字写进那一级的 docstring,**不编造单调递增的阶梯**;没收益就如实写没收益,并解释为什么(现有的 `triton/2-occupancy`、`triton/3-splitk`、`tilelang/1-swizzle` 都是这么写的,照着学)
