# 接下来做什么

> 交接文档。**只写「还没做的」和「怎么做」**;现在的样子在 [00-START.md](00-START.md),操作流程在 [01-AGENT-PLAYBOOK.md](01-AGENT-PLAYBOOK.md),某条规矩为什么这么定在 [03-DECISIONS.md](03-DECISIONS.md)。

## 一、现在到哪了

一道题(matmul)、五种语言、四张卡、六档形状。**参考答案 30 份,全部上机验证过。**

| 语言 | 级数 | 内容 |
|---|---|---|
| cuda | 8 | naive → coalesce → smem → regtile → doublebuf → tensorcore → cutlass-hopper → cutlass-blackwell |
| tk | 8 | tiles → hopper-wgmma → hopper-tma →(2b-warpspec,负结果)→ blackwell: tcgen05 → warpspec → epilogue → cluster |
| triton | 6 | naive → swizzle → occupancy → splitk → autotune → tma |
| tilelang | 5 | naive → swizzle → tiles → splitk → autotune |
| cute | 3 | naive → atom → tiledcopy |

**换代矩阵 15 格,13 格有答案。** 空着的两格是 cute 的 Hopper / Blackwell,卡在同一个未解问题上
(半成品在 `problems/matmul/wip/`,见第 2 项)。

### 最值钱的两条完整阶梯

**B200 上,tk 一级一级爬到 cuBLAS 边上:**

| | 做了什么 | TFLOPS | ×torch | 本级增益 |
|---|---|---|---|---|
| `0-tiles` | mma.sync | 170.98 | 0.11× | — |
| `3-blackwell-tcgen05` | 换上 tcgen05 | 294.98 | 0.20× | 1.7× |
| `4-blackwell-warpspec` | 搬算分家 | 710.32 | 0.46× | 2.4× |
| `5-blackwell-epilogue` | 写回也流水 | 1056.83 | 0.70× | 1.5× |
| `6-blackwell-cluster` | 两个 block 合伙 | **1355.73** | **0.91×** | 1.3× |

**换指令只值 1.7 倍,后面「怎么喂」的三级合起来值 4.6 倍。** 8192³ 上这一级到 1614.65 TFLOPS,**0.99× cuBLAS**。

**同一张 B200,CUTLASS 改四行参数直接就在那儿:** `cuda/7-cutlass-blackwell` = 1387.71 TFLOPS(0.92×)。
两条路都该走 —— 一条告诉你那 4.6 倍由什么构成,一条告诉你工业级实现长什么样。

### 三条反例(阶梯不是单调的)

- **`tk/2b-hopper-warpspec`**:照抄官方 educational_h100,**慢 3.5 倍**。体检单:寄存器溢出 3564 万次、
  共享内存 232.6 KB 让每 SM 只放 1 块。信号事后很明显 —— b200 的 README 每级都标 TFLOPs,h100 的一个都没有。
- **`deepK` 在 B200 阶梯上一路 30 → 68 → 63 → 48**:每一级都在把分块开大,而它 M/N 只有 512,
  到最后只切出 4 个 block(B200 有 148 个 SM)。
- **`smallK` 从 `2-hopper-tma` 起就钉在 72–84 TFLOPS**:K=64 只有一轮 K 循环,流水/预取/分工全部失效。

## 二、待办清单

> 按「官方推荐的路」写,判定表见 [CLAUDE.md 定位节](../CLAUDE.md#定位写参考答案前必读)。
> **不手写 `wgmma` / `tcgen05` 的 PTX 协议,也不手写 `mma.sync` + `ldmatrix` 的 fragment 布局。**
>
> 矩阵乘这道题**只剩下面第 1 项**;它解决了,15 个格子就全满。第 2 项是新算子。
> 已完成的 30 份答案不在这里列 —— `git log --oneline` 一眼能看完,每一份的结论都在自己的 docstring 里。

### 1. cute × Hopper / Blackwell —— 唯一没解决的格子

**状态**:编得过、跑得通、**结果错 13%**。半成品在 `problems/matmul/wip/cute-hopper-wgmma.py`,
文件头列清了已验证正确的五件事,**不要重新查那些**。

**下次接手请先验这个假设(2026-09-24 提出,还没试过)**:

> **swizzle 只在一侧生效。**

推理链:

1. wgmma 要求 swizzle 挂在**指针**上、layout 保持仿射 —— 编译器会直接报
   `Expected affine layout ... use recast_ptr`。所以代码里把 `make_smem_layout_a/b`
   返回的 `ComposedLayout` 拆成了 `allocate_tensor(dtype, lA.outer, swizzle=lA.inner)`。
2. 我写 `sA0[m,k]` 时,地址是 `swizzle(layout(m,k))`;wgmma 的描述符**自己在硬件里
   编码了一个 swizzle 模式**。如果这两个不一致(比如 `make_tiled_copy_tv` 的
   `partition_D` 组合时把它吃掉了,或者描述符是从**剥掉 swizzle 的那个 layout** 推出来的),
   两边就各读各的排列。
3. **swizzle 本质是对几个地址位做 XOR —— 那几位本来是 0 时,XOR 是空操作。**
   这解释了一个当时看着矛盾的现象:手算验证过的 `acc[0..3]` 对应 `C[0,0]/C[0,1]/C[8,0]/C[8,1]`,
   **全在 tile 左上角,正是 XOR 不生效的区域**。「抽查对了」和「整体错 13%」不冲突,反而互证。
4. 它还能解释那条唯一对不上的矛盾:**去掉 `fill(0.0)` 就 NaN**(说明 wgmma 读到了没写过的地方),
   而 `cosize` 又说共享内存没有空洞 —— 如果两边的地址函数不同,这两条就同时成立了。

**怎么验**:把完整的 `ComposedLayout` 直接交给拷贝的目的端(而不是拆开后的 `.outer`),
看错误率变不变。一次跑就能证伪:**变了 → 方向对;一点不变 → 排除掉,写进文件头**。

**再不行的退路**:去 NVIDIA/cutlass 仓库 `examples/python/` 找一份官方的 CuTe DSL
Hopper dense GEMM 逐行比对(pip 包里只有 C++ 示例,没有 Python 的)。

**做完这一格顺手把 Blackwell 那格也补上** —— 按 `editorial.md` 第 8 节那张表,
只是把 `warpgroup.MmaF16BF16Op` 换成 `tcgen05.MmaF16BF16Op`。

### 2. 第二道题(新算子)

**用户明确说放在最后。** 建议 `softmax` 或 `layernorm`:

matmul 这道题**本质上触发不了**体检单「卡在搬数据上」那个判定分支(加了 `smallK` 那档想触发,
实测是「两边吃得差不多」)。访存密集类算子的优化手段和 matmul 几乎没有重叠(融合、在线算法、
warp shuffle 规约、persistent kernel),体检单有一半能力现在是闲置的。

再往后:flash attention(把前面全用上,含金量最高)。

**起手式**:照 [01-AGENT-PLAYBOOK.md](01-AGENT-PLAYBOOK.md) 第 1 节生成 `specs/<名>/`,
case 要挑**不同画像的形状**(同尺寸重复没有意义),然后从 `problems/<题>/problem.md` 与
`backbone/` 开始 —— 五种语言的骨架都要有,面板的「↺ 重置」靠它。

## 三、已知的限制(别重复踩)

- **TK 要求形状是 tile 尺寸的整数倍**,所以 `specs/matmul_tk/` 的 case 比别人少一档
  (没有 1000x999x777)。这是 TK 的设计取舍,不是 bug。
- **TileLang 0.1.14 在 B200 上不用 tcgen05**,退回 `mma.sync`(Triton 3.8 会用)。
  想在 TileLang 上练 Blackwell,先确认新版本有没有支持。
- **Hopper 的 TMA 要求行距是 16 字节的整数倍**,而 Hopper 上没有非 TMA 的通路
  (把 `Alignment` 降到 1 也没用)。`1000x999x777` 只能 pad 着算 —— 见
  `cuda/6-cutlass-hopper.cu` 的 `matmul_launch`。
- 其余的坑全在 [04-PITFALLS.md](04-PITFALLS.md),改相关代码前先看。**其中两条是
  「会给出看起来合理但完全错误的结论」那一类**(cppext 的构建缓存、klab ptx 找 .so 的方式),
  值得先扫一眼。

## 四、怎么验证一级写完了

1. `uv run pytest` 全绿
2. `klab run <算子> -s <序号-名字> --target <该级需要的卡>` —— check 全过、bench 有数
3. `klab ptx <算子> -s <序号-名字> --target <同上>` —— **确认指令真的换了**(换代类答案的唯一证据)

`--solution/-s` 让这两跑用 `problems/<题>/solutions/` 里的那一份,**`kernels/<名>/` 原样不动** ——
别再把答案拷进用户的目录验证了。

4. 把实测数字写进那一级的 docstring,**不编造单调递增的阶梯**。没收益就如实写没收益并解释为什么 ——
   现成的反例:`tk/2b-hopper-warpspec`(照抄官方慢 3.5 倍)、`cuda/2-smem`(慢 10%)、
   `triton/3-splitk`(越切越慢)、`tilelang/1-swizzle`(没用)。
5. 换代类的答案在讲解里写一行 **`需要: modal-h100`**(或 `modal-b200`),面板会挂徽章提醒切后端。
