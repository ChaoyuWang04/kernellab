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

**接着做了 `2-hopper-tma`**(照 level_06):TMA + 双缓冲补上流水,4096³ 从 136 →
**319.54 TFLOPS**(0.17× → 0.41× torch),deepK 从倒退的 16.52 → 47.58。
`ld.global` 整族消失,换成 `cp.async.bulk×8 + mbarrier×8`。

**照抄的官方源**:`envs/tk/ThunderKittens/kernels/gemm/educational_h100/`,
TK 自带 level_01..08 的 Hopper 阶梯。**还剩两级可抄**:level_07 = work partitioning,
level_08 = 多 consumer warpgroup(生产者 warp 专职搬、消费者 warpgroup 专职算)。

### ~~3. tk × Blackwell~~ ✅ 2026-09-20

`problems/matmul/solutions/tk/3-blackwell-tcgen05.cu`,照抄 `educational_b200/level_06.cu`。
B200 实测 **294.98 TFLOPS**(官方 README 标 293,对上了),`tcgen05×46`,五档全过。
相对 0-tiles 快 1.73×,但只有 torch 的 0.20% —— cuBLAS 在 B200 上是 ~1510 TFLOPS。

换代换的不只是命名空间:**累加器从寄存器搬进了 tensor memory**(`tt<float,128,128>`),
协议也从「一句 mma」变成「三个信号量 + mm/mma + commit + tensor_load_wait」。

**官方阶梯还剩三级可抄**(README 里有实测值):level_07 warp specialization 731 TFLOPs、
level_08 epilogue 流水 1050、level_09 2-CTA cluster 1285。

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

### ~~5. cuda × Hopper+:改 CUTLASS 的参数~~ ✅ 2026-09-20

`problems/matmul/solutions/cuda/6-cutlass-hopper.cu`。**这一格是「学优化不是学手搓」那条原则的最好例证**:
一行 kernel 代码没写,只填了三行模板参数(TileShape / ClusterShape / KernelSchedule),
H100 上从手写 WMMA 的 43.72 TFLOPS 到 **482.15 TFLOPS,快 11 倍**,六档全过。

`klab ptx`:`wmma.mma×8`(Ampere)→ `wgmma×14 + cp.async.bulk×3 + mbarrier×32 + setmaxnreg×2`(Hopper)。
三样 Hopper 特性一次到齐,全是 `CollectiveBuilder` 推出来的。

**基建**:`envs/cuda/requirements.txt` 加了 `nvidia-cutlass`(只要 C++ 头,pip 装,不用 git clone);
`cppext.load_extension(..., cutlass=True)` 负责 include 路径与 `compute_90a`;
`specs/matmul_cuda/meta.toml` 的 `kernel_regex` 扩成 `matmul_kernel|device_kernel`。

**还能往下走**:Blackwell 只要把 `cutlass::arch::Sm90` 换成 `Sm100`、TileShape/ClusterShape 调一下。

### 6. 补厚主线阶梯

- ~~tilelang autotune~~ ✅ 2026-09-20。接线扩了一条分支:算子模块有 `build()` 就用它
  (`build(M,N,K,dtype,accum,out_dtype) -> 可调用的 kernel`),没有就照旧 jit `gemm()`。
  实测:手调那组在 4096³ 上追平 autotune,但 deepK 上 0.15× vs 0.55×(差 3.6 倍)。
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
