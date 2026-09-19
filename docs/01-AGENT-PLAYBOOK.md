# Agent 操作手册:用户写了一个 kernel,我该怎么做

> 用户的工作流固定是四步:**在 VSCode 里写 kernel → 让 agent 接线并跑 → 拿到 NCU 与体检单 → 让 agent 评价这个 kernel 写得怎么样**。本手册按这四步写,是 agent 在这个仓库里的默认行为。系统怎么运转、改哪里、怎么验证,见 [00-START.md](00-START.md)。

## 0. 边界

- `kernels/<名>/` **是用户的**。里面只有算子源码(`kernel.py`、`kernel.cu`、任何 DSL,可以多文件)。**不改、不加文件**,除非用户明确要求;评价里可以引用它的行。
- `specs/<名>/` **是 agent 的**。`meta.toml`(工具链、case、容差、扫参)+ `spec.py`(四个契约函数)+ `baselines/`。用户不碰这里。
- `runs/` 是结果,`klab` 是工具。用户不需要知道它们的细节,除非问。

## 1. 接到「测一下这个 kernel」

**读源码,回答四个问题**,答不出就问用户,不猜:

| 问题 | 怎么判断 |
|---|---|
| 工具链是哪种 | `import triton` → `triton`;`import tilelang` → `tilelang`;`import cutlass.cute` → `cute`;`.cu` 里 `#include "kittens.cuh"` → `tk`;其他 `.cu` → `cuda` |
| 入口函数与签名 | 用户暴露的那个 Python 函数(或 `.cu` 里 pybind 导出的);它接什么张量、返回什么 |
| 输入的形状、dtype、布局约束 | 看 kernel 对形状的假设(整除、对齐、是否要求连续、B 是 (k,n) 还是 (n,k)) |
| 数学上等价的 torch 写法 | 这是 `reference()`,也是「相对 torch」标尺的来源;选最贴近的库调用(`torch.matmul`、`torch.softmax`、`F.layer_norm`…),**用与算子相同的 dtype 直接调**,不要先 `.float()`:那会多一次拷贝、把 GEMM 变成 fp32,标尺就虚高(matmul 曾因此显示 11× 快于 torch,修正后是 0.9×) |

> bf16 matmul 的参考还有一个坑:PyTorch 的 `allow_bf16_reduced_precision_reduction` 默认 True,块数喂不满 GPU 时(如 1000×999×777 只有 64 块、5090 有 170 个 SM)cuBLAS 走 split-K 并用 bf16 归约部分和,给**接近 0 的输出**带来约 0.1 的绝对误差,check 会误判成用户算子写错。在 `spec.py` 里 `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` 关掉;实测对速度的影响在 ±1% 噪声内,标尺不受影响。判断方法:拿 `a.float() @ b.float()` 当真值,看是算子离得远还是 torch 离得远。

**生成 `specs/<名>/meta.toml`**(直接抄 `specs/softmax/meta.toml` 或 `specs/matmul/meta.toml` 改):

- `toolchain`:上表的结果。
- `kernel_regex`:用户 kernel 的函数名(Triton / cuda / tk 就是 `@triton.jit` 或 `__global__` 的名字);TileLang 写 `gemm_kernel`;CuTe DSL 写 `cutlass`。写宽了 ncu 会把 torch 造输入的 kernel 抓进来、体检单取错行。
- `requires`:默认 `min_cc = 8.0, features = []`。只有源码明确用了 Hopper / B200 专属特性(wgmma、TMA、cluster、tcgen05)才写进 `features`,门禁据此拒绝 5090。
- `tolerance`:fp32 用 `1e-5`;fp16 / bf16 累加类(matmul、attention)用 `atol = 1e-1, rtol = 1e-2`;逐元素 fp16 用 `1e-2`。
- `cases`:两到三档规模,名字带尺寸与 dtype(如 `4096-f16`、`8192x1024-f16`);第一个 case 是 ncu 默认抓的那个,取中等规模。规模要让 kernel 时长在 0.05 ms 到 5 ms 之间,太小测的是发射开销。
- `[sweep]`:只在用户源码里有可调常量(分块、stages、warps)时加,参数名必须是用户模块里的全局常量名。

**生成 `specs/<名>/spec.py`**(抄同工具链的现成样板):

- Python 工具链:`k = kernel_module(__file__)` 拿到用户模块,`run()` 调用它的入口函数。
- cuda / tk:`load_extension("<名>", __file__, sources=["kernel.cu"], tk=<是否 TK>)`,`run()` 调用 pybind 导出的函数。
- `make_inputs` 用固定种子;dtype 从 case 取;不要在里面做用户 kernel 不做的预处理(比如转置),否则测的不是用户的算子。
- `workload` 的 flops 与 bytes 按数学定义算,不按实现算:matmul 是 `2MNK` 与 `(MK+KN+MN)×元素字节`;逐元素是读写各一遍。
- 有 `[sweep]` 就加 `configure(**p)` 转发给用户模块的 `configure`,用户模块没有就不要加 `[sweep]`。

**本地自检**:`uv run pytest`(契约、工具链登记、目录结构)。过了再上机。

## 2. 选后端并跑

规则写在 `targets.toml` 注释里,复述一遍:**默认 `5090home`**;`requires.features` 里有 5090 没有的特性,或用户点名要 H100 / B200,才用 `modal-h100`(或再加一段配置换卡型)。用户点名永远优先。价格不是考虑因素(用户 2026-09-18 说明)。

一条命令:

```bash
uv run klab run <名>                       # 默认后端:check → bench → ncu(第一个 case)→ 体检单
uv run klab run <名> --target modal-h100   # 换后端,其余不变
```

失败时看输出停在哪一层:

| 现象 | 含义 | 处理 |
|---|---|---|
| `[requires] … 不满足` | 门禁:这张卡没有算子要的特性 | 换 `--target`;做跨代对照才加 `--ignore-requires` |
| 编译错误(nvcc / Triton / TileLang 报错) | 用户算子或接线编不过 | 先分清是接线还是算子:接线错自己改 `spec.py`;算子错把报错原文和行号给用户,不擅自改 `kernels/` |
| `[check] … FAIL` | 结果与 `reference()` 不符 | 先怀疑接线(布局、dtype、参考实现选错),再怀疑算子;`max_abs_err` 量级能说明是精度还是逻辑错 |
| `没有产出结果` | harness 崩在更早处 | 用 `klab exec "<命令>"` 到后端复现 |
| Modal `Could not connect` | 网络 | 直连/代理选路已自动,再失败就报给用户 |

第一次跑一个新工具链的算子时,如果该后端还没装它:`klab setup --target <后端> --toolchain <工具链>`。

## 3. 读结果

跑完看三样,按顺序:

1. **`runs/<id>/report.md`(体检单)**,先读「判定」,再读判定对应的那一段。
2. **`klab compare <名>`**:同一算子在其他后端、或仓库里同一数学的其他实现(`matmul` / `matmul_tl` / `matmul_cute`)的数字。
3. **用户源码**:把指标对回代码。寄存器数对应累加器和分块常量;共享内存对应 `BLOCK_*` 与 stages;stall 原因对应访存与同步的写法。
4. **`klab ptx <名>`**(用户在练某一代的特性时必看):确认 kernel 真的降到了那一代的标志指令。体检单答不了这个 —— 它只说 Tensor pipe 用了多少,不说走的是 `mma.sync` 还是 `wgmma`。命中世代与目标架构不符时,先怀疑 DSL 降级(Triton 在 sm_120 上没有 wgmma,退回 mma.sync)或算子根本没走 tensor core(`tl.dot` / `T.gemm` 写法不对)。

指标 → 常见原因 → 改法(体检单各段的读法):

| 看到 | 多半是 | 建议方向 |
|---|---|---|
| 判定「卡在搬数据上」且显存 ≥ 85% | 带宽到头 | 减字节:融合、复用、低精度存储;逐元素类算子到这里就是终点 |
| 「卡在搬数据上」但显存低、L2 或 L1 高 | 访存模式差:重复读、未合并、bank 冲突 | 检查 L1/L2 命中率、每请求 sector 数;改分块让数据复用发生在寄存器或共享内存 |
| 「卡在算上」且 tensor core ≥ 80% | tensor core 喂饱了 | 这是好状态;再快只能换算法或降精度 |
| 「卡在算上」但 tensor core 低、普通浮点/整数逻辑高 | 没用上 tensor core,或大量标量指令(地址计算、类型转换、softmax 的 exp) | 看是否走了 `tl.dot` / `T.gemm`;减少循环里的整数运算 |
| 「两头都没跑满」+ 卡在寄存器上 | 分块太大、累加器太多、每线程寄存器超 128 | 减 `BLOCK_N` 或 warps 数;`sweep` 一把 |
| 「两头都没跑满」+ 卡在共享内存上 | stages × 分块 超了(5090 上限 101376 字节) | 减 stages 或分块 |
| 「等显存把数据取回来」占比高 | 全局访存没被流水掩盖 | 加 stages、提前 prefetch、增大占用率 |
| 「等计算单元排队」占比高 | tensor core 排队,正常的「计算密集」信号 | 无需处理 |
| 「等其他线程到达同步点」占比高 | `__syncthreads` 太频繁或 warp 间不均衡 | 减少同步点、warp specialization |
| 「访存指令排不进队」占比高 | 访存指令队列满 | 向量化加载、减少小请求 |
| 「切成多少块」那行报了尾波浪费 | 块数不是一波的整数倍,最后一波大半 SM 空转 | 调 grid 或分块让波数接近整数或足够大 |
| 本地内存溢出 > 0 | 寄存器溢出到 local | 减寄存器压力,这是硬伤 |

## 4. 评价「写得怎么样」

用户要的是判断,不是数字复述。按这个结构说,一段话到三段话:

1. **一句总评**:相对 torch 参考多少倍(体检单「相对 torch 参考」那行),相对本卡峰值几成,是否正确。这三个数决定基调:超过 torch 且峰值 80% 以上是「已经很好」;0.7× 到 1× 是「能用,有明确的改进点」;低于 0.5× 是「主要瓶颈还没解决」。
2. **瓶颈在哪、为什么**:判定 + 上表对应的原因,**落到用户代码的具体行或常量**。一到两条,挑最值钱的,不罗列。
3. **建议怎么改**:给方向和预期(NCU 的 `Est. Speedup` 可以引用),有 `[sweep]` 就顺手跑一轮把最优参数报出来。
4. 如果仓库里有同一数学的其他实现(或该后端上的 torch 参考已经很强),说明差距来自哪一层:算法、分块、访存模式,还是 DSL 本身的降级(如 Triton 在 sm_120 上退回 mma.sync)。

措辞:直说问题,不铺垫,不夸;数字进表格或单独一行;引用源码时给文件与行号。

## 5. 不做的事

- 不改 `kernels/<名>/` 里的文件,除非用户要求。
- 不跳过 check 直接 bench;不主动跑 `--full` NCU(慢,且体检单用不到)。
- 不为了让 check 通过放宽容差;容差不合理时说明原因再改。
- 不做 `--target auto`。
- 结果是对的但慢,先说慢在哪,不先动手优化;优化是用户的练习。
