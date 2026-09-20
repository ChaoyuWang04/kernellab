# kernellab

**学写 GPU 算子的个人实验台。** Mac 上写,远端 GPU 上编译、跑、测速、抓 NCU,结果回流成一张固定模板的体检单。执行后端可插拔,用 `--target` 显式选。

日常入口是 **`uv run klab web`** —— 一个对标 LeetCode 的本地面板:左边题面与优化路线,右边写算子,Run 看对不对、Submit 出体检单。CLI 是它的全集。

```text
Mac 浏览器/CLI ── klab ──┬── ssh:5090home   (home lab,RTX 5090 / sm_120 消费级 Blackwell)
                         ├── modal-a100     (Modal,A100-80GB / sm_80  Ampere)
                         ├── modal-h100     (Modal,H100      / sm_90  Hopper)
                         ├── modal-b200     (Modal,B200      / sm_100 数据中心 Blackwell)
                         ├── ssh:<租的机器> (vast / runpod / autodl,同一套 SshTarget)
                         └── local          (在 GPU 盒子上调试 harness 自己用)
```

**三层分离**:代码住在 Mac(本仓库,git 管理);环境定义在仓库里(`envs/<工具链>/`);后端只回答「在哪跑」(`targets.toml`)。

**用户与 agent 的目录分开**:`kernels/<名>/` 只放算子源码(面板的编辑器写的就是它);`specs/<名>/` 是 agent 写的接线,`problems/<题>/` 是 agent 写的题面与参考答案。

| 要什么 | 去哪 |
|---|---|
| 本文 | 命令、契约、面板、体检单的**用法** |
| agent 的操作流程 | [docs/01-AGENT-PLAYBOOK.md](docs/01-AGENT-PLAYBOOK.md) |
| 维护与扩展 | [docs/00-START.md](docs/00-START.md) |
| 还没做的事 | [docs/02-NEXT.md](docs/02-NEXT.md) |
| 某条规矩为什么这么定 | [docs/03-DECISIONS.md](docs/03-DECISIONS.md) |
| 踩过的坑 | [docs/04-PITFALLS.md](docs/04-PITFALLS.md) |

本地测试:`uv run pytest`。

## 日常用法

```bash
uv sync                                        # 首次,装本地 CLI
uv run klab web                                # 面板(日常入口,见下一节)
uv run klab targets                            # 列后端
uv run klab setup --target 5090home            # 在后端装工具链环境(幂等),--toolchain 选哪种
uv run klab probe  --target 5090home           # 设备属性 + 实测拷贝带宽 + fp16 matmul 吞吐
uv run klab run    matmul_triton_ampere        # 一条龙:check → bench → ncu → 体检单
uv run klab check  matmul_triton_ampere        # 只看正确性
uv run klab bench  matmul_tilelang --target modal-h100   # 先 check 再测速,报「相对 torch」的倍数
uv run klab ncu    matmul_cuda --case 8192-bf16 --open   # NCU,拉回报告并用本地 Nsight Compute 打开
uv run klab ptx    matmul_tk                   # dump PTX,按世代统计 mma.sync / wgmma / tcgen05 等
uv run klab report runs/<某次 ncu 运行>         # 重新渲染体检单
uv run klab compare matmul_triton_ampere matmul_tilelang   # 跨语言 / 跨后端对比表(读 runs/,不联网)
uv run klab sweep  matmul_triton_ampere --case 4096-bf16   # 按 meta.toml [sweep] 扫参
uv run klab baseline matmul_triton_ampere      # 把最新 bench 钉成基线;之后 bench 自动报差
uv run klab sh     --target 5090home           # 进后端 shell(已 cd 到远端 repo)
uv run klab exec   "nvidia-smi" --target modal-h100        # 在后端执行一段命令(诊断)
```

`<算子>` 可以写名字、`kernels/<名>`、`specs/<名>` 或其中任一文件。`--target` 不给就用 `targets.toml` 的默认后端。

每次运行在 `runs/<时间>-<算子>-<后端>-<模式>/` 落一份 `result.json`;NCU 另有 `ncu.ncu-rep`(GUI 打开)、`ncu-details.txt`(编辑器里直接读)、`ncu-raw.csv`。`runs/` 不进 git。

也可以用 VSCode 写 `kernels/<名>/kernel.py`:`Tasks: Run Task` 选 `klab: run 当前算子`,任务会弹出后端选择框。面板与 VSCode 改的是同一个文件。

## 本地面板

```bash
uv run klab web            # http://127.0.0.1:8777,自动开浏览器;--no-open 不开,--port 换端口
```

布局对标 LeetCode:**左边题面与讲解,右边写算子**。

| 区域 | 内容 |
|---|---|
| 左 · 题目 | `problems/<题>/problem.md`:数学定义、输入约束、评判标准、测试用例 |
| 左 · 优化路线 | `problems/<题>/editorial.md`:一级级的优化阶梯,每级标注对应体检单哪个指标 |
| 左 · 参考答案 | 从零到最优的完整晋升路径,每级一份代码 + 实测结论;可切语言,点一下载入编辑器 |
| 左 · 提交记录 | 历次提交的判定与数字;点开看**当次提交的源码快照** |
| 左 · 结果 | 体检单。Submit 完自动切到这里 |
| 右 · 编辑器 | Monaco(VSCode 同款内核),Python / CUDA 高亮 + 算子 API 片段补全 |
| 右 · 控制台 | case 选择 + 流式日志 + 判定 |

**Run 与 Submit 是两件事**,和 LeetCode 一样:

| 按钮 | 干什么 | 多久 | 快捷键 |
|---|---|---|---|
| ▶ Run | `klab check`,只看对不对 | 几秒 | ⌘↵ |
| ⬆ Submit | `klab run`:check → bench → ncu → 体检单 | 约一分钟 | ⌘⇧↵ |

编辑器里的代码停手约 1 秒自动存回 `kernels/<名>/kernel.py`(git 照常管它);每次运行另存一份源码快照到 `runs/<id>/submission/`,所以「这个 204 TFLOPS 对应哪份代码」永远查得到。**「↺ 重置」**恢复成 `problems/<题>/backbone/<工具链>.<后缀>`:契约(启动常量 + 函数签名)给全,函数体留空。

**一道题 × 多种语言**:`meta.toml` 的 `problem` 键把多个算子聚成一道题。`matmul` 下面挂着 triton / tilelang / cuda / cute / tk 五个实现,同样的 case 与容差,数字可以直接横着比 —— 编辑器头部下拉切语言,右上角下拉切后端。(tk 少一档非整数倍的 case:它的 tile 操作要求形状是 tile 尺寸的整数倍。)

编辑器是 Monaco(MIT,24 MB 预编译产物),首次运行自动取到 `klab/webui/vendor/`(gitignore);取不到会退回纯文本编辑框。除此之外**不引入任何依赖**:服务端是 stdlib 的 `http.server`,没有 web 框架、没有构建步骤、没有 node 运行时。只监听 `127.0.0.1`;算子名 / 后端名 / case 名一律先过白名单才进子进程命令行,子进程不走 shell。后端就一张卡,所以一次只许跑一个。

## 写一个算子

你只写 `kernels/<名>/` 里的算子本体。**分配输出、算 grid、传 stride 这类样板归接线管** —— Triton 的算子只需要一个 `@triton.jit` 函数加几个 tile 常量。接线由 agent 生成在 `specs/<名>/`:

```text
kernels/<名>/kernel.py      你的算子(任何 DSL)
specs/<名>/meta.toml        工具链、problem 键、ncu 过滤名、架构要求、容差、case 列表、[sweep]
specs/<名>/spec.py          make_inputs / run / reference / workload(可选 configure)
specs/<名>/baselines/       klab baseline 钉下的基线
```

`spec.py` 的契约与 DSL 无关:

| 函数 | 作用 |
|---|---|
| `make_inputs(case, device) -> dict` | 按 case 造输入,固定种子 |
| `run(**inputs) -> Tensor` | **启动你的算子**:`k = kernel_module(__file__)` 拿到 `kernels/<名>/kernel.py`,分配输出、按 `k.BLOCK_*` 算 grid、传 stride |
| `reference(**inputs) -> Tensor` | 同 dtype 的原生 torch 调用;check 用它比对,bench 用它当「相对 torch」标尺 |
| `workload(case, **inputs) -> {flops, bytes}` | 按**数学定义**换算 TFLOPS 与 GB/s |

题面、优化路线与各语言骨架在 `problems/<题>/`,由 `meta.toml` 的 `problem` 键关联:

```text
problems/<题>/problem.md                    题面:数学定义、输入约束、评判标准、测试用例
problems/<题>/editorial.md                  优化路线:一级级的阶梯,每级标注对应体检单哪个指标
problems/<题>/backbone/<工具链>.<后缀>       骨架:常量与函数签名给全,函数体留空;面板「重置」用它
problems/<题>/solutions/<工具链>/N-名字.<后缀> 参考答案第 N 级;docstring 第一行是标题,其余是说明
```

`cuda` 与 `tk` 的 `spec.py` 通过 `klab.harness.cppext.load_extension()` 编译 `kernels/<名>/` 下的 `.cu`,缓存在后端的 `~/.cache/klab/<名>-<架构>`(Modal 用 Volume 持久化)。

## 架构门禁

`meta.toml` 的 `requires.features` 与 harness 里的 `ARCH_FEATURES` 表匹配。四张卡的实测画像(同一份 Triton 代码,`klab ptx` 打出来):

| 卡 | 架构 | tensor core | 异步拷贝 |
|---|---|---|---|
| A100 | sm_80 | `mma.sync` | `cp.async` |
| 5090 | sm_120 | `mma.sync` | `cp.async` + **TMA** |
| H100 | sm_90 | **`wgmma`** | `cp.async` + TMA |
| B200 | sm_100 | **`tcgen05`** | `cp.async` + TMA |

**cc 数字不是超集关系**:5090 数字最大却没有 `wgmma`。声明了缺失特性的算子打上去会被拒并提示换 target。

```bash
uv run klab check matmul_tk                    # 拒:sm_120 缺 wgmma
uv run klab bench matmul_tk --ignore-requires  # 强行跑,拿跨代对照
uv run klab bench matmul_tk --target modal-h100
```

## 测速方法

预热 → 每次迭代前用 256 MB 的写把 L2 冲干净 → CUDA event 计时 → 取中位数与 p10/p90 → 按 `workload()` 换算带宽与算力 → 对 `targets.toml` 的峰值算百分比 → **同样方法给 `reference()` 计时,报「相对 torch」的倍数**(大于 1 是比 torch 快)。`--no-flush` 关掉 L2 冲刷,`--iters/--warmup` 调次数。

`peak_*` 是手填的参考值:`peak_gbps` 取公开规格,`peak_tflops_fp16` 填**见过的最好成绩**。

> **GB/s 与 %BW 是「有效带宽」,不是显存带宽。** 它按 `workload()` 声明的数学最小字节数除以时间算,工作集能被缓存吃下时会严重高估 —— `smallK` 那档 bench 报 1689 GB/s(94%),NCU 看显存只有 3%。真实显存流量看体检单的「显存」那一行。

## NCU 与体检单

默认抓 SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / LaunchStats / ComputeWorkloadAnalysis 五个 section,`--full` 换成 `--set full`(replay 次数多很多)。`meta.toml` 的 `kernel_regex` 用作 `-k regex:` 过滤,只抓你的 kernel。

体检单模板固定,任何算子、任何后端都一样,便于横着比:

| 段 | 回答什么 |
|---|---|
| 判定 | 一句话:卡在搬数据上 / 卡在算上 / 两头都没跑满在等 / 两边吃得差不多 |
| 跑多快 | 和 torch 比几倍、耗时与分位数、算力与带宽各用掉这张卡的几成、每搬 1 字节要算几次 |
| 哪个部件忙,哪个闲 | 计算单元、tensor core、普通浮点/整数逻辑/访存指令、内存通道(显存 / L2 / L1,含命中率) |
| 卡子怎么切的,SM 喂饱了吗 | 块数 × 线程数与**尾波浪费**、每线程寄存器、每块共享内存、warp 位置占用率与**卡在谁身上**、寄存器溢出 |
| warp 在等什么 | 每调度器手上有几个 warp、能立刻发几个,以及前四个等待原因(译成人话,括号里给 NCU 原名) |

措辞一律用人话,不留 NCU 英文术语,不摘录 OPT 原文。**读法:先看判定,再看它指向的那一段。**

5090home 已把驱动的 `RestrictProfilingToAdminUsers` 关掉,非 root 可用;租来的机器需要同样条件或 root,容器里还要 `--cap-add=SYS_ADMIN`。Modal 容器里 ncu 可用但锁不了时钟,`targets.toml` 给它设了 `ncu_clock_control = "none"`,数字比锁频的 5090 抖。

## PTX:确认降到了哪一代指令

体检单只说 tensor core 忙到几成,不说走的是哪条指令。练「某一代的特性」时用 `klab ptx`:在后端跑一次触发编译,从编译缓存取 PTX,按指令族计数,原文落 `runs/<id>/ptx/<kernel>.ptx`。

```bash
uv run klab ptx matmul_triton_ampere   # 判定:命中世代 Ampere(mma.sync×64, cp.async×32)
```

「命中世代」只看**标志指令**:`mma.sync` / `wmma.mma` / `cp.async`(Ampere)、`wgmma` / `cp.async.bulk`(Hopper)、`tcgen05`(数据中心 Blackwell)。`ldmatrix` / `stmatrix` / `mbarrier` / `setmaxnreg` 是辅助指令,只报条数不进判定 —— 它们「某代起就有」,5090 照样会发 `stmatrix`。

形状是运行期参数,PTX 只随 tile 常量变,所以默认只跑第一个 case。**已接 triton / tilelang / cuda / tk,cute 接不上**(见 [D10](docs/03-DECISIONS.md#d10-cute-的-ptx-先不接))。加法见 `klab/harness/ptxdump.py` 的 `_COLLECTORS`。

## Modal 后端

`targets.toml` 里 `kind = "modal"`,`gpu` 选卡型(H100 / H200 / B200 / A100-80GB / L40S)。凭据由 Modal SDK 读 `~/.modal.toml`,仓库里不放任何 token。

- 镜像 = `nvidia/cuda` devel 基础镜像 + `envs/<工具链>/` 的依赖,首次构建几分钟,之后命中 Modal 的镜像缓存
- 仓库以本地目录挂载进容器,改代码不重建镜像;`runs/` 打包传回 Mac,落盘位置与 SSH 后端完全一样
- 容器按调用起停、按秒计费;bench 循环在一次调用里做完
- 网络:先试直连 `api.modal.com:443`,不通就走 `api_proxy`(默认 `http://127.0.0.1:3213`)
- `klab sh` 对 Modal 无效,交互调试用 `klab exec` 或 `modal shell --gpu H100 <镜像>`

## 扫参与基线

`meta.toml` 加 `[sweep]`(参数名 = 候选列表),算子提供 `configure(**params)` 把一组参数应用上去(改全局常量、清编译缓存)。`klab sweep` 在后端按笛卡尔积逐组 check + bench,编译失败或共享内存超限的组记为失败而不中断,最后按 case 列出最快的几组。

`klab baseline` 把某次 bench 复制到 `specs/<名>/baselines/<后端>.json`(进 git)。之后同一后端的 `klab bench` 自动打印「现在 / 基线 / 变化」,±3% 以外标色。

## 加后端 / 加工具链

**加后端**:`targets.toml` 加一段。`kind = "ssh"` 的只需要 `host`(`~/.ssh/config` 里的别名)与 `root`。远端要有:能跑 CUDA 的驱动、`curl`、`rsync`;`klab setup` 会自己装 uv 与 venv。`klab/targets/` 里每个后端一个类,**只实现 `sync / run / fetch / shell` 四个方法**。

**加工具链**:`klab/toolchains/<name>.py` 给出 `setup_script()`,`envs/<name>/` 放依赖清单,`klab/toolchains/__init__.py` 登记。五种都共用 `_pip.py` 的装法:SSH 后端每种工具链一个 uv venv(`~/klab/envs/<name>`),Modal 每种工具链一个镜像(`klab-<name>` app)。可选属性 `APT` / `MODAL_RUN_COMMANDS` / `MODAL_ENV` 给 Modal 镜像加东西,`local_prepare(root)` 在 Mac 上做准备。

| 工具链 | 环境 | 编译 | 备注 |
|---|---|---|---|
| triton | pip | JIT | |
| tilelang | pip | JIT,调 nvcc | 生成的 kernel 名 `gemm_kernel` |
| cute | pip `nvidia-cutlass-dsl[cu13]` | JIT | kernel 名以 `kernel_cutlass_kernel_` 开头 |
| cuda | pip ninja + pybind11 | `cppext.load_extension()` 现场 nvcc | |
| tk | 同 cuda + ThunderKittens 源码 | 同上,加 include 与 `KITTENS_SM<xx>` 宏,sm_90 起用 `compute_XXa` | 5090home 连不上 github,TK 源码在 Mac 克隆到 `envs/tk/ThunderKittens`(gitignore)随 rsync 同步;Modal 镜像构建时自己克隆 |
