# kernellab

Mac 上写算子,远端 GPU 上编译、跑、测速、NCU。执行后端可插拔,用 `--target` 显式选。

日常入口是 **`uv run klab web`** —— 一个对标 LeetCode 的本地面板:左边题面与优化路线,右边写算子,Run 看对不对、Submit 出 NCU 体检单。CLI 仍然是全功能的,面板只是它的皮。

本文面向使用。agent 的操作流程在 **[docs/01-AGENT-PLAYBOOK.md](docs/01-AGENT-PLAYBOOK.md)**,维护与扩展看 **[docs/00-START.md](docs/00-START.md)**。本地测试:`uv run pytest`。

```text
Mac 浏览器/CLI ── klab ──┬── ssh:5090home   (home lab,RTX 5090)
                         ├── ssh:<租的机器> (vast / runpod / autodl,同一套 SshTarget)
                         ├── modal-h100     (Modal 云 GPU,gpu 字段选卡型)
                         └── local          (在 GPU 盒子上调试 harness 自己用)
```

三层分离:**代码住在 Mac**(本仓库,git 管理);**环境定义在仓库里**(`envs/<工具链>/`);**后端只回答「在哪跑」**(`targets.toml`)。

用户与 agent 的目录分开:**`kernels/<名>/` 只放算子源码**(面板的编辑器写的就是它);`specs/<名>/` 是 agent 写的接线(工具链、case、参考实现),`problems/<题>/` 是 agent 写的题面与优化路线。

## 日常用法

```bash
uv sync                                        # 首次,装本地 CLI
uv run klab targets                            # 列后端
uv run klab setup --target 5090home            # 在后端装 Triton 环境(幂等)
uv run klab probe --target 5090home            # 设备属性 + 实测拷贝带宽 + fp16 matmul 吞吐
uv run klab run   softmax                      # 一条龙:check → bench → ncu → 体检单;--target 不给用 targets.toml 的默认后端
uv run klab check softmax                      # 正确性(<算子> 可写名字、kernels/<名>、specs/<名> 或其中的文件)
uv run klab bench softmax --target modal-h100  # 先 check 再测速,报「相对 torch 参考」的倍数
uv run klab ncu   softmax --case 8192x8192-f16 --open   # NCU,拉回报告并用本地 Nsight Compute 打开
uv run klab ptx   softmax                      # dump PTX,按世代统计 mma.sync / wgmma / cp.async / tcgen05 等指令
uv run klab web                                # 面板(日常入口,见下一节)
uv run klab sh    --target 5090home            # 进后端 shell(已 cd 到远端 repo)
uv run klab exec  "nvidia-smi" --target modal-h100   # 在后端执行一段命令(诊断)
uv run klab report runs/<某次 ncu 运行>          # 重新渲染体检单
uv run klab compare matmul matmul_tl            # 跨 DSL / 跨后端的 bench 对比表(读 runs/,不联网)
uv run klab sweep matmul --case 4096-f16        # 按 meta.toml [sweep] 扫参,列每个 case 最快的几组
uv run klab baseline matmul                     # 把最新 bench 钉成基线;之后 bench 自动报与基线的差
```

也可以继续用 VSCode 写 `kernels/<名>/kernel.py`:`Tasks: Run Task` 选 `klab: run 当前算子`(或 check / bench / ncu),任务会弹出后端选择框。面板与 VSCode 改的是同一个文件,两边可以混着用。

每次运行在 `runs/<时间>-<算子>-<后端>-<模式>/` 落一份 `result.json`;NCU 另有 `ncu.ncu-rep`(GUI 打开)、`ncu-details.txt`(直接在编辑器里读)、`ncu-raw.csv`。`runs/` 不进 git。

## 本地面板(算子版 LeetCode)

```bash
uv run klab web            # http://127.0.0.1:8777,自动开浏览器;--no-open 不开,--port 换端口
```

布局对标 LeetCode:**左边题面与讲解,右边写算子**。

| 区域 | 内容 |
|---|---|
| 左 · 题目 | `problems/<题>/problem.md`:数学定义、输入约束、评判标准、测试用例 |
| 左 · 优化路线 | `problems/<题>/editorial.md`:一级一级的优化阶梯,每级标注对应体检单的哪个指标 |
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

编辑器里的代码停手约 1 秒自动存回 `kernels/<名>/kernel.py`(git 照常管它);每次运行另存一份源码快照到 `runs/<id>/submission/`,所以「这个 204 TFLOPS 对应哪份代码」永远查得到。「↺ 重置」恢复成 `problems/<题>/backbone/<工具链>.py`:**契约(启动常量 + 函数签名)给全,函数体留空**,和 LeetCode 给你函数签名是一个意思。

**一道题 × 多种语言**:`meta.toml` 的 `problem` 键把多个算子聚成一道题。`matmul` 这道题下面目前挂着 triton 与 tilelang 两个实现(cuda / cute 待加),同样的 case 与容差,数字可以直接横着比 —— 编辑器头部下拉切语言,右上角下拉切后端。

编辑器是 Monaco(MIT,24 MB 预编译产物),首次运行 `klab web` 时自动从 npm registry 取到 `klab/webui/vendor/`(gitignore,照 ThunderKittens 的先例)。取不到也能用,会退回纯文本编辑框。除此之外**不引入任何依赖**:服务端是 stdlib 的 `http.server`,没有 web 框架、没有构建步骤、没有 node 运行时。

只监听 `127.0.0.1`;算子名 / 后端名 / case 名一律先过白名单才进子进程命令行,子进程不走 shell。后端就一张卡,所以一次只许跑一个,跑着的时候再提交会被拒。

## 写一个算子

你只写 `kernels/<名>/` 里的算子本体。**分配输出、算 grid、传 stride 这类样板归接线管** —— Triton 的算子只需要一个 `@triton.jit` 函数加几个 tile 常量,`spec.py` 的 `run()` 读常量算 grid 并启动。接线由 agent 生成在 `specs/<名>/`:

```text
kernels/<名>/kernel.py      你的算子(任何 DSL)
specs/<名>/meta.toml        工具链、ncu 过滤名、架构要求、容差、case 列表、[sweep]
specs/<名>/spec.py          make_inputs / run / reference / workload(可选 configure)
specs/<名>/baselines/       klab baseline 钉下的基线
```

`spec.py` 的契约与 DSL 无关:

| 函数 | 作用 |
|---|---|
| `make_inputs(case, device) -> dict` | 按 case 造输入,固定种子 |
| `run(**inputs) -> Tensor` | **启动你的算子**:`k = kernel_module(__file__)` 拿到 `kernels/<名>/kernel.py`,分配输出、按 `k.BLOCK_*` 算 grid、传 stride |
| `reference(**inputs) -> Tensor` | 同 dtype 的原生 torch 调用;check 用它比对,bench 用它当「相对 torch」标尺 |
| `workload(case, **inputs) -> {flops, bytes}` | 按数学定义换算 TFLOPS 与 GB/s |

`meta.toml` 的 `requires.features` 是架构门禁:5090 是 sm_120,没有 wgmma / tcgen05 / cluster,声明了这些特性的算子打 5090 会被拒并提示换 target。

题面、优化路线与各语言骨架在 `problems/<题>/`,由 `meta.toml` 的 `problem` 键关联:

```text
problems/<题>/problem.md                   题面:数学定义、输入约束、评判标准
problems/<题>/editorial.md                 优化路线:一级级的优化阶梯,每级标注对应体检单哪个指标
problems/<题>/backbone/<工具链>.py          骨架:常量与函数签名给全,函数体留空;面板「重置」用它
problems/<题>/solutions/<工具链>/N-名字.py   参考答案第 N 级;docstring 第一行是标题,其余是说明
```

五种工具链的样板算子在 git 历史里(`git show 655f0e3 --stat`),需要时捞出来当参考。

`cuda` 与 `tk` 的 `spec.py` 通过 `klab.harness.cppext.load_extension()` 编译 `kernels/<名>/` 下的 `.cu`,缓存在后端的 `~/.cache/klab/<名>-<架构>`(Modal 用 Volume 持久化)。

## 测速方法

预热 → 每次迭代前用 256 MB 的写把 L2 冲干净 → CUDA event 计时 → 取中位数与 p10/p90 → 按 `workload()` 换算带宽与算力 → 对 `targets.toml` 里的峰值算百分比 → **同样方法给 `reference()` 计时,报「相对 torch」的倍数**(大于 1 是比 torch 快)。`--no-flush` 关掉 L2 冲刷(看热缓存表现),`--iters/--warmup` 调次数。

`peak_*` 是手填的参考值:`peak_gbps` 取公开规格,`peak_tflops_fp16` 取 `klab probe` 实测的 matmul 吞吐。

## NCU

默认抓 SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / LaunchStats / ComputeWorkloadAnalysis 五个 section,`--full` 换成 `--set full`(replay 次数多很多)。`meta.toml` 的 `kernel_regex` 用作 `-k regex:` 过滤,只抓你的 kernel,不抓 torch 造输入的那些。

5090home 已把驱动的 `RestrictProfilingToAdminUsers` 关掉,非 root 可用;租来的机器需要同样条件或 root,容器里还要 `--cap-add=SYS_ADMIN`。

Modal 的 H100 容器里 ncu 可用、计数器可读(2026-09-18 实测),但锁不了 GPU 时钟,所以 `targets.toml` 里给它设了 `ncu_clock_control = "none"`;数字会比锁频的 5090 抖一些。

## PTX:确认降到了哪一代指令

体检单只说 tensor core 忙到几成,不说走的是哪条指令。练「某一代的特性」时用 `klab ptx`:在后端跑一次触发编译,从编译缓存取 PTX,按指令族计数,PTX 原文落 `runs/<id>/ptx/<kernel>.ptx`。

```bash
uv run klab ptx matmul_triton_ampere            # 判定:命中世代 Ampere(mma.sync×64, cp.async×32)
```

「命中世代」只看**标志指令**:`mma.sync` / `cp.async`(Ampere)、`wgmma` / `cp.async.bulk`(Hopper)、`tcgen05`(数据中心 Blackwell)。`ldmatrix` / `stmatrix` / `mbarrier` / `setmaxnreg` 是辅助指令,只报条数不进判定 —— 它们「某代起就有」,5090(sm_120)照样会发 `stmatrix`,出现了并不说明你用上了那一代的核心能力。

形状是运行期参数,PTX 只随 tile 常量变,所以默认只跑第一个 case。目前只接了 triton;其他工具链的取法(tilelang 生成的 `.cu`、CuTe 的 JIT cubin、cppext 编出的 `.so`)等各自有算子时按实测补,加法见 `klab/harness/ptxdump.py` 的 `_COLLECTORS`。

## 算子体检单(固定模板)

`klab ncu` 跑完自动打印并写入 `runs/<id>/report.md`;`klab report <run_dir>` 可重新渲染。模板固定,任何算子、任何后端都一样,便于横着比。数据来自 NCU 的 raw CSV 与 details 文本,加上同一算子、同一后端最近一次 bench 的数字。

| 段 | 回答什么 |
|---|---|
| 判定 | 一句话:卡在搬数据上 / 卡在算上 / 两头都没跑满在等 / 两边吃得差不多,外加 warp 位置用了多少 |
| 跑多快 | 和 torch 比几倍、耗时与分位数、算力与带宽各用掉这张卡的几成、每搬 1 字节要算几次(对 roofline 拐点) |
| 哪个部件忙,哪个闲 | 计算单元、tensor core、普通浮点/整数逻辑/访存指令、内存通道(显存 / L2 / L1,含命中率) |
| 卡子怎么切的,SM 喂饱了吗 | 块数 × 线程数与**尾波浪费**、每线程寄存器、每块共享内存、warp 位置占用率与**卡在谁身上**、寄存器溢出 |
| warp 在等什么 | 每调度器手上有几个 warp、能立刻发几个,以及前四个等待原因(译成人话,括号里给 NCU 原名) |

措辞一律用人话,不留 NCU 英文术语。**不摘录 NCU 的 OPT 原文** —— 里面最值钱的尾波浪费,体检单自己算好了直接写在「切成多少块」那一行。

读法:先看判定,再看它指向的那一段。卡在搬数据上就看显存是否已到 80%(是就到头了,只能减字节);卡在算上就看 tensor core 与「卡在谁身上」;两头都没跑满就看「warp 在等什么」。

## Modal 后端

`targets.toml` 里 `kind = "modal"`,`gpu` 选卡型(H100 / H200 / B200 / A100-80GB / L40S)。凭据由 Modal SDK 读 `~/.modal.toml`,仓库里不放任何 token。

- 镜像 = `nvidia/cuda` devel 基础镜像 + `envs/<工具链>/` 的依赖,首次构建几分钟,之后命中 Modal 的镜像缓存
- 仓库以本地目录挂载进容器,改代码不重建镜像;`runs/` 打包传回 Mac,落盘位置与 SSH 后端完全一样
- 容器按调用起停,按秒计费;bench 循环在一次调用里做完,不要一个 case 一次调用
- 网络:先试直连 `api.modal.com:443`,不通就走 `api_proxy`(默认 `http://127.0.0.1:3213`)。本机 HTTPS_PROXY 指向的代理过不了 Modal 的 gRPC,所以直连优先
- `klab sh` 对 Modal 无效,交互调试用 `klab exec` 或 `modal shell --gpu H100 <镜像>`

## 架构门禁与后端插拔

`meta.toml` 的 `requires.features` 与 harness 里的 `ARCH_FEATURES` 表匹配。cc 数字不是超集关系:5090 是 sm_120,数字最大,却没有 Hopper 的 `wgmma`、也没有 B200 的 `tcgen05`。所以声明了 `wgmma` 的算子打 5090 会被拒并提示换 target;`--ignore-requires` 可以强行跑(Triton 会退回 mma.sync),用来做跨代对照。

```bash
uv run klab check matmul                              # 拒:sm_120 缺 wgmma
uv run klab bench matmul --ignore-requires            # 强行跑,拿对照
uv run klab bench matmul --target modal-h100          # 换后端,原样跑
```

## 扫参与基线

`meta.toml` 加 `[sweep]`(参数名 = 候选列表),`kernel.py` 提供 `configure(**params)` 把一组参数应用到算子(改全局常量、清编译缓存)。`klab sweep` 在后端按笛卡尔积逐组 check + bench,编译失败或共享内存超限的组记为失败而不中断,最后按 case 列出最快的几组。5090 的共享内存上限是 101376 字节,`BLOCK_N=256` 配 4 级流水会超。

`klab baseline` 把某次 bench 复制到 `specs/<名>/baselines/<后端>.json`(进 git)。之后同一后端的 `klab bench` 自动打印「现在 / 基线 / 变化」,±3% 以外标色。

## 加一个后端

`targets.toml` 加一段。`kind = "ssh"` 的只需要 `host`(`~/.ssh/config` 里的别名)与 `root`,其余走默认。远端要有:能跑 CUDA 的驱动、`curl`、`rsync`;`klab setup` 会自己装 uv 与 venv。

`klab/targets/` 里每个后端一个类,只实现 `sync / run / fetch / shell` 四个方法;编译、跑、测速、NCU 都是 `run()` 之上的命令串,不进后端类。

## 加一种工具链

`klab/toolchains/<name>.py` 给出 `setup_script()`,`envs/<name>/` 放依赖清单,`klab/toolchains/__init__.py` 登记。五种都共用 `_pip.py` 的装法:SSH 后端每种工具链一个 uv venv(`~/klab/envs/<name>`),Modal 每种工具链一个镜像(`klab-<name>` app)。模块可选属性 `APT` / `MODAL_RUN_COMMANDS` / `MODAL_ENV` 给 Modal 镜像加东西,`local_prepare(root)` 在 Mac 上做准备。

| 工具链 | 环境 | 编译 | 备注 |
|---|---|---|---|
| triton | pip | JIT | |
| tilelang | pip | JIT,调 nvcc | 生成 kernel 名 `gemm_kernel` |
| cute | pip `nvidia-cutlass-dsl[cu13]` | JIT | sm_120 与 sm_90 各有官方实现 |
| cuda | pip ninja + pybind11 | `cppext.load_extension()` 现场 nvcc | |
| tk | 同 cuda + ThunderKittens 源码 | 同上,加 include 与 `KITTENS_SM<xx>` 宏,sm_90 起用 `compute_XXa` | 5090home 连不上 github,TK 源码在 Mac 克隆到 `envs/tk/ThunderKittens`(gitignore)随 rsync 同步;Modal 镜像构建时自己克隆 |

## 路线

1. ✅ SSH 后端 + Triton + check / bench / ncu,打通 5090home
2. ✅ Modal 后端(同一份 requirements)+ 架构特性门禁,matmul 在 5090 被拒、在 H100 原样跑
3. ✅ TileLang、CuTe DSL、裸 CUDA、ThunderKittens 工具链 + `klab compare`
4. ✅ `klab sweep` 扫参 + `klab baseline` 基线回归
5. `--target auto`:按 `requires` 匹配后端(用户 2026-09-18 决定先不做,等算子多到手选变烦再加)
6. 租用机验证:vast / runpod / autodl 三个别名当时都离线,没跑过;租到机器后 `klab setup` 走一遍即可
7. 昇腾:需要租一台 910B。要改的只有三处:`toolchains/ascend.py`(CANN + triton-ascend 或 Ascend C 的装法)、harness 的设备探测(torch.cuda → torch_npu)与 `ARCH_FEATURES` 加 Ascend 一行、profiler 适配器(ncu → msprof,体检单的字段映射)。Target 层不用动
