# kernellab 维护手册

> **交接语**:这是一个「Mac 上写 GPU 算子、远端机器上跑、性能指标回流 Mac」的个人实验台。接手先读完本页,再按需要看 README 的对应节。改动前跑 `uv run pytest`,改了跑 GPU 的部分再按第六节的矩阵上机验证。

## 一、为什么有这个项目

用户在 Mac 的 VSCode 上学习和练习写算子(Triton、TileLang、CuTe DSL、裸 CUDA、ThunderKittens),但 Mac 没有 NVIDIA GPU。需求有三条,全部已实现:

1. **前端只在 Mac**:写、编译、运行、看结果都不离开 VSCode,不建网页。
2. **后端任意插拔**:同一份算子,显式指定打到 home lab 的 RTX 5090,或 Modal 云上的 H100 / B200,将来还有租用机和昇腾。
3. **可观测**:每次运行拿到正确性、测速、NCU 计数器,并压成一张固定模板的报告,快速看到算子最重要的信息。

它与 `~/1Project/interviewprep`(面试准备系统)并排、**互不引用代码**。那边的 `projects/Kernel与GPU编程/` 下有 triton、tilelang、cutlass 等源码镜像,只作阅读材料。算子实验得到的理解写回那边的知识库文章,不在这里沉淀文档。

## 二、设计原则(改代码前先对照)

1. **三层分离**:代码住在 Mac(本仓库,git 管理);环境定义在仓库里(`envs/<工具链>/`);后端只回答「在哪跑」(`targets.toml`)。任何新功能先问它属于哪一层。
2. **Target 接口只有四个方法**:`sync / run / fetch / shell`。编译、跑、测速、NCU、扫参全是 `run()` 之上的命令串,**不进后端类**。这是插拔成立的前提,加后端时不得往接口里加方法。
3. **算子契约与 DSL 无关**:`kernel.py` 只暴露 `make_inputs / run / reference / workload`(可选 `configure`),harness 不认识任何 DSL。DSL 差异全部落在 `toolchains/<name>.py`(怎么装)和算子自己的 `kernel.py`(怎么调)。
4. **文件即数据**:`runs/<id>/result.json` 是唯一的结果真源,报告、对比表、基线都从它推导;不引入数据库、不做网页。
5. **显式优先**:后端用 `--target` 手选;门禁只负责拒绝不匹配的组合并提示。`--target auto` 用户 2026-09-18 明确决定先不做。
6. **不放凭据**:Modal 读 `~/.modal.toml`,SSH 读 `~/.ssh/config`,仓库里没有也不允许有 token。
7. **不为未验证的平台写代码**:租用机、昇腾都还没上过机,只留设计说明(第八节),不写猜测性的适配器。

## 三、架构与数据流

```text
Mac VSCode ── klab CLI(uv venv,只装 typer/rich/modal)
      │  ① sync   rsync 仓库(排除 .git/.venv/runs)→ 远端 <root>/repo   |  Modal: add_local_dir 挂载
      │  ② run    ssh host bash -c "<命令串>"                             |  Modal: 远程函数里 bash -c
      │           命令串 = export PATH/PYTHONPATH/KLAB_* ; cd repo ; <python> -m klab.harness.runner ...
      │  ③ fetch  rsync 远端 <root>/runs/<id> → 本地 runs/<id>            |  Modal: 函数返回 runs 的 tar
      ▼
runs/<时间>-<算子>-<后端>-<模式>/result.json (+ ncu.ncu-rep, ncu-details.txt, ncu-raw.csv, report.md)
      │
      ├─ klab report   → 体检单(kreport.py)
      ├─ klab compare  → 跨算子 / 跨后端表(compare.py)
      └─ klab baseline → kernels/<名>/baselines/<后端>.json,bench 时报差
```

远端只有三样东西:`<root>/repo`(仓库副本)、`<root>/envs/<工具链>`(uv venv,SSH 后端)或镜像(Modal)、`<root>/runs`(结果)。远端不保存任何不能从 Mac 重建的状态,机器丢了重跑 `klab setup` 即可。

## 四、目录与职责

| 路径 | 职责 | 改动时注意 |
|---|---|---|
| `klab/cli.py` | 所有命令的入口;`_remote_run()` 是 check/bench/sweep 的公共路径,ncu 有自己的 `run_with_ncu()` | 新命令先看能否复用 `_remote_run` |
| `klab/config.py` | 读 `targets.toml`;`TargetConfig.extra` 原样透传后端私有选项 | 新字段优先走 `extra`,不改数据类 |
| `klab/targets/base.py` | Target 抽象;`env_prefix()` 决定远端 PATH/PYTHONPATH/KLAB_ROOT/KLAB_TK_ROOT | 远端找不到命令十有八九是这里 |
| `klab/targets/ssh.py` | rsync + ssh,ControlMaster 复用连接 | Mac 的 rsync 是 openrsync,只用 `-az --delete --exclude` |
| `klab/targets/modal.py` | 镜像构建、挂载、Volume 缓存、直连/代理选路、输出回流 | 见第七节的四个坑 |
| `klab/targets/local.py` | 在 GPU 盒子上直接跑,调试 harness 用 | |
| `klab/toolchains/` | 每种工具链一个模块:`setup_script()` + 可选 `APT / MODAL_RUN_COMMANDS / MODAL_ENV / local_prepare()` | 纯 pip 的直接复用 `_pip.py` |
| `klab/harness/runner.py` | **在后端跑**;check / bench / ncu / sweep 四种模式;`ARCH_FEATURES` 门禁表 | 只能依赖 torch 与标准库 |
| `klab/harness/spec.py` | `meta.toml` 与 `kernel.py` 的契约 | 契约变了要同步 README「写一个算子」与 tests |
| `klab/harness/probe.py` | 设备属性 + 实测带宽 / matmul 吞吐 | 实测值手工填回 `targets.toml` 的 `peak_*` |
| `klab/harness/cppext.py` | cuda / tk 工具链的 nvcc 现场编译 | 架构后缀、TK 宏、缓存目录都在这 |
| `klab/kreport.py` | 体检单:raw CSV + details 文本 + 最近一次 bench → markdown | 指标名依赖 NCU 版本,tests 里守着 |
| `klab/report.py`、`klab/compare.py` | 终端表格;对比表 | |
| `kernels/<名>/` | 算子:`meta.toml` + `kernel.py`(+ `.cu`、`_vendor/`、`baselines/`) | `_vendor/` 里是第三方原文,不改 |
| `envs/<工具链>/` | `requirements.txt` + `torch-index.txt`;`envs/tk/ThunderKittens/` 是 Mac 上的克隆,gitignore | |
| `targets.toml` | 后端登记与峰值 | 加后端要同步 `.vscode/tasks.json` 的下拉(tests 守) |
| `tests/` | 不需要 GPU 的本地测试,夹具是一次真实的 5090 NCU 运行 | |
| `runs/` | 结果,gitignore | 可随时清空 |

## 五、已实现的功能清单

| 命令 | 做什么 | 状态 |
|---|---|---|
| `klab targets` | 列后端 | ✅ |
| `klab setup --target T --toolchain X` | 后端装环境(SSH:uv venv;Modal:触发镜像构建);幂等 | ✅ 五种工具链 × 两后端 |
| `klab probe --target T` | 设备属性、实测拷贝带宽与 fp16 matmul 吞吐 | ✅ |
| `klab check <算子> --target T` | 与 `reference()` 逐 case 比对;先过 `requires` 门禁 | ✅ |
| `klab bench <算子> --target T` | 先 check;预热、每次刷 L2、event 计时、中位数与分位数、对峰值百分比;有基线则报差 | ✅ |
| `klab ncu <算子> --target T` | 后端跑 ncu(默认七个 section,`--full` 全量),拉回 `.ncu-rep`、文本、CSV,自动渲染体检单 | ✅ |
| `klab report <run_dir>` | 重渲染体检单 | ✅ |
| `klab compare [算子...] [-t 后端]` | 每个 (算子, 后端) 取最新 bench,出表 | ✅ |
| `klab sweep <算子> --target T` | 按 `[sweep]` 笛卡尔积逐组 check + bench,失败组不中断 | ✅ |
| `klab baseline <算子> --target T` | 钉基线到 `kernels/<名>/baselines/<后端>.json` | ✅ |
| `klab exec "<shell>" --target T` | 在后端 repo 目录执行命令(诊断) | ✅ |
| `klab sh --target T` | 交互 shell(仅 SSH) | ✅ |
| `klab open <run_dir>` | 用本地 Nsight Compute 打开 `.ncu-rep` | ✅ |
| `--target auto` | 按 `requires` 自动挑后端 | ⏸ 用户搁置 |

后端:`5090home`(SSH,RTX 5090 sm_120)、`modal-h100`(Modal,H100 sm_90)。工具链:triton、tilelang、cute、cuda、tk。每种工具链至少一个样板算子,全部在两个后端跑过 check / bench / ncu(2026-09-18)。

观测层的决定:**NCU 是核心,不接 Nsight Systems**。单算子的问题全在计数器里,时间线是整条模型链路的事。

## 六、验证矩阵(改了什么就跑什么)

| 改动 | 必跑 | 上机验证 |
|---|---|---|
| 任何改动 | `uv run pytest` | |
| `klab/cli.py`、`klab/config.py` | pytest | `klab check kernels/vector_add --target 5090home` |
| `klab/targets/ssh.py`、`base.py` | pytest | 同上,再 `klab ncu kernels/vector_add --target 5090home --case 1M` |
| `klab/targets/modal.py` | pytest | `klab check kernels/matmul --target modal-h100`(注意首次会重建镜像) |
| `klab/harness/runner.py`、`spec.py` | pytest | 一个 triton 算子的 check + bench + ncu,一个 cuda 算子的 check |
| `klab/harness/cppext.py` | pytest | `klab check kernels/sgemm_cuda` 与 `kernels/tile_add_tk`,两个后端各一次 |
| `klab/kreport.py` | pytest(夹具) | `klab report` 一个已有的 ncu 运行目录 |
| `klab/toolchains/*`、`envs/*` | pytest | 对应后端 `klab setup --toolchain X`,再跑该工具链的样板算子 |
| 新算子 | pytest(契约检查) | 目标后端 check → bench → ncu 三步,ncu 后确认 `kernel_regex` 只抓到自己的 kernel |
| 新后端 | pytest(tasks.json 下拉) | setup → probe → vector_add check/bench/ncu |
| `targets.toml` 峰值 | pytest | 跑一次 `klab probe`,用实测值 |
| README / 本手册 | 无 | 按第四节的路径核对一遍 |

pytest 覆盖的是不需要 GPU 的部分:配置、算子契约、工具链登记、体检单解析与渲染、对比表。它证明不了算子在卡上是对的。

## 七、踩过的坑(改相关代码前先看)

**SSH 后端**

- 非交互 ssh 的 PATH 不含 `~/.local/bin` 与 `/usr/local/cuda/bin`,`env_prefix()` 自己补;uv 装在远端 `~/.local/bin`。
- 环境变量里带 `~` 不会被 Python 展开,`cppext` 对 `KLAB_TK_ROOT` 做了 `expanduser`。
- 5090home 直连不了 github:凡是要 `git clone` 的东西,在 Mac 上克隆进仓库(gitignore)随 rsync 同步。ThunderKittens 就是这么处理的。
- ncu 的 `-k regex:` 参数必须 `shlex.quote`,正则里的 `|` 会被远端 shell 当管道。

**Modal 后端**

- 本机 `HTTPS_PROXY`(127.0.0.1:3213)过不了 Modal 的 gRPC:`_choose_route()` 先试直连 `api.modal.com:443`,不通再走 `targets.toml` 的 `api_proxy`。决策必须在 `import modal` 之前,SDK 导入时读环境变量。
- 程序化 `app.run()` 必须包在 `modal.enable_output()` 里,否则容器 stdout 不回流,看起来像「跑了但没输出」。
- 容器里 ncu 可用、计数器可读,但锁不了 GPU 时钟:该后端 `ncu_clock_control = "none"`,数字比锁频的 5090 抖。
- 镜像按工具链分(`klab-<工具链>` app),CLI 通过 `KLAB_TOOLCHAIN` 环境变量告诉 `ModalTarget` 用哪个;忘了设会拿 triton 镜像去跑别的工具链。
- 编译缓存挂 Volume `klab-cache` 到 `/root/.cache`,否则 cuda / tk 每次冷启动重编一分钟。

**工具链**

- 5090 是 sm_120:没有 wgmma / tcgen05 / cluster,共享内存上限 101376 字节。`ARCH_FEATURES` 表按 major 版本给特性,cc 数字不是超集关系。
- TileLang 0.1.14 生成的 kernel 名是 `gemm_kernel`;CuTe DSL 的名字以 `kernel_cutlass_kernel_` 开头,`kernel_regex` 写 `cutlass`,写宽了会把 torch 造输入的 kernel 抓进报告、体检单取错行。
- ThunderKittens:sm_90 起要 `compute_XXa` 架构目标;宏 `KITTENS_SM<xx>` 只能定义一个;`gl` 的编译期维度要传 `nullptr`,用 `make_gl<GL>(ptr, b, d, r, c)` 省事;`warpid()` 在 `kittens::` 命名空间。
- torch 扩展里用 `getCurrentCUDAStream` 要 `#include <ATen/cuda/CUDAContext.h>`。
- 5090 上 ncu 报带宽用 `Tbyte/s`,H100 用 `Gbyte/s`,`kreport._gbps()` 统一。
- ncu 对超出默认 carveout 的动态共享内存,`launch__occupancy_limit_shared_mem` 报 0,不能当限制因子。

## 八、未接入的平台怎么接

**租用机(vast / runpod / autodl)**:`~/.ssh/config` 里已有别名。`targets.toml` 加一段 `kind = "ssh"`,填 `host`、`root`、`cuda_bin`(看机器上 CUDA 装在哪),再 `klab setup --toolchain triton` → `klab probe` → `vector_add` 三步。可能遇到:没有 `rsync`(apt 装)、容器里 ncu 没权限(需要 `--cap-add=SYS_ADMIN` 或 `ncu_clock_control = "none"`)、非 root 用户 `~/.local/bin` 不在 PATH(已处理)。不需要改代码;若某台机器连不上 github,照 TK 的做法在 Mac 克隆。

**Modal 换卡型**:复制 `modal-h100` 段,改 `gpu = "B200"` 与峰值;镜像自动复用。B200 是 sm_100,`ARCH_FEATURES` 已有该行。

**昇腾(910B 等)**:需要租机后实测,预计改三处,Target 层不动:

1. `toolchains/ascend.py`:CANN 环境 + triton-ascend(或 Ascend C 的编译方式),`envs/ascend/` 放依赖。
2. `harness/runner.py`:设备探测从 `torch.cuda` 改成按后端选 `torch_npu`;`ARCH_FEATURES` 加 Ascend 一行(cube / vector 单元);`device_info()` 字段照旧。
3. profiler 适配:`klab ncu` 对昇腾后端改调 `msprof`,`kreport.py` 加一个从 msprof 输出到体检单字段的映射;体检单模板不变,采不到的字段显示 `-`。

## 九、维护规矩

- **README 面向使用,本手册面向维护**:README 讲命令怎么用、算子怎么写;本手册讲为什么这么设计、改哪里、怎么验证、踩过什么坑。同一件事只在一处完整解释,另一处链接。
- 加后端、加工具链、改契约,三件事必须同步:代码、README 对应节、`tests/`。
- `targets.toml` 的 `peak_*` 是手填的:`peak_gbps` 取公开规格,`peak_tflops_fp16` 取 `klab probe` 实测,注释里写日期。
- 决策记录只写「定了什么、谁定的、哪天」,不写论证:
  - 2026-09-18 用户:单独仓库,与 interviewprep 并排;先跑通 Triton。
  - 2026-09-18 用户:Modal 直连优先,不通再走 127.0.0.1:3213。
  - 2026-09-18 用户:NCU 是核心,不做 nsys;要固定模板的体检单。
  - 2026-09-18 用户:`--target auto` 先不做。
  - 2026-09-18 用户:租用机、昇腾等租到后再单独接入。
- 提交信息用中文,写清改了哪一层;`runs/`、`.venv/`、`envs/tk/ThunderKittens/` 不进 git。
