# kernellab 维护手册

Mac 上写 GPU 算子,远端 GPU 上编译、跑、测速、抓 NCU,结果回流成一张固定模板的体检单。日常入口是 `klab web`(算子版 LeetCode),CLI 是它的全集。

本页讲**系统怎么运转、改哪里、怎么验证**。agent 接到「测一下这个 kernel」时的操作流程在 [01-AGENT-PLAYBOOK.md](01-AGENT-PLAYBOOK.md);命令与契约的用法在 [README.md](../README.md)。

## 一、核心守则

1. **三层分离**:代码住在 Mac(本仓库,git 管理);环境定义在仓库里(`envs/<工具链>/`);后端只回答「在哪跑」(`targets.toml`)。新功能先问它属于哪一层。
2. **Target 接口只有四个方法**:`sync / run / fetch / shell`。编译、跑、测速、NCU、扫参全是 `run()` 之上的命令串,**不进后端类**。这是后端可插拔的前提,加后端时不得往接口里加方法。
3. **四种目录,各有其主**:`kernels/<名>/` 只放算子源码(面板的编辑器写它,agent 不擅自改);`specs/<名>/` 是接线;`problems/<题>/` 是题面与讲解;`runs/` 是结果。后三者是 agent 的。
4. **spec 契约与 DSL 无关**:只有 `make_inputs / run / reference / workload`(可选 `configure`)。harness 不认识任何 DSL;DSL 差异落在 `toolchains/<name>.py`(怎么装)与 `spec.py`(怎么调)。
5. **`reference()` 用与算子相同 dtype 的原生 torch 调用**,不许先 `.float()`。它既是 check 的基准,也是 bench 的「相对 torch」标尺。
6. **文件即数据**:`runs/<id>/result.json` 是唯一的结果真源,体检单、对比表、基线、面板全从它推导。不引入数据库,面板不许有自己的持久化状态。
7. **面板不许引入 web 框架或构建步骤**:服务端是 stdlib `http.server`,前端是无构建单页,编辑器是预编译的 Monaco。它只读仓库与 `runs/`、只 fork `klab` 子进程。
8. **`klab/harness/` 在后端运行**,只能依赖 torch 与标准库。
9. **显式优先,有默认值**:后端用 `--target` 选,不给就用 `targets.toml` 的 `[defaults] target`。架构门禁只负责拒绝不匹配的组合并提示。
10. **不放凭据**:Modal 读 `~/.modal.toml`,SSH 读 `~/.ssh/config`,仓库里没有也不允许有 token。
11. **不为未验证的平台写代码**:没上过机的后端只加 `targets.toml` 配置,不写猜测性的适配器。
12. **任何改动先 `uv run pytest`**;改了跑 GPU 的部分按第五节上机验证,不能上机就明说没验证。
13. **改动必须同步**:加后端 / 加工具链 / 改契约 = 代码 + README 对应节 + `tests/`;改了 agent 的操作流程还要同步 playbook。同一件事只在一处完整解释,另一处链接。
14. **提交前先看 `git status`**:用户在面板里写算子,`kernels/` 随时可能有他的改动,别用 `git add -A` 一把扫进无关的 commit。
15. 提交信息用中文,写清改了哪一层。`runs/`、`.venv/`、`envs/tk/ThunderKittens/`、`klab/webui/vendor/` 不进 git。

## 二、启动

```bash
uv sync                    # 首次:装本地 CLI
uv run pytest              # 应全绿
uv run klab web            # 面板,http://127.0.0.1:8777,自动开浏览器
```

`klab web` 首次运行会自动取 Monaco 编辑器(MIT,约 24 MB)到 `klab/webui/vendor/`;取不到也能用,退回纯文本编辑框。Ctrl-C 退出。

新机器还需要:

1. `~/.ssh/config` 里有 `5090home` 别名(ProxyCommand 在局域网与 FRP 间自动选路,私钥 `~/.ssh/home_5090_local_ed25519`);`ssh 5090home true` 通了再继续。
2. Modal:`uv run modal setup` 登录一次,生成 `~/.modal.toml`。
3. 每种工具链在每个后端第一次用前 `klab setup --target <后端> --toolchain <名>`;远端 `~/klab/{repo,envs,runs}` 可随时删掉重建。
4. 本地 Nsight Compute GUI(`/Applications/NVIDIA Nsight Compute.app`)用于打开 `.ncu-rep`,非必需。

## 三、架构与数据流

```text
浏览器(klab web,stdlib http.server + 无构建单页 + Monaco)
      │  编辑器改的是 kernels/<名>/kernel.py;Run/Submit fork 出 klab 子进程
      ▼
klab CLI(uv venv,只装 typer/rich/modal)
      │  ① sync   rsync 仓库(排除 .git/.venv/runs)→ 远端 <root>/repo  |  Modal: add_local_dir 挂载
      │  ② run    ssh host bash -c "<命令串>"                           |  Modal: 远程函数里 bash -c
      │           命令串 = export PATH/PYTHONPATH/KLAB_* ; cd repo ; <python> -m klab.harness.runner ...
      │  ③ fetch  rsync 远端 <root>/runs/<id> → 本地 runs/<id>          |  Modal: 函数返回 runs 的 tar
      ▼
runs/<时间>-<算子>-<后端>-<模式>/
      result.json                              唯一真源
      report.md + ncu.ncu-rep + ncu-details.txt + ncu-raw.csv   (ncu 模式)
      ptx/<kernel>.ptx                         (ptx 模式)
      submission/                              当次运行的源码快照
      │
      ├─ klab report   → 体检单(kreport.py)
      ├─ klab compare  → 跨算子 / 跨后端表(compare.py)
      ├─ klab baseline → specs/<名>/baselines/<后端>.json,bench 时报差
      └─ klab web      → 面板的「结果」「提交记录」两个 tab
```

远端只有三样东西:`<root>/repo`(仓库副本)、`<root>/envs/<工具链>`(uv venv,SSH 后端)或镜像(Modal)、`<root>/runs`(结果)。远端不保存任何不能从 Mac 重建的状态,机器丢了重跑 `klab setup` 即可。

## 四、目录与职责

| 路径 | 职责 | 改动时注意 |
|---|---|---|
| `klab/cli.py` | 所有命令的入口;`_remote_run()` 是 check/bench/sweep/ptx 的公共路径,ncu 有自己的 `_ncu_impl()` | 新命令先看能否复用 `_remote_run` |
| `klab/config.py` | 读 `targets.toml`;`TargetConfig.extra` 原样透传后端私有选项 | 新字段优先走 `extra`,不改数据类 |
| `klab/targets/base.py` | Target 抽象;`env_prefix()` 决定远端 PATH/PYTHONPATH/KLAB_ROOT/KLAB_TK_ROOT | 远端找不到命令十有八九是这里 |
| `klab/targets/ssh.py` | rsync + ssh,ControlMaster 复用连接 | Mac 的 rsync 是 openrsync,只用 `-az --delete --exclude` |
| `klab/targets/modal.py` | 镜像构建、挂载、Volume 缓存、直连/代理选路、输出回流 | 见第六节 |
| `klab/targets/local.py` | 在 GPU 盒子上直接跑,调试 harness 用 | |
| `klab/toolchains/` | 每种工具链一个模块:`setup_script()` + 可选 `APT / MODAL_RUN_COMMANDS / MODAL_ENV / local_prepare()` | 纯 pip 的直接复用 `_pip.py` |
| `klab/harness/runner.py` | **在后端跑**;check / bench / ncu / sweep / ptx 五种模式;`ARCH_FEATURES` 门禁表 | 只能依赖 torch 与标准库 |
| `klab/harness/spec.py` | `specs/<名>/meta.toml` 与 `spec.py` 的契约;`kernel_module()` 按目录名导入 `kernels/<名>/kernel.py` | 契约变了要同步 README、playbook 与 tests |
| `klab/harness/probe.py` | 设备属性 + 实测带宽 / matmul 吞吐 | 实测值手工填回 `targets.toml` 的 `peak_*` |
| `klab/harness/cppext.py` | cuda / tk 工具链的 nvcc 现场编译 | 架构后缀、TK 宏、缓存目录都在这;`spec_sources` 让接线把 torch/pybind 绑定文件一起编,用户的 `.cu` 就只写 CUDA |
| `klab/harness/ptxdump.py` | 取 PTX、按指令族计数、判定命中世代 | `PTX_FAMILIES` 里只有标志指令进判定;取法在 `_COLLECTORS`,一种工具链一个 |
| `klab/kreport.py` | 体检单:raw CSV + details 文本 + 最近一次 bench → markdown | 指标名依赖 NCU 版本,tests 里守着 |
| `klab/report.py`、`klab/compare.py` | 终端表格;对比表 | |
| `klab/web.py` | 面板服务端:路由、白名单、markdown 子集转 HTML、源码快照、Monaco 取用 | 只读 `runs/` 与源码、只 fork 子进程;算子/后端/case 名一律先过白名单 |
| `klab/webui/` | 面板前端:`index.html` + `app.css` + `app.js`;`vendor/` 是 Monaco(gitignore) | 无构建,改完刷新即可 |
| `kernels/<名>/` | 算子源码(`kernel.py` / `kernel.cu` / `_vendor/`) | 面板的编辑器写它;agent 不擅自改 |
| `specs/<名>/` | 接线:`meta.toml`(含 `problem` 键)+ `spec.py` + `baselines/` | 生成规则见 playbook 第 1 节 |
| `problems/<题>/` | `problem.md` 题面、`editorial.md` 优化路线、`backbone/<工具链>.<后缀>` 骨架(函数体留空)、`solutions/<工具链>/N-名字.<后缀>` 参考答案阶梯 | 靠 `meta.toml` 的 `problem` 键与 `specs/` 关联;参考答案的结论必须是实测的 |
| `envs/<工具链>/` | `requirements.txt` + `torch-index.txt`;`envs/tk/ThunderKittens/` 是 Mac 上的克隆,gitignore | 故意不钉版本,见第七节 |
| `targets.toml` | 后端登记与峰值 | 加后端要同步 `.vscode/tasks.json` 的下拉(tests 守) |
| `tests/` | 不需要 GPU 的本地测试,夹具是一次真实的 5090 NCU 运行 | |
| `runs/` | 结果,gitignore | 可随时清空 |

## 五、验证矩阵(改了什么就跑什么)

| 改动 | 必跑 | 上机验证 |
|---|---|---|
| 任何改动 | `uv run pytest` | |
| `klab/cli.py`、`klab/config.py` | pytest | 一个算子的 `klab check --target 5090home` |
| `klab/targets/ssh.py`、`base.py` | pytest | 同上,再 `klab ncu` 一次 |
| `klab/targets/modal.py` | pytest | `klab check <算子> --target modal-h100`(首次会重建镜像) |
| `klab/harness/runner.py`、`spec.py` | pytest | 一个 triton 算子的 check + bench + ncu,一个 cuda 算子的 check |
| `klab/harness/cppext.py` | pytest | 一个 cuda 与一个 tk 算子,两个后端各一次 |
| `klab/kreport.py` | pytest(夹具) | `klab report` 一个已有的 ncu 运行目录 |
| `klab/harness/ptxdump.py` | pytest(手写 PTX 片段) | `klab ptx` 一个 triton 算子,确认命中世代与卡的架构相符 |
| `klab/web.py`、`klab/webui/` | pytest(markdown 子集、run 目录解析、题目聚合、白名单) | 面板里 Run 一次 + Submit 一次,确认判定、体检单、提交记录里的源码快照都对 |
| `klab/toolchains/*`、`envs/*` | pytest | 对应后端 `klab setup --toolchain X`,再跑该工具链的算子 |
| 新算子接线(`specs/<名>/`) | pytest(契约与目录结构) | `klab run <名>`,ncu 后确认 `kernel_regex` 只抓到自己的 kernel |
| 新后端 | pytest(tasks.json 下拉) | setup → probe → 一个算子的 check/bench/ncu |
| `targets.toml` 峰值 | pytest | 跑一次 `klab probe`,用实测值 |
| README / 本手册 | 无 | 按第四节的路径核对一遍 |

pytest 覆盖的是不需要 GPU 的部分:配置、算子契约、工具链登记、体检单解析与渲染、对比表、面板路由与渲染。**它证明不了算子在卡上是对的。**

## 六、踩过的坑(改相关代码前先看)

**SSH 后端**

- 非交互 ssh 的 PATH 不含 `~/.local/bin` 与 `/usr/local/cuda/bin`,`env_prefix()` 自己补;uv 装在远端 `~/.local/bin`。
- 环境变量里带 `~` 不会被 Python 展开,`cppext` 对 `KLAB_TK_ROOT` 做了 `expanduser`。
- 5090home 直连不了 github:凡是要 `git clone` 的东西,在 Mac 上克隆进仓库(gitignore)随 rsync 同步。ThunderKittens 就是这么处理的。
- ncu 的 `-k regex:` 参数必须 `shlex.quote`,正则里的 `|` 会被远端 shell 当管道。
- Mac 的 openrsync 删不掉远端的非空目录,`--delete` 会刷 `cannot delete non-empty directory`。本地删了目录后,远端要手动 `klab exec "rm -rf ..."` 清一次。

**Modal 后端**

- 本机 `HTTPS_PROXY`(127.0.0.1:3213)过不了 Modal 的 gRPC:`_choose_route()` 先试直连 `api.modal.com:443`,不通再走 `targets.toml` 的 `api_proxy`。决策必须在 `import modal` 之前,SDK 导入时读环境变量。
- 程序化 `app.run()` 必须包在 `modal.enable_output()` 里,否则容器 stdout 不回流,看起来像「跑了但没输出」。
- 容器里 ncu 可用、计数器可读,但锁不了 GPU 时钟:该后端 `ncu_clock_control = "none"`,数字比锁频的 5090 抖。
- 镜像按工具链分(`klab-<工具链>` app),CLI 通过 `KLAB_TOOLCHAIN` 环境变量告诉 `ModalTarget` 用哪个;忘了设会拿 triton 镜像去跑别的工具链。
- 编译缓存挂 Volume `klab-cache` 到 `/root/.cache`,否则 cuda / tk 每次冷启动重编一分钟。

**架构与工具链**

- 5090 是 sm_120:没有 wgmma / tcgen05,**但有 TMA**(`klab ptx` 实测 TileLang 在它上面发 `cp.async.bulk.tensor`)。共享内存上限 101376 字节,170 个 SM。`ARCH_FEATURES` 按 major 版本给特性,**cc 数字不是超集关系**。
- 5090 是消费卡,持续满载的大 GEMM(如 8192³)会降频,bench 的 p10 可能只有中位数的一半。这一档的绝对值不可比,只能和同时段的 torch 比。A100 没有这个问题。
- `klab probe` 的 `peak_tflops_fp16` 是用固定配置测的,会低于真实可达上限(A100 上 probe 测 241,autotune 后的 triton matmul 到 257.9)。峰值应该填「见过的最好成绩」,否则 %峰值 会超过 100%。
- TileLang 生成的 kernel 名是 `gemm_kernel`;CuTe DSL 的名字以 `kernel_cutlass_kernel_` 开头,`kernel_regex` 写 `cutlass`。写宽了会把 torch 造输入的 kernel 抓进报告、体检单取错行。
- ThunderKittens:sm_90 起要 `compute_XXa` 架构目标;宏 `KITTENS_SM<xx>` 只能定义一个;`gl` 的编译期维度要传 `nullptr`,用 `make_gl<GL>(ptr, b, d, r, c)` 省事;`warpid()` 在 `kittens::` 命名空间。
- torch 扩展里用 `getCurrentCUDAStream` 要 `#include <ATen/cuda/CUDAContext.h>`。
- Triton 3.8 的编译缓存是 `JITFunction.device_caches`(旧版叫 `cache`),device → tuple → dict 嵌套,层级各版本不同;`ptxdump._walk()` 按容器递归找叶子,不写死结构。
- PTX 里出现 `stmatrix` 不代表用上了 Hopper:它 sm_90 起就有,5090 照样发。判定只认标志指令(`mma.sync` / `wmma.mma` / `cp.async` / `wgmma` / `cp.async.bulk` / `tcgen05`)。
- `nvcuda::wmma` 在 PTX 里是 `wmma.mma.sync`,**不是** `mma.sync` —— 漏了这一族会把手写 WMMA 的 kernel 判成「没走 tensor core」。
- nvcc 的 `-gencode` 只写 `code=sm_XX` 时产物里不嵌 PTX,`cuobjdump -ptx` 什么也抠不出来;要 `code=[sm_XX,compute_XX]`。
- 每种工具链取 PTX 的办法都不一样:triton 读 `JITFunction.device_caches` 里的 `asm['ptx']`;tilelang 用 `JITKernel._get_ptx()`;cuda / tk 用 `cuobjdump -ptx` 抠 `.so`。CuTe DSL 4.7.1 取不到(`artifacts.PTX` 字段在但填不上,打开 `DeviceTarget` 也是 None)。
- `klab/harness/runner.py` 顶层 import torch,本地没有 —— 想在 pytest 里读 `ARCH_FEATURES` 这类常量得用 AST 静态解析,不能 import。

**数值**

- bf16 matmul 的 torch 参考:PyTorch 的 `allow_bf16_reduced_precision_reduction` 默认 True,块数喂不满 GPU 时 cuBLAS 走 split-K 并用 bf16 归约部分和,给接近 0 的输出带来约 0.1 的绝对误差,check 会误判成算子写错。`spec.py` 里关掉它;实测对速度的影响在 ±1% 噪声内。判断方法:拿 `a.float() @ b.float()` 当真值,看是算子离得远还是 torch 离得远。

**NCU / 报告**

- 5090 上 ncu 报带宽用 `Tbyte/s`,H100 用 `Gbyte/s`,`kreport._gbps()` 统一。
- ncu 对超出默认 carveout 的动态共享内存,`launch__occupancy_limit_shared_mem` 报 0,不能当限制因子。

**面板**

- CSS 里给 `.sheet` 设了 `display:flex` 会盖掉 `[hidden]` 的默认 `display:none`,遮罩会常驻在最上层。`app.css` 顶部的 `[hidden] { display:none !important }` 兜着。

## 七、版本策略

`envs/*/requirements.txt` 故意不钉版本:torch 取 PyTorch 官方 cu130 索引的最新版,triton 随 torch,tilelang / nvidia-cutlass-dsl 取 PyPI 最新。`klab setup` 会打印实际装到的版本。某次升级把哪条工具链弄坏了,就在对应 `requirements.txt` 里钉住上一个可用版本并在注释里写日期与原因;Modal 镜像会随 requirements 变化自动重建。

面板的 Monaco 版本钉在 `klab/web.py` 的 `MONACO_VERSION`。

已验证可用的组合:torch 2.14.0+cu130、triton 3.8.0、tilelang 0.1.14、nvidia-cutlass-dsl 4.7.1、ThunderKittens main、monaco-editor 0.56.0。
