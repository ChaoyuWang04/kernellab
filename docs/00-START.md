# 维护手册:系统怎么运转,改哪里,怎么验证

Mac 上写 GPU 算子,远端 GPU 上编译、跑、测速、抓 NCU,结果回流成一张固定模板的体检单。日常入口是 `klab web`(算子版 LeetCode),CLI 是它的全集。

现在的规模:**一道题**(matmul)× **五种语言**(triton / tilelang / cuda / cute / tk)× **四张卡**(sm_80 / 90 / 100 / 120)× **六档形状**,18 份参考答案,全部上机验证过。

| 要什么 | 去哪 |
|---|---|
| 守则与边界 | [../CLAUDE.md](../CLAUDE.md) |
| 接到「测一下这个 kernel」怎么做 | [01-AGENT-PLAYBOOK.md](01-AGENT-PLAYBOOK.md) |
| 还没做的事 | [02-NEXT.md](02-NEXT.md) |
| 某条守则为什么这么定 | [03-DECISIONS.md](03-DECISIONS.md) |
| **改代码前查一遍踩过的坑** | [04-PITFALLS.md](04-PITFALLS.md) |
| 命令与契约的用法 | [../README.md](../README.md) |

## 一、启动

```bash
uv sync                    # 首次:装本地 CLI
uv run pytest              # 应全绿
uv run klab web            # 面板,http://127.0.0.1:8777,自动开浏览器
```

首次运行 `klab web` 会自动取 Monaco(MIT,约 24 MB)到 `klab/webui/vendor/`;取不到也能用,退回纯文本编辑框。Ctrl-C 退出。

新机器还需要:

1. `~/.ssh/config` 里有 `5090home` 别名(ProxyCommand 在局域网与 FRP 间自动选路,私钥 `~/.ssh/home_5090_local_ed25519`);`ssh 5090home true` 通了再继续。
2. Modal:`uv run modal setup` 登录一次,生成 `~/.modal.toml`。三张云卡都走它:`modal-a100`(sm_80)、`modal-h100`(sm_90)、`modal-b200`(sm_100)。
3. 每种工具链在每个后端第一次用前 `klab setup --target <后端> --toolchain <名>`;远端 `~/klab/{repo,envs,runs}` 可随时删掉重建。
4. 本地 Nsight Compute GUI(`/Applications/NVIDIA Nsight Compute.app`)用于打开 `.ncu-rep`,非必需。

## 二、架构与数据流

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

远端只有三样东西:`<root>/repo`(仓库副本)、`<root>/envs/<工具链>`(uv venv,SSH 后端)或镜像(Modal)、`<root>/runs`(结果)。**远端不保存任何不能从 Mac 重建的状态**,机器丢了重跑 `klab setup` 即可。

## 三、目录与职责

| 路径 | 职责 | 改动时注意 |
|---|---|---|
| `klab/cli.py` | 所有命令的入口;`_remote_run()` 是 check/bench/sweep/ptx 的公共路径,ncu 有自己的 `_ncu_impl()` | 新命令先看能否复用 `_remote_run` |
| `klab/config.py` | 读 `targets.toml`;`TargetConfig.extra` 原样透传后端私有选项 | 新字段优先走 `extra`,不改数据类 |
| `klab/targets/base.py` | Target 抽象;`env_prefix()` 决定远端 PATH/PYTHONPATH/KLAB_ROOT/KLAB_TK_ROOT | 远端找不到命令十有八九是这里 |
| `klab/targets/ssh.py` | rsync + ssh,ControlMaster 复用连接 | Mac 的 rsync 是 openrsync,只用 `-az --delete --exclude` |
| `klab/targets/modal.py` | 镜像构建、挂载、Volume 缓存、直连/代理选路、输出回流 | 坑多,见 04-PITFALLS |
| `klab/targets/local.py` | 在 GPU 盒子上直接跑,调试 harness 用 | |
| `klab/toolchains/` | 每种工具链一个模块:`setup_script()` + 可选 `APT / MODAL_RUN_COMMANDS / MODAL_ENV / local_prepare()` | 纯 pip 的直接复用 `_pip.py` |
| `klab/harness/runner.py` | **在后端跑**;check / bench / ncu / sweep / ptx 五种模式;`ARCH_FEATURES` 门禁表 | 只能依赖 torch 与标准库 |
| `klab/harness/spec.py` | `specs/<名>/meta.toml` 与 `spec.py` 的契约;`kernel_module()` 按目录名导入 `kernels/<名>/kernel.py` | 契约变了要同步 README、playbook 与 tests |
| `klab/harness/probe.py` | 设备属性 + 实测带宽 / matmul 吞吐 | 实测值手工填回 `targets.toml` 的 `peak_*` |
| `klab/harness/cppext.py` | cuda / tk 的 nvcc 现场编译 | 架构后缀、TK 宏、缓存目录都在这;`spec_sources` 让接线把 torch/pybind 绑定一起编,用户的 `.cu` 就只写 CUDA |
| `klab/harness/ptxdump.py` | 取 PTX、按指令族计数、判定命中世代 | `PTX_FAMILIES` 里只有标志指令进判定;取法在 `_COLLECTORS`,一种工具链一个。已接 triton / tilelang / cuda / tk,cute 接不上 |
| `klab/kreport.py` | 体检单:raw CSV + details 文本 + 最近一次 bench → markdown | 指标名依赖 NCU 版本,tests 里守着 |
| `klab/report.py`、`klab/compare.py` | 终端表格;对比表 | |
| `klab/web.py` | 面板服务端:路由、白名单、markdown 子集转 HTML、源码快照、Monaco 取用 | 只读 `runs/` 与源码、只 fork 子进程;算子/后端/case 名一律先过白名单 |
| `klab/webui/` | 面板前端:`index.html` + `app.css` + `app.js`;`vendor/` 是 Monaco(gitignore) | 无构建,改完刷新即可 |
| `kernels/<名>/` | 算子源码(`kernel.py` / `kernel.cu` / `_vendor/`) | 面板的编辑器写它;agent 不擅自改 |
| `specs/<名>/` | 接线:`meta.toml`(含 `problem` 键)+ `spec.py` + `baselines/` | 生成规则见 playbook 第 1 节 |
| `problems/<题>/` | `problem.md` 题面、`editorial.md` 优化路线、`backbone/<工具链>.<后缀>` 骨架(函数体留空)、`solutions/<工具链>/N-名字.<后缀>` 参考答案阶梯 | 参考答案的结论必须是实测的 |
| `envs/<工具链>/` | `requirements.txt` + `torch-index.txt`;`envs/tk/ThunderKittens/` 是 Mac 上的克隆,gitignore | 故意不钉版本,见 [D11](03-DECISIONS.md#d11-版本不钉) |
| `targets.toml` | 后端登记与峰值 | 加后端要同步 `.vscode/tasks.json` 的下拉(tests 守) |
| `tests/` | 不需要 GPU 的本地测试,夹具是一次真实的 5090 NCU 运行 | |
| `runs/` | 结果,gitignore | 可随时清空 |

## 四、验证矩阵(改了什么就跑什么)

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
| `klab/web.py`、`klab/webui/` | pytest(markdown 子集、run 目录解析、题目聚合、白名单) | 面板里 Run 一次 + Submit 一次,确认判定、体检单、源码快照都对 |
| `klab/toolchains/*`、`envs/*` | pytest | 对应后端 `klab setup --toolchain X`,再跑该工具链的算子 |
| 新算子接线(`specs/<名>/`) | pytest(契约与目录结构) | `klab run <名>`,ncu 后确认 `kernel_regex` 只抓到自己的 kernel |
| 新参考答案(`problems/*/solutions/`) | pytest | `klab run` + `klab ptx`,数字进 docstring |
| 新后端 | pytest(tasks.json 下拉) | setup → probe → 一个算子的 check/bench/ncu |
| `targets.toml` 峰值 | pytest | 跑一次 `klab probe`,用实测值 |
| 文档 | 无 | 按第三节的路径核对一遍 |

pytest 覆盖的是不需要 GPU 的部分:配置、算子契约、工具链登记、体检单解析与渲染、对比表、面板路由与渲染。**它证明不了算子在卡上是对的。**

## 五、改动的同步义务

加后端 / 加工具链 / 改契约 = **代码 + README 对应节 + `tests/`**;改了 agent 的操作流程还要同步 playbook;踩了新坑记进 [04-PITFALLS.md](04-PITFALLS.md);定了新规矩记进 [03-DECISIONS.md](03-DECISIONS.md)。

**同一件事只在一处完整解释,另一处链接。**

已验证可用的组合:torch 2.14.0+cu130、triton 3.8.0、tilelang 0.1.14、nvidia-cutlass-dsl 4.7.1、ThunderKittens main、monaco-editor 0.56.0。面板的 Monaco 版本钉在 `klab/web.py` 的 `MONACO_VERSION`。
