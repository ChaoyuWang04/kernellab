# kernellab

Mac 上写算子,远端 GPU 上编译、跑、测速、NCU。执行后端可插拔,用 `--target` 显式选。

```text
Mac VSCode ── klab CLI ──┬── ssh:5090home   (home lab,RTX 5090)
                         ├── ssh:<租的机器> (vast / runpod / autodl,同一套 SshTarget)
                         ├── modal-h100     (Modal 云 GPU,gpu 字段选卡型)
                         └── local          (在 GPU 盒子上调试 harness 自己用)
```

三层分离:**代码住在 Mac**(本仓库,git 管理);**环境定义在仓库里**(`envs/<工具链>/`);**后端只回答「在哪跑」**(`targets.toml`)。

## 日常用法

```bash
uv sync                                        # 首次,装本地 CLI
uv run klab targets                            # 列后端
uv run klab setup --target 5090home            # 在后端装 Triton 环境(幂等)
uv run klab probe --target 5090home            # 设备属性 + 实测拷贝带宽 + fp16 matmul 吞吐
uv run klab check kernels/softmax --target 5090home    # 正确性
uv run klab bench kernels/softmax --target 5090home    # 先 check 再测速
uv run klab ncu   kernels/softmax --target 5090home --case 8192x8192-f16 --open   # NCU,拉回报告并用本地 Nsight Compute 打开
uv run klab sh    --target 5090home            # 进后端 shell(已 cd 到远端 repo)
uv run klab exec  "nvidia-smi" --target modal-h100   # 在后端执行一段命令(诊断)
uv run klab report runs/<某次 ncu 运行>          # 重新渲染体检单
```

VSCode 里打开 `kernel.py`,`Cmd+Shift+B` 或 `Tasks: Run Task` 选 `klab: check / bench / ncu 当前算子`,任务会弹出后端选择框。想绑快捷键,在用户级 `keybindings.json` 加:

```json
{ "key": "cmd+k cmd+b", "command": "workbench.action.tasks.runTask", "args": "klab: bench 当前算子" }
```

每次运行在 `runs/<时间>-<算子>-<后端>-<模式>/` 落一份 `result.json`;NCU 另有 `ncu.ncu-rep`(GUI 打开)、`ncu-details.txt`(直接在编辑器里读)、`ncu-raw.csv`。`runs/` 不进 git。

## 写一个算子

一个目录两份文件:

```text
kernels/<name>/
  meta.toml     工具链、ncu 过滤名、架构要求、容差、case 列表
  kernel.py     make_inputs / run / reference / workload 四个函数
```

`kernel.py` 的契约与 DSL 无关:

| 函数 | 作用 |
|---|---|
| `make_inputs(case, device) -> dict` | 按 case 造输入,固定种子 |
| `run(**inputs) -> Tensor` | 调用你的算子 |
| `reference(**inputs) -> Tensor` | torch 参考实现,check 用 |
| `workload(case, **inputs) -> {flops, bytes}` | 换算 TFLOPS 与 GB/s |

`meta.toml` 的 `requires.min_cc` 与 `features` 是给后续 `--target auto` 用的:5090 是 sm_120,没有 wgmma / tcgen05 / cluster,Hopper 优先的算子要标出来,到时候自动路由到 Modal 的 H100。

样板:`kernels/vector_add`(最小链路)、`kernels/softmax`(融合行 softmax)。

## 测速方法

预热 → 每次迭代前用 256 MB 的写把 L2 冲干净 → CUDA event 计时 → 取中位数与 p10/p90 → 按 `workload()` 换算带宽与算力 → 对 `targets.toml` 里的峰值算百分比。`--no-flush` 关掉 L2 冲刷(看热缓存表现),`--iters/--warmup` 调次数。

`peak_*` 是手填的参考值:`peak_gbps` 取公开规格,`peak_tflops_fp16` 取 `klab probe` 实测的 matmul 吞吐。

## NCU

默认抓 SpeedOfLight / MemoryWorkloadAnalysis / Occupancy / LaunchStats / ComputeWorkloadAnalysis 五个 section,`--full` 换成 `--set full`(replay 次数多很多)。`meta.toml` 的 `kernel_regex` 用作 `-k regex:` 过滤,只抓你的 kernel,不抓 torch 造输入的那些。

5090home 已把驱动的 `RestrictProfilingToAdminUsers` 关掉,非 root 可用;租来的机器需要同样条件或 root,容器里还要 `--cap-add=SYS_ADMIN`。

Modal 的 H100 容器里 ncu 可用、计数器可读(2026-09-18 实测),但锁不了 GPU 时钟,所以 `targets.toml` 里给它设了 `ncu_clock_control = "none"`;数字会比锁频的 5090 抖一些。

## 算子体检单(固定模板)

`klab ncu` 跑完自动打印并写入 `runs/<id>/report.md`;`klab report <run_dir>` 可重新渲染。模板固定,任何算子、任何后端都一样,便于横着比。数据来自 NCU 的 raw CSV 与 details 文本,加上同一算子、同一后端最近一次 bench 的数字。

| 段 | 回答什么 |
|---|---|
| 判定 | 一句话:内存侧 / 计算侧 / 延迟受限 / 接近均衡,加实测占用率。规则与 NCU 自己的一致:最高单元 ≥ 80% 判该侧受限,都 < 60% 判延迟受限 |
| 速度 | NCU 内核时长、bench 中位数与分位数、TFLOPS 与 GB/s 对峰值的百分比、算术强度对 roofline 拐点在哪一侧 |
| 各单元利用率 | SM、Tensor pipe、FMA/ALU/LSU、内存总、DRAM、L2(命中率)、L1(命中率) |
| 发射与占用 | grid × block、波数、寄存器与共享内存用量、理论与实测占用率、**占用限制因子** |
| 调度与 stall | 每调度器活跃与可发射 warp 数、前四个 stall 原因 |
| NCU 建议 | details 页里的 OPT 段落原文摘录(含 Est. Speedup) |

读法:先看判定,再看对应那一段。内存侧受限看 DRAM 是否已到 80% 以上(是就到头了,要减字节数);计算侧看 Tensor pipe 与占用限制因子;延迟受限看 stall 原因与可发射 warp 数。

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
uv run klab check kernels/matmul --target 5090home                    # 拒:sm_120 缺 wgmma
uv run klab bench kernels/matmul --target 5090home --ignore-requires  # 强行跑,拿对照
uv run klab bench kernels/matmul --target modal-h100                  # 换后端,原样跑
```

## 加一个后端

`targets.toml` 加一段。`kind = "ssh"` 的只需要 `host`(`~/.ssh/config` 里的别名)与 `root`,其余走默认。远端要有:能跑 CUDA 的驱动、`curl`、`rsync`;`klab setup` 会自己装 uv 与 venv。

`klab/targets/` 里每个后端一个类,只实现 `sync / run / fetch / shell` 四个方法;编译、跑、测速、NCU 都是 `run()` 之上的命令串,不进后端类。

## 加一种工具链

`klab/toolchains/<name>.py` 给出 `setup_script()`,`envs/<name>/` 放依赖清单,`klab/toolchains/__init__.py` 登记。纯 pip 的(Triton、TileLang、CuTe DSL)照 triton 那样写;需要特定 nvcc 的(ThunderKittens、裸 CUDA)后续用 Docker 镜像。

## 路线

1. ✅ SSH 后端 + Triton + check / bench / ncu,打通 5090home
2. ✅ Modal 后端(同一份 requirements)+ 架构特性门禁,matmul 在 5090 被拒、在 H100 原样跑
3. TileLang、CuTe DSL、ThunderKittens 三套工具链
4. `--target auto`:按 `requires` 匹配后端
5. 昇腾:CANN 环境 + msprof 适配
