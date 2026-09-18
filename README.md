# kernellab

Mac 上写算子,远端 GPU 上编译、跑、测速、NCU。执行后端可插拔,用 `--target` 显式选。

```text
Mac VSCode ── klab CLI ──┬── ssh:5090home   (home lab,RTX 5090)
                         ├── ssh:<租的机器> (vast / runpod / autodl,同一套 SshTarget)
                         ├── modal          (下一阶段)
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

## 加一个后端

`targets.toml` 加一段。`kind = "ssh"` 的只需要 `host`(`~/.ssh/config` 里的别名)与 `root`,其余走默认。远端要有:能跑 CUDA 的驱动、`curl`、`rsync`;`klab setup` 会自己装 uv 与 venv。

`klab/targets/` 里每个后端一个类,只实现 `sync / run / fetch / shell` 四个方法;编译、跑、测速、NCU 都是 `run()` 之上的命令串,不进后端类。

## 加一种工具链

`klab/toolchains/<name>.py` 给出 `setup_script()`,`envs/<name>/` 放依赖清单,`klab/toolchains/__init__.py` 登记。纯 pip 的(Triton、TileLang、CuTe DSL)照 triton 那样写;需要特定 nvcc 的(ThunderKittens、裸 CUDA)后续用 Docker 镜像。

## 路线

1. ✅ SSH 后端 + Triton + check / bench / ncu,打通 5090home
2. Modal 后端(同一份 requirements),跑一个 5090 跑不了的 Hopper 算子
3. TileLang、CuTe DSL、ThunderKittens 三套工具链
4. `--target auto`:按 `requires` 匹配后端
5. 昇腾:CANN 环境 + msprof 适配
