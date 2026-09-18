# kernellab

Mac 上写算子,远端 GPU 上编译、跑、测速、NCU。执行后端可插拔,用 `--target` 显式选。

本文面向使用。要维护或扩展这个项目,先读 **[docs/00-START.md](docs/00-START.md)**(为什么这么设计、目录职责、验证矩阵、踩过的坑、未接入平台怎么接)。本地测试:`uv run pytest`。

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
uv run klab compare matmul matmul_tl            # 跨 DSL / 跨后端的 bench 对比表(读 runs/,不联网)
uv run klab sweep kernels/matmul --target 5090home --case 4096-f16   # 按 meta.toml [sweep] 扫参,列每个 case 最快的几组
uv run klab baseline kernels/matmul --target 5090home                # 把最新 bench 钉成基线;之后 bench 自动报与基线的差
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

样板(五种工具链各至少一个):

| 目录 | 工具链 | 说明 |
|---|---|---|
| `vector_add`、`softmax` | triton | 最小链路;融合行 softmax |
| `matmul` | triton | 分块 matmul,声明需要 wgmma(演示门禁),带 `[sweep]` |
| `matmul_tl` | tilelang | 与上面同尺寸,跨 DSL 对照,带 `[sweep]` |
| `matmul_cute` | cute | NVIDIA 官方 CuTe DSL 示例原样 vendor 在 `_vendor/`,按卡挑 sm_120 或 Hopper 类 |
| `sgemm_cuda` | cuda | 经典共享内存分块 SGEMM(fp32、CUDA core),`.cu` 现场 nvcc 编成 torch 扩展 |
| `tile_add_tk` | tk | ThunderKittens 烟测:寄存器 tile 的 load / add / store |

`cuda` 与 `tk` 算子的 `kernel.py` 通过 `klab.harness.cppext.load_extension()` 编译同目录的 `.cu`,缓存在后端的 `~/.cache/klab/<名>-<架构>`(Modal 用 Volume 持久化)。

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

## 扫参与基线

`meta.toml` 加 `[sweep]`(参数名 = 候选列表),`kernel.py` 提供 `configure(**params)` 把一组参数应用到算子(改全局常量、清编译缓存)。`klab sweep` 在后端按笛卡尔积逐组 check + bench,编译失败或共享内存超限的组记为失败而不中断,最后按 case 列出最快的几组。5090 的共享内存上限是 101376 字节,`BLOCK_N=256` 配 4 级流水会超。

`klab baseline` 把某次 bench 复制到 `kernels/<名>/baselines/<后端>.json`(进 git)。之后同一后端的 `klab bench` 自动打印「现在 / 基线 / 变化」,±3% 以外标色。

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
