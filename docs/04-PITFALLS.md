# 踩过的坑

> 改相关代码前先查这里。每一条都是真跑出来的,不是推测。
> 结论性的「为什么这么设计」在 [03-DECISIONS.md](03-DECISIONS.md)。

## SSH 后端

- 非交互 ssh 的 PATH 不含 `~/.local/bin` 与 `/usr/local/cuda/bin`,`env_prefix()` 自己补;uv 装在远端 `~/.local/bin`。
- 环境变量里带 `~` 不会被 Python 展开,`cppext` 对 `KLAB_TK_ROOT` 做了 `expanduser`。
- **5090home 连不上 github**:凡是要 `git clone` 的东西,在 Mac 上克隆进仓库(gitignore)随 rsync 同步。ThunderKittens 就是这么处理的。
- ncu 的 `-k regex:` 参数必须 `shlex.quote`,正则里的 `|` 会被远端 shell 当管道。
- Mac 的 openrsync 删不掉远端的非空目录,`--delete` 会刷 `cannot delete non-empty directory`。本地删了目录后,远端要手动 `klab exec "rm -rf ..."` 清一次。

## Modal 后端

- 本机 `HTTPS_PROXY`(127.0.0.1:3213)过不了 Modal 的 gRPC:`_choose_route()` 先试直连 `api.modal.com:443`,不通再走 `targets.toml` 的 `api_proxy`。**决策必须在 `import modal` 之前**,SDK 导入时读环境变量。
- 程序化 `app.run()` 必须包在 `modal.enable_output()` 里,否则容器 stdout 不回流,看起来像「跑了但没输出」。
- 容器里 ncu 可用、计数器可读,但锁不了 GPU 时钟:该后端 `ncu_clock_control = "none"`,数字比锁频的 5090 抖。
- 镜像按工具链分(`klab-<工具链>` app),CLI 通过 `KLAB_TOOLCHAIN` 环境变量告诉 `ModalTarget` 用哪个;忘了设会拿 triton 镜像去跑别的工具链。
- 编译缓存挂 Volume `klab-cache` 到 `/root/.cache`,否则 cuda / tk 每次冷启动重编一分钟。
- 容器按调用起停、按秒计费:bench 循环要在**一次**调用里做完,不要一个 case 一次调用。

## 架构与指令

- **四张卡的实测画像**(同一份 Triton 代码,`klab ptx`):A100 sm_80 → `mma.sync`+`cp.async`;5090 sm_120 → `mma.sync`+TMA;H100 sm_90 → `wgmma`+TMA;B200 sm_100 → `tcgen05`+TMA。
- **cc 数字不是超集关系**:5090 是 sm_120,数字最大,却没有 `wgmma`。门禁表 `ARCH_FEATURES` 必须按卡逐个列,不能按大小推。
- **Triton 的 TMA API(`tl.make_tensor_descriptor`)在 sm_80 上静默降级成普通 `cp.async`** —— 不报错、不变慢,只是不是 TMA。「用了 API」必须用 `klab ptx` 验成「硬件真的用了」。
- 用了 `tl.make_tensor_descriptor` 的 kernel 需要宿主侧 `triton.set_allocator(...)`,否则运行期报 "no allocator was set"。接线里一直开着,不用的级别不占内存。
- PTX 里出现 `stmatrix` **不代表**用上了 Hopper:它 sm_90 起就有,5090 照样发。判定只认标志指令。
- `nvcuda::wmma` 在 PTX 里是 `wmma.mma.sync`,**不是** `mma.sync` —— 漏了这一族会把手写 WMMA 的 kernel 判成「没走 tensor core」。
- nvcc 的 `-gencode` 只写 `code=sm_XX` 时产物里不嵌 PTX,`cuobjdump -ptx` 什么也抠不出来;要 `code=[sm_XX,compute_XX]`。
- 每种工具链取 PTX 的办法都不一样:triton 读 `JITFunction.device_caches` 里的 `asm['ptx']`;tilelang 用 `JITKernel._get_ptx()`;cuda / tk 用 `cuobjdump -ptx` 抠 `.so`。CuTe DSL 4.7.1 取不到(见 [D10](03-DECISIONS.md#d10-cute-的-ptx-先不接))。
- Triton 3.8 的编译缓存是 `JITFunction.device_caches`(旧版叫 `cache`),device → tuple → dict 嵌套,层级各版本不同;`ptxdump._walk()` 按容器递归找叶子,不写死结构。

## 工具链

**TileLang**

- `tilelang.jit(out_idx=[-1])` 会自己分配输出;split-K 需要接线预先清零并传进去,这时**不能标 `out_idx`**,否则报 `Kernel expected 2 inputs, but 3 are provided`。
- 共享内存超限报 `Failed to set the allowed dynamic shared memory size to N`,N 就是它要的字节数。5090 上限 101376,H100 是 228 KB。
- 生成的 kernel 名是 `gemm_kernel`。

**CuTe DSL**

- kernel 名以 `kernel_cutlass_kernel_` 开头,`kernel_regex` 写 `cutlass`。写宽了会把 torch 造输入的 kernel 抓进报告、体检单取错行。

**ThunderKittens**

- sm_90 起要架构专属目标(`compute_90a` 这种带 `a` 的);宏 `KITTENS_SM<xx>` 只能定义一个。
- **必须显式 `-lcuda`**,否则编译通过、导入时才报 `undefined symbol: cuGetErrorString`。`cppext` 在 `tk=True` 时自动加。
- 取共享内存子块的成员函数叫 `subtile`(`As.template subtile<R,C>({r,c})`),不是 `subtile_inplace`。
- `gl` 的编译期维度要传 `nullptr`,用 `make_gl<GL>(ptr, b, d, r, c)` 省事;`warpid()` 在 `kittens::` 命名空间。
- **tile 操作要求 M/N/K 都是 tile 尺寸的整数倍**,不规整形状要调用方 pad。所以 `specs/matmul_tk/` 的 case 列表比另外四种语言少一档。

**裸 CUDA**

- torch 扩展里用 `getCurrentCUDAStream` 要 `#include <ATen/cuda/CUDAContext.h>`。

**harness 本身**

- `klab/harness/runner.py` 顶层 import torch,本地没有 —— 想在 pytest 里读 `ARCH_FEATURES` 这类常量得用 AST 静态解析,不能 import。

## 数值

- **bf16 matmul 的 torch 参考**:PyTorch 的 `allow_bf16_reduced_precision_reduction` 默认 True,块数喂不满 GPU 时(如 1000×999×777 只有 64 块、5090 有 170 个 SM)cuBLAS 走 split-K 并用 bf16 归约部分和,给**接近 0 的输出**带来约 0.1 的绝对误差,check 会误判成算子写错。`spec.py` 里关掉它;实测对速度的影响在 ±1% 噪声内。
- **判断方法**:拿 `a.float() @ b.float()` 当真值,看是算子离得远还是 torch 离得远。上次就是这么查出来「算子比 torch 更准」的(0.2577 vs 0.5938)。

## 测速与 NCU

- **bench 的 GB/s / %BW 是「有效带宽」**:`workload()` 的数学最小字节数 ÷ 时间。工作集被缓存吃下时会严重高估 —— `smallK` 那档报 1689 GB/s(94%),NCU 看显存只有 3%,差 35 倍。要真实显存流量就看体检单的「显存」那一行。
- 5090 是消费卡,持续满载的大 GEMM(如 8192³)会降频,bench 的 p10 可能只有中位数的一半。这一档的绝对值不可比,只能和同时段的 torch 比。A100 没有这个问题。
- `klab probe` 的 `peak_tflops_fp16` 是用固定配置测的,会低于真实可达上限(A100 上 probe 测 241,autotune 后的 triton matmul 到 257.9)。峰值应该填「见过的最好成绩」,否则 %峰值会超过 100%。
- 5090 上 ncu 报带宽用 `Tbyte/s`,H100 用 `Gbyte/s`,`kreport._gbps()` 统一。
- ncu 对超出默认 carveout 的动态共享内存,`launch__occupancy_limit_shared_mem` 报 0,不能当限制因子。
- 尾波浪费的算法容易写反:**3.01 波比 3.98 波糟得多**(前者最后一波只有 1% 满,浪费 25%)。`_tail_wave()` 用 `math.ceil`,tests 里有回归。

## 面板

- CSS 里给 `.sheet` 设了 `display:flex` 会盖掉 `[hidden]` 的默认 `display:none`,遮罩会常驻在最上层挡住 tab 栏。`app.css` 顶部的 `[hidden] { display:none !important }` 兜着。
- Monaco 没加载好时 `code()` 必须返回 `null` 而不是 `''`,否则 Run / Submit 会把用户的 `kernel.py` 静默清空。
