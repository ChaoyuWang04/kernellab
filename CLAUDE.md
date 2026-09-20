# AI 维护入口

**学写 GPU 算子的个人实验台**:Mac 上写、远端 GPU 上跑、性能指标回流成一张体检单。
日常入口 `uv run klab web` —— 对标 LeetCode 的本地面板,用户在里面写算子。

**整个项目面向 agent 调用**:用户只写算子本体(落到 `kernels/<名>/`),其余(接线、题面、参考答案、选后端、跑、读 NCU、评价)都是 agent 的活。

## 读哪些文档

| 什么时候 | 读 |
|---|---|
| 用户说「测一下这个 kernel」 | **[docs/01-AGENT-PLAYBOOK.md](docs/01-AGENT-PLAYBOOK.md)** —— 默认行为,按它做 |
| 要改代码 | **[docs/00-START.md](docs/00-START.md)** 系统怎么运转 + [docs/04-PITFALLS.md](docs/04-PITFALLS.md) 踩过的坑 |
| 要接着干活 | **[docs/02-NEXT.md](docs/02-NEXT.md)** 还没做什么、怎么做 |
| 想改某条守则 | **[docs/03-DECISIONS.md](docs/03-DECISIONS.md)** 它当初为什么那么定 |
| 查命令用法 | [README.md](README.md) |

动手前先 `git status`,确认没有别人的未提交工作。

## 定位(写参考答案前必读)

> **学的是「如何把算子优化到工业级」,不是「从零手搓一个工业级算子」。**

每种语言走它**官方推荐**的路。判定方法:问「官方文档在这一格推荐什么」,照做 —— 答案经常就是手写。

- 合并访存 / 分块 / 双缓冲 / Ampere 的 WMMA:**官方就是教手写,那就手写**。
- Hopper 之后的 `wgmma` / `tcgen05` 协议:官方自己都不推荐手写,用它给的封装(裸 CUDA 走 CUTLASS,tk 换命名空间,cute 换 atom)。
- Triton / TileLang:代码不用动,编译器按目标卡选指令,我们的活是**去量**。

**这不等于「一律用 CUTLASS」**,只有「官方没有手写封装」的那几格才需要。完整判定表与原话见 [D1](docs/03-DECISIONS.md#d1-这个仓库学的是优化不是手搓)。

一级的价值在于让用户**感受到某个旋钮动了之后数字怎么变**,不在于写起来多难。**结论必须实测,不编造单调递增的阶梯**;没收益就写没收益并解释为什么。

## 与用户的协作方式

- 用户一句话「测一下 X」之后,接线、选后端、跑、读体检单、评价**全做完再汇报**,中途不逐步请示。需要拍板的只有:改用户的算子源码、放宽容差、动他没点名的付费资源、删东西。
- 汇报用紧凑表格,先给判定再给数字;评价落到源码的行或常量;不复述 NCU 全部指标。
- 用户常用语音输入,术语可能被识别错:Modal → "model"、TileLang → "Triton-TEL"、ThunderKittens → "thunder / kittens"、CuTe DSL → "long CUTE"。按上下文理解,不要按字面追问。
- 有疑问先看 `runs/<id>/report.md` 与源码再问;问的时候给选项并附推荐。

## 不可破坏的约束

1. **`kernels/<名>/` 是用户的** —— 只有算子源码,不改、不加文件,除非用户要求。接线一律放 `specs/<名>/`。
2. **Target 接口只有 `sync / run / fetch / shell` 四个方法**;编译、测速、NCU 一律是 `run()` 之上的命令串,不进后端类。
3. **spec 契约只有 `make_inputs / run / reference / workload`**(可选 `configure`),harness 不认识任何 DSL。`reference()` 用与算子相同 dtype 的原生 torch 调用,不许先 `.float()`。
4. **`klab/harness/` 在后端运行**,只能依赖 torch 与标准库。
5. **不放凭据** —— Modal 读 `~/.modal.toml`,SSH 读 `~/.ssh/config`;仓库里没有也不允许有 token。
6. **`kernels/*/_vendor/`、`envs/tk/ThunderKittens/` 是第三方原文**,不改。
7. **不引数据库、不引 web 框架、不做 `--target auto`**。面板只绑 `127.0.0.1`、只读 `runs/` 与仓库源码、只 fork `klab` 子进程,不做多用户,不许有自己的持久化状态。
8. **不为未验证的平台写代码** —— 没上过机的后端只加 `targets.toml` 配置,不写猜测性的适配器。
9. **题面与参考答案在 `problems/<题>/`**,靠 `meta.toml` 的 `problem` 键把同一道题的多语言实现聚起来。
10. **任何改动先 `uv run pytest`**;改了跑 GPU 的部分按 `docs/00-START.md` 的验证矩阵上机验证,**不能上机就明说没验证**。
11. **提交前先看 `git status`** —— 用户随时在面板里写算子,别用 `git add -A` 把他的改动扫进无关的 commit。提交信息用中文,写清改了哪一层。
