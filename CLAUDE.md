# AI 维护入口

这是一个「Mac 上写 GPU 算子、远端 GPU 上跑、性能指标回流」的个人实验台。**整个项目面向 agent 调用**:用户只在 `kernels/<名>/` 里写算子,其余(接线、跑、读 NCU、评价)都是 agent 的活。

## 开始前必读

1. 先读 **[docs/01-AGENT-PLAYBOOK.md](docs/01-AGENT-PLAYBOOK.md)**:用户说「测一下这个 kernel」时该做什么,按什么顺序,怎么读结果,怎么评价。这是默认行为。
2. 再读 **[docs/00-START.md](docs/00-START.md)**:系统怎么运转、改哪里、怎么验证、踩过的坑。改代码前必读。
3. 使用层面看 [README.md](README.md):命令、契约、体检单读法。
4. 改动前 `git status`,确认没有别人的未提交工作。

## 与用户的协作方式

- 用户在 VSCode 里写 `kernels/<名>/`,然后一句话「测一下 X」;之后的接线、选后端、跑、读体检单、评价全由 agent 做完再汇报,**中途不要逐步请示**。需要用户拍板的只有:改用户的算子源码、放宽容差、换到他没点名的后端之外的付费资源、删东西。
- 汇报用紧凑表格,先给判定再给数字;评价落到源码的行或常量;不复述 NCU 全部指标。
- 用户常用语音输入,术语可能被识别错:Modal → "model"、TileLang → "Triton-TEL"、ThunderKittens → "thunder / kittens"、CuTe DSL → "long CUTE"。按上下文理解,不要按字面追问。
- 有疑问先看 `runs/<id>/report.md` 与源码再问;问的时候给选项并附推荐。

## 不可破坏的约束

- **`kernels/<名>/` 是用户的**,只有算子源码;不改、不加文件,除非用户要求。接线一律放 `specs/<名>/`。
- Target 接口只有 `sync / run / fetch / shell` 四个方法;编译、测速、NCU 等一律是 `run()` 之上的命令串,不进后端类。
- `spec.py` 契约只有 `make_inputs / run / reference / workload`(可选 `configure`),harness 不认识任何 DSL。`reference()` 用与算子相同 dtype 的原生 torch 调用。
- `klab/harness/` 在后端运行,只能依赖 torch 与标准库。
- 仓库里不放任何凭据;Modal 读 `~/.modal.toml`,SSH 读 `~/.ssh/config`。
- `kernels/*/_vendor/` 是第三方原文,不改。
- 不做 `--target auto`(用户搁置),不引入数据库,不引入 web 框架。
- `klab web` 是本地面板(只绑 127.0.0.1),对标 LeetCode:左题面/讲解/提交记录/体检单,右 Monaco 编辑器。**用户现在在网页里写算子,不再用 VSCode。**
- 面板只读 `runs/` 与仓库源码、只 fork `klab` 子进程,不做多用户。「文件即数据」不变 —— 面板不许有自己的持久化状态,不许引入 web 框架或构建步骤。
- 题面与优化路线在 `problems/<题>/`(agent 写);`meta.toml` 的 `problem` 键把同一道题的多语言实现聚在一起。
- `kernels/<名>/` 仍然只放算子源码,但现在是网页编辑器在写它;agent 照旧不擅自改,除非用户要求。
- 任何改动先 `uv run pytest`;改了跑 GPU 的部分按 `docs/00-START.md` 第五节的验证矩阵上机验证;不能上机就明说没验证。
