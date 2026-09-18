# AI 维护入口

这是一个「Mac 上写 GPU 算子、远端 GPU 上跑、性能指标回流」的个人实验台。

## 开始前必读

1. 先读 **[docs/00-START.md](docs/00-START.md)**:为什么有这个项目、设计原则、目录职责、功能清单、验证矩阵、踩过的坑、未接入平台怎么接。
2. 使用层面看 [README.md](README.md):命令、算子契约、体检单读法。
3. 改动前 `git status`,确认没有别人的未提交工作。

## 不可破坏的约束

- Target 接口只有 `sync / run / fetch / shell` 四个方法;编译、测速、NCU 等一律是 `run()` 之上的命令串,不进后端类。
- `kernel.py` 契约只有 `make_inputs / run / reference / workload`(可选 `configure`),harness 不认识任何 DSL。
- `klab/harness/` 在后端运行,只能依赖 torch 与标准库。
- 仓库里不放任何凭据;Modal 读 `~/.modal.toml`,SSH 读 `~/.ssh/config`。
- `kernels/*/_vendor/` 是第三方原文,不改。
- 不做 `--target auto`(用户搁置),不做网页,不引入数据库。
- 任何改动先 `uv run pytest`;改了跑 GPU 的部分按 `docs/00-START.md` 第六节的矩阵上机验证;不能上机就明说没验证。
