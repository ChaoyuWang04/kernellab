"""不需要 GPU 的本地测试:配置、算子契约、工具链登记、体检单渲染、对比表。`uv run pytest`。"""
import ast
from pathlib import Path

from klab import toolchains
from klab.config import load_targets, repo_root
from klab.harness.spec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]


def test_repo_root_and_targets():
    root = repo_root(ROOT)
    cfgs = load_targets(root)
    assert "5090home" in cfgs and cfgs["5090home"].kind == "ssh"
    assert cfgs["modal-h100"].kind == "modal" and cfgs["modal-h100"].gpu == "H100"
    assert cfgs["modal-h100"].extra.get("ncu_clock_control") == "none"
    for c in cfgs.values():
        assert c.kind in ("ssh", "modal", "local")


def test_every_spec_has_valid_meta_contract_and_user_kernel():
    """specs/<名>:meta.toml 能解析、工具链已登记、spec.py 静态定义四个契约函数、对应 kernels/<名> 存在且不含接线文件。"""
    sdirs = sorted(p for p in (ROOT / "specs").iterdir() if (p / "meta.toml").exists())
    assert sdirs, "specs/ 下没有算子接线"
    for sd in sdirs:
        spec = KernelSpec.load(sd)
        assert spec.toolchain in toolchains.REGISTRY, f"{sd.name}: 未登记的工具链 {spec.toolchain}"
        assert spec.cases and all("name" in c for c in spec.cases), f"{sd.name}: case 缺 name"
        assert spec.kernel_regex, f"{sd.name}: 缺 kernel_regex(ncu 过滤用)"
        assert spec.kernel_dir.is_dir(), f"{sd.name}: 缺用户目录 kernels/{sd.name}"
        assert not (spec.kernel_dir / "meta.toml").exists(), f"kernels/{sd.name} 里不该有 meta.toml(接线放 specs/)"
        assert not (spec.kernel_dir / "spec.py").exists(), f"kernels/{sd.name} 里不该有 spec.py(接线放 specs/)"
        tree = ast.parse((sd / "spec.py").read_text())
        fns = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        for fn in ("make_inputs", "run", "reference", "workload"):
            assert fn in fns, f"specs/{sd.name}/spec.py 缺 {fn}()"
        if spec.sweep:
            assert "configure" in fns, f"{sd.name}: 有 [sweep] 但 spec.py 没有 configure()"


def test_every_user_kernel_dir_has_a_spec():
    kdirs = sorted(p.name for p in (ROOT / "kernels").iterdir() if p.is_dir())
    for name in kdirs:
        assert (ROOT / "specs" / name / "meta.toml").exists(), f"kernels/{name} 还没有 specs/{name}/(按 playbook 生成)"


def test_default_target_exists():
    from klab.config import default_target

    assert default_target(ROOT) in load_targets(ROOT)


def test_toolchains_registered_and_have_env_specs():
    for name, mod in toolchains.REGISTRY.items():
        assert hasattr(mod, "setup_script")
        env = ROOT / "envs" / name
        assert (env / "requirements.txt").exists(), f"envs/{name}/requirements.txt 缺失"
        assert (env / "torch-index.txt").exists(), f"envs/{name}/torch-index.txt 缺失"
        script = mod.setup_script("~/klab", "3.12", "~/klab/repo")
        assert "uv venv" in script and f"envs/{name}/requirements.txt" in script


def test_tk_macro_mapping():
    from klab.harness.cppext import tk_macro

    assert tk_macro(12, 0) == "KITTENS_SM120"
    assert tk_macro(9, 0) == "KITTENS_SM90"
    assert tk_macro(10, 0) == "KITTENS_SM100"
    assert tk_macro(8, 9) == "KITTENS_SM80"


def test_vscode_tasks_list_all_targets():
    import json

    tasks = json.loads((ROOT / ".vscode" / "tasks.json").read_text())
    opts = next(i for i in tasks["inputs"] if i["id"] == "target")["options"]
    cfgs = load_targets(ROOT)
    for name, c in cfgs.items():
        if c.kind != "local":
            assert name in opts, f".vscode/tasks.json 的后端下拉缺 {name}"


def test_arch_features_sm120_has_tma_but_not_wgmma():
    """5090(sm_120)实测:TileLang 在它上面发 cp.async.bulk.tensor(TMA),但 mma 仍是 mma.sync。
    cc 数字不是超集关系 —— 数字最大的 sm_120 没有 Hopper 的 wgmma。

    用 AST 读而不是 import:runner.py 在后端跑、顶层 import torch,本地没有。
    """
    tree = ast.parse((ROOT / "klab" / "harness" / "runner.py").read_text())
    node = next(n.value for n in tree.body
                if isinstance(n, ast.Assign)
                and any(getattr(t, "id", "") == "ARCH_FEATURES" for t in n.targets))
    table = {k.value: {e.value for e in v.elts} for k, v in zip(node.keys, node.values)}
    assert "tma" in table[12], "klab ptx 在 5090 上实测到了 cp.async.bulk.tensor"
    assert "wgmma" not in table[12] and "tcgen05" not in table[12]
    assert "wgmma" in table[9] and "tcgen05" in table[10]


def test_override_file_lets_a_solution_stand_in_for_the_users_kernel(monkeypatch):
    """`--solution` 靠 KLAB_KERNEL_FILE 生效。没有它的话,验证参考答案就得把答案拷进
    kernels/<名>/ —— 那是用户的目录,拷进去等于替他改代码,而且改完常常忘了还原。"""
    from klab.harness.spec import override_file

    monkeypatch.delenv("KLAB_KERNEL_FILE", raising=False)
    assert override_file() is None
    monkeypatch.setenv("KLAB_KERNEL_FILE", "/tmp/x/1-atom.py")
    assert override_file() == Path("/tmp/x/1-atom.py")


def test_every_spec_can_name_its_solutions_directory():
    """--solution 按 problems/<题>/solutions/<工具链>/ 找答案;meta.toml 的 problem 与
    toolchain 拼错了,这条路径就永远是空的。"""
    root = Path(__file__).resolve().parents[1]
    for meta in sorted((root / "specs").glob("*/meta.toml")):
        spec = KernelSpec.load(meta.parent)
        d = root / "problems" / spec.problem / "solutions" / spec.toolchain
        assert d.is_dir(), f"{meta.parent.name}: {d.relative_to(root)} 不存在"
        assert list(d.glob("*.py")) or list(d.glob("*.cu")), f"{d.relative_to(root)} 里没有答案"
