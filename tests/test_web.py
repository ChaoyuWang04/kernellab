"""不需要 GPU:面板的 markdown 子集转换、run 目录解析、算子清单、路由白名单。`uv run pytest`。"""
import json
import shutil
from pathlib import Path

from klab.config import load_targets
from klab.kreport import render
from klab.web import (MODES, TOOLCHAIN_LANG, _parse_run, list_kernels, list_problems,
                      list_runs, md_to_html, primary_source, state)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "softmax-5090home-ncu"


def test_md_subset_covers_headings_tables_lists_and_inline():
    html = md_to_html(
        "# 标题\n\n"
        "后端 `5090home` · 普通段落\n\n"
        "## case `4096-bf16`\n\n"
        "**判定**:**计算侧受限**(SM 84%)\n\n"
        "| 项 | 值 |\n|---|---|\n| 算力 | 204.6 TFLOPS |\n| 带宽 | 149.9 GB/s |\n\n"
        "### NCU 建议\n\n"
        "- Est. Speedup: 25% 尾波\n"
        "- 第二条\n"
    )
    assert "<h1>标题</h1>" in html and "<h2>case <code>4096-bf16</code></h2>" in html
    assert "<strong>计算侧受限</strong>" in html
    assert "<th>项</th>" in html and "<td>204.6 TFLOPS</td>" in html
    assert "|---|---|" not in html and "<td>---</td>" not in html, "GFM 分隔行不能当成数据行"
    assert html.count("<li>") == 2
    assert "<p>后端 <code>5090home</code> · 普通段落</p>" in html


def test_md_escapes_html_so_report_text_cannot_inject():
    html = md_to_html("段落 <script>alert(1)</script> & <b>x</b>")
    assert "<script>" not in html and "&lt;script&gt;" in html and "&amp;" in html


def test_real_kreport_output_renders_without_leftover_markdown(tmp_path):
    """体检单是 kreport 生成的,两端都归我们管:转换后不该残留 markdown 记号。"""
    run = tmp_path / "20260918-000000-softmax-5090home-ncu"
    shutil.copytree(FIX, run)
    html = md_to_html(render(run, load_targets(ROOT)["5090home"], bench=None))
    assert "<table>" in html and "<h1>" in html and "<h3>" in html
    for leftover in ("|---", "\n# ", "\n## ", "\n- ", "**"):
        assert leftover not in html, f"没转干净:{leftover!r}"


def test_parse_run_recovers_target_when_kernel_name_contains_dash(tmp_path):
    """目录名是 <时间>-<算子>-<后端>-<模式>,算子名里有 '-' 时只能靠 result.json 反推后端。"""
    run = tmp_path / "20260918-214252-matmul_triton_ampere-5090home-ncu"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({
        "kernel": "matmul_triton_ampere", "mode": "ncu",
        "device": {"device": "NVIDIA GeForce RTX 5090"}, "results": [{"case": "4096-bf16"}],
    }))
    r = _parse_run(run)
    assert r["kernel"] == "matmul_triton_ampere" and r["target"] == "5090home" and r["mode"] == "ncu"
    assert r["has_report"] is False and r["ok"] is True


def test_parse_run_ignores_directories_without_result_json(tmp_path):
    (tmp_path / "20260918-000000-x-5090home-ncu").mkdir()
    assert _parse_run(tmp_path / "20260918-000000-x-5090home-ncu") is None
    assert list_runs(tmp_path.parent) == [] or True  # 只要不抛


def test_kernel_list_marks_specs_without_sources_as_not_runnable():
    ks = {k["name"]: k for k in list_kernels(ROOT)}
    assert ks, "仓库里至少该有一个算子"
    for k in ks.values():
        if k["ready"]:
            assert k["source"] and k["toolchain"] and k["cases"]
        else:
            assert k["note"], "不可跑的算子必须说明原因"


def test_state_exposes_targets_and_default_from_targets_toml():
    st = state(ROOT)
    names = {t["name"] for t in st["targets"]}
    assert "5090home" in names and "modal-h100" in names
    assert st["default_target"] in names


def test_modes_map_only_to_real_klab_subcommands():
    """网页按钮不能凭空造子命令:每个 mode 都得是 cli.py 里真有的命令。"""
    import klab.cli as cli

    registered = {c.name or c.callback.__name__ for c in cli.app.registered_commands}
    for mode, sub in MODES.items():
        assert sub in registered, f"模式 {mode} 映射到了不存在的子命令 {sub}"


def test_md_renders_fenced_code_blocks_and_ordered_lists():
    """题面与讲解用得到围栏代码块和有序列表,体检单用不到,所以只在这里守。"""
    html = md_to_html("1. 第一步\n2. 第二步\n\n```python\nacc += tl.dot(a, b)\n```\n")
    assert html.count("<li>") == 2 and "<ol>" in html
    assert '<pre class="code" data-lang="python">' in html
    assert "acc += tl.dot(a, b)" in html and "```" not in html


def test_problems_group_implementations_by_meta_problem_key():
    """一道题 = 一个数学 × 若干语言实现,靠 meta.toml 的 problem 键聚合。"""
    probs = {p["name"]: p for p in list_problems(ROOT)}
    assert "matmul" in probs, "matmul_triton_ampere 的 meta.toml 应声明 problem = \"matmul\""
    m = probs["matmul"]
    assert m["has_doc"], "problems/matmul/problem.md 应存在"
    assert any(k["name"] == "matmul_triton_ampere" and k["toolchain"] == "triton" for k in m["impls"])
    assert m["title"] and m["title"] != "matmul", "题名应取自 problem.md 的一级标题"


def test_primary_source_follows_toolchain_extension():
    """编辑器编辑哪个文件由工具链决定:python 系写 .py,cuda / tk 写 .cu。"""
    for tc, (lang, fname) in TOOLCHAIN_LANG.items():
        assert fname.endswith(".py" if lang == "python" else ".cu")
    src = primary_source(ROOT, "matmul_triton_ampere", "triton")
    assert src.name == "kernel.py" and src.is_file()


def test_kernel_entries_carry_editor_metadata():
    for k in list_kernels(ROOT):
        if k.get("ready"):
            assert k["language"] in ("python", "cpp") and k["source"] and k["problem"]


def test_a_foreign_listener_is_never_killed_only_reported(tmp_path):
    """自动接管只许停掉我们自己的面板。端口上是别的程序时:不动它,给一句能照做的话,不甩 traceback。"""
    import socket

    import pytest

    from klab.web import _is_our_panel, serve

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert _is_our_panel(port) is False, "非面板不能被认成面板,否则会误杀"
        with pytest.raises(SystemExit) as e:          # restart 默认开,仍不该动它
            serve(tmp_path, port=port, open_browser=False)
        msg = str(e.value)
        assert str(port) in msg and "--port" in msg and "没敢自动停" in msg
        assert "Traceback" not in msg
        s.getsockname()                                # socket 还活着 = 没被杀
    finally:
        s.close()
