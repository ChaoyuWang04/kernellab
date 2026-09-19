import json
import shutil
from pathlib import Path

from klab.compare import collect
from klab.config import load_targets
from klab.kreport import _tail_wave, load_raw, render, verdict

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "softmax-5090home-ncu"


def test_raw_csv_parses_and_key_metrics_present():
    rows = load_raw(FIX / "ncu-raw.csv")
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "launch__registers_per_thread",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ):
        assert key in row, f"体检单依赖的指标 {key} 不在 raw CSV 里(NCU section 集合变了?)"


def test_verdict_is_memory_bound_for_softmax_fixture():
    """softmax 是逐元素算子,该判成搬数据受限;措辞用人话,不出现 NCU 术语。"""
    v = verdict(load_raw(FIX / "ncu-raw.csv")[0])
    assert "卡在搬数据上" in v and "显存" in v
    assert "受限" not in v and "throughput" not in v


def test_render_report_markdown_has_all_sections(tmp_path):
    run = tmp_path / "20260918-000000-softmax-5090home-ncu"
    shutil.copytree(FIX, run)
    cfg = load_targets(ROOT)["5090home"]
    md = render(run, cfg, bench=None)
    for section in ("判定", "### 跑多快", "### 哪个部件忙", "### 卡子怎么切的", "### warp 在等什么"):
        assert section in md
    assert "显存" in md and "GB/s" in md
    assert "NCU 建议" not in md, "NCU 英文原文摘录已去掉"
    for jargon in ("Speed of Light", "Est. Speedup", "pct_of_peak", "Tensor pipe", "stall"):
        assert jargon not in md, f"体检单不该出现术语 {jargon!r}"
    assert "1 GB/s" not in md  # 5090 的 Tbyte/s 单位要被换算成 GB/s


def test_compare_collects_latest_per_kernel_target(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "targets.toml").write_text((ROOT / "targets.toml").read_text())

    def mk(ts, kernel, target, ms):
        d = runs / f"{ts}-{kernel}-{target}-bench"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({
            "kernel": kernel, "toolchain": "triton", "mode": "bench",
            "device": {"device": "X", "cc": "9.0"},
            "cases": [{"name": "c", "dtype": "float16"}],
            "results": [{"case": "c", "median_ms": ms, "tflops": 1.0, "gbps": 1.0, "ref_median_ms": 1.0, "speedup_vs_ref": 1.0 / ms}],
        }))

    mk("20260918-100000", "matmul", "5090home", 2.0)
    mk("20260918-110000", "matmul", "5090home", 1.0)   # 更新的一次
    mk("20260918-100000", "matmul", "modal-h100", 0.5)
    rows = collect(tmp_path, ["matmul"], None, latest_only=True)
    assert {(r["target"], r["median_ms"]) for r in rows} == {("5090home", 1.0), ("modal-h100", 0.5)}
    assert len(collect(tmp_path, ["matmul"], None, latest_only=False)) == 3


def test_tail_wave_waste_is_worst_just_above_an_integer():
    """3.01 波 = 要跑 4 波、最后一波只有 1% 的位置在用(NCU 给这一条的 Est. Speedup 是 25%);
    3.98 波反而几乎没浪费。小数部分小 = 浪费大,不能按「接近整数」来判。"""
    bad = _tail_wave(1024, 3.01, 170)
    assert "25%" in bad and "1%" in bad and "没有浪费" not in bad

    good = _tail_wave(1024, 3.98, 170)
    assert "**" not in good, "浪费很小就不该加粗告警"
    assert "没有浪费" in good

    exact = _tail_wave(1020, 3.0, 170)
    assert "没有浪费" in exact

    assert _tail_wave(None, 3.01, 170) == "-" and _tail_wave(1024, 0, 170) == "-"
