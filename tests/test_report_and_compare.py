import json
import shutil
from pathlib import Path

from klab.compare import collect
from klab.config import load_targets
from klab.kreport import load_advisories, load_raw, render, verdict

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
    row = load_raw(FIX / "ncu-raw.csv")[0]
    assert "内存侧受限" in verdict(row)


def test_advisories_extracted():
    adv = load_advisories(FIX / "ncu-details.txt")
    assert adv and all(len(a) > 20 for a in adv)


def test_render_report_markdown_has_all_sections(tmp_path):
    run = tmp_path / "20260918-000000-softmax-5090home-ncu"
    shutil.copytree(FIX, run)
    cfg = load_targets(ROOT)["5090home"]
    md = render(run, cfg, bench=None)
    for section in ("判定", "### 速度", "### 各单元利用率", "### 发射与占用", "### 调度与 stall", "### NCU 建议"):
        assert section in md
    assert "DRAM" in md and "GB/s" in md
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
            "results": [{"case": "c", "median_ms": ms, "tflops": 1.0, "gbps": 1.0}],
        }))

    mk("20260918-100000", "matmul", "5090home", 2.0)
    mk("20260918-110000", "matmul", "5090home", 1.0)   # 更新的一次
    mk("20260918-100000", "matmul", "modal-h100", 0.5)
    rows = collect(tmp_path, ["matmul"], None, latest_only=True)
    assert {(r["target"], r["median_ms"]) for r in rows} == {("5090home", 1.0), ("modal-h100", 0.5)}
    assert len(collect(tmp_path, ["matmul"], None, latest_only=False)) == 3
