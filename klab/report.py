"""把 harness 的 JSON 结果在 Mac 上打印成表,并按 targets.toml 的峰值算 roofline 百分比。"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from klab.config import TargetConfig

console = Console()


def _peak_for_dtype(cfg: TargetConfig, dtype: str) -> float | None:
    if dtype in ("float16", "bfloat16"):
        return cfg.peaks.get("peak_tflops_fp16")
    if dtype == "float32":
        return cfg.peaks.get("peak_tflops_fp32")
    return None


def print_result(json_path: Path, cfg: TargetConfig) -> bool:
    data = json.loads(json_path.read_text())
    dev = data["device"]
    console.print(
        f"[bold]{data['kernel']}[/] · {data['mode']} · {dev['device']} (cc {dev['cc']}) · "
        f"torch {dev.get('torch')} · triton {dev.get('triton', '-')} · {data['elapsed_s']}s"
    )
    ok = True
    if data["mode"] == "check":
        t = Table("case", "result", "max_abs_err", "mean_abs_err", "shape", "dtype")
        for r in data["results"]:
            ok &= r["ok"]
            t.add_row(
                r["case"], "[green]PASS[/]" if r["ok"] else "[red]FAIL[/]",
                f"{r['max_abs_err']:.3e}", f"{r['mean_abs_err']:.3e}", "×".join(map(str, r["shape"])), r["dtype"],
            )
        console.print(t)
    elif data["mode"] == "bench":
        peak_bw = cfg.peaks.get("peak_gbps")
        t = Table("case", "median ms", "p10", "p90", "GB/s", "%BW", "TFLOPS", "%FP")
        cases = {c["name"]: c for c in data.get("cases", [])}
        for r in data["results"]:
            dtype = cases.get(r["case"], {}).get("dtype", "float32")
            peak_fp = _peak_for_dtype(cfg, dtype)
            t.add_row(
                r["case"], f"{r['median_ms']:.4f}", f"{r['p10_ms']:.4f}", f"{r['p90_ms']:.4f}",
                f"{r['gbps']:.1f}", f"{100 * r['gbps'] / peak_bw:.0f}%" if peak_bw else "-",
                f"{r['tflops']:.2f}", f"{100 * r['tflops'] / peak_fp:.1f}%" if peak_fp else "-",
            )
        console.print(t)
        if peak_bw:
            console.print(f"[dim]%BW 按 targets.toml 的 peak_gbps={peak_bw} 计;跑一次 klab probe 可得实测值[/]")
    else:
        for r in data["results"]:
            console.print(f"  {r['case']}: launched x{r['launches']}")
    return ok
