"""跨算子、跨后端的对比表:扫 runs/ 里的 bench 结果,每个 (算子, 后端) 取最新一次。"""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from klab.config import TargetConfig

console = Console()


def collect(root: Path, kernels: list[str] | None, targets: list[str] | None, latest_only: bool = True) -> list[dict]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for p in sorted((root / "runs").glob("*-bench/result.json"), reverse=True):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        run = p.parent.name
        # 目录名 <时间>-<算子>-<后端>-bench;算子名与后端名都可能含 '-',用 result.json 里的 kernel 反推后端
        kernel = d["kernel"]
        mid = run[len("20000101-000000-"):-len("-bench")]
        if not mid.startswith(kernel + "-"):
            continue
        target = mid[len(kernel) + 1:]
        if kernels and kernel not in kernels:
            continue
        if targets and target not in targets:
            continue
        if latest_only and (kernel, target) in seen:
            continue
        seen.add((kernel, target))
        cases = {c["name"]: c for c in d.get("cases", [])}
        for r in d["results"]:
            rows.append({
                "kernel": kernel, "toolchain": d.get("toolchain", "-"), "target": target,
                "device": d["device"]["device"], "case": r["case"],
                "dtype": cases.get(r["case"], {}).get("dtype", "float32"),
                "median_ms": r["median_ms"], "tflops": r["tflops"], "gbps": r["gbps"], "run": run,
            })
    rows.sort(key=lambda r: (r["case"], r["kernel"], r["target"]))
    return rows


def print_table(rows: list[dict], cfgs: dict[str, TargetConfig]) -> None:
    if not rows:
        console.print("runs/ 里没有匹配的 bench 结果")
        return
    t = Table("case", "kernel", "target", "median ms", "TFLOPS", "%FP", "GB/s", "%BW", "date")
    for r in rows:
        cfg = cfgs.get(r["target"])
        peak_fp = peak_bw = None
        if cfg:
            peak_fp = cfg.peaks.get("peak_tflops_fp16") if r["dtype"] in ("float16", "bfloat16") else cfg.peaks.get("peak_tflops_fp32")
            peak_bw = cfg.peaks.get("peak_gbps")
        t.add_row(
            r["case"], f"{r['kernel']} ({r['toolchain']})", r["target"],
            f"{r['median_ms']:.4f}", f"{r['tflops']:.1f}", f"{100 * r['tflops'] / peak_fp:.0f}%" if peak_fp else "-",
            f"{r['gbps']:.0f}", f"{100 * r['gbps'] / peak_bw:.0f}%" if peak_bw else "-", r["run"][:8],
        )
    console.print(t)
