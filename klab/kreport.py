"""算子体检单:把 bench 的 result.json 与 NCU 的 raw CSV / details 文本压成一张固定模板。

NCU 信息全但慢,体检单只回答写算子的人最先要问的几件事:
  这次跑多快、离峰值多远、卡在哪个单元、发射配置有没有明显问题、NCU 自己给了什么建议。
模板固定,任何算子、任何后端都长一样,便于横着比。
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from klab.config import TargetConfig

# ---------- 读取 ----------


def load_raw(csv_path: Path) -> list[dict[str, tuple[str, str]]]:
    """ncu --page raw --csv:第一行指标名,第二行单位,之后每行一个 kernel 启动。"""
    rows = list(csv.reader(csv_path.open()))
    if len(rows) < 3:
        return []
    hdr, units = rows[0], rows[1]
    return [{h: (u, v) for h, u, v in zip(hdr, units, r)} for r in rows[2:]]


def load_advisories(details_txt: Path) -> list[str]:
    """把 details 页里的 OPT 段落(含 Est. Speedup)抓成一段一条。"""
    out, cur = [], None
    for line in details_txt.read_text(errors="replace").splitlines():
        if re.match(r"^\s*OPT\s", line):
            if cur:
                out.append(" ".join(cur))
            cur = [line.strip()[3:].strip()]
        elif cur is not None:
            if line.strip() == "" or re.match(r"^\s*(Section:|INF|WRN|-{5,}|[A-Za-z].*?\s{2,}\S+\s{2,}\S+$)", line):
                out.append(" ".join(cur))
                cur = None
            else:
                cur.append(line.strip())
    if cur:
        out.append(" ".join(cur))
    # 去掉同一 kernel 重复的段落,保留顺序
    seen, uniq = set(), []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


# ---------- 取值 ----------


def _num(row: dict, key: str) -> float | None:
    if key not in row:
        return None
    v = row[key][1].replace(",", "")
    try:
        return float(v)
    except ValueError:
        return None


def _unit(row: dict, key: str) -> str:
    return row.get(key, ("", ""))[0]


def _fmt(v: float | None, spec: str = ".1f", suffix: str = "") -> str:
    return "-" if v is None else f"{v:{spec}}{suffix}"


def _pct(v: float | None) -> str:
    return _fmt(v, ".0f", "%")


def _to_us(row: dict, key: str) -> float | None:
    v = _num(row, key)
    if v is None:
        return None
    u = _unit(row, key)
    return {"ns": v / 1e3, "us": v, "usecond": v, "ms": v * 1e3, "msecond": v * 1e3, "s": v * 1e6}.get(u, v)


def _gbps(row: dict, key: str) -> float | None:
    """带宽统一成 GB/s(5090 上 ncu 会用 Tbyte/s 报)。"""
    v = _num(row, key)
    if v is None:
        return None
    u = _unit(row, key)
    return {"Tbyte/s": v * 1000, "Gbyte/s": v, "Mbyte/s": v / 1000, "byte/s": v / 1e9}.get(u, v)


def _kb(row: dict, key: str) -> float | None:
    v = _num(row, key)
    if v is None:
        return None
    u = _unit(row, key)
    if u.startswith("byte"):
        return v / 1024
    if u.startswith("Mbyte"):
        return v * 1024
    return v


def stall_reasons(row: dict, top: int = 4) -> list[tuple[str, float]]:
    pat = re.compile(r"^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$")
    found = []
    for k in row:
        m = pat.match(k)
        if m:
            v = _num(row, k)
            if v is not None and m.group(1) not in ("not_selected", "selected"):
                found.append((m.group(1), v))
    found.sort(key=lambda x: -x[1])
    return found[:top]


# ---------- 判定 ----------


def verdict(row: dict) -> str:
    sm = _num(row, "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    mem = _num(row, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    dram = _num(row, "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
    l2 = _num(row, "lts__throughput.avg.pct_of_peak_sustained_elapsed")
    l1 = _num(row, "l1tex__throughput.avg.pct_of_peak_sustained_active")
    tensor = _num(row, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
    occ = _num(row, "sm__warps_active.avg.pct_of_peak_sustained_active")
    if sm is None or mem is None:
        return "NCU 没有采到 Speed of Light 指标,无法判定"
    parts = []
    hi = max(sm, mem)
    if hi >= 80:
        if mem >= sm:
            unit = max(
                [("DRAM", dram or 0), ("L2", l2 or 0), ("L1/TEX", l1 or 0)], key=lambda x: x[1]
            )
            parts.append(f"**内存侧受限**(内存总利用 {mem:.0f}%,最忙的是 {unit[0]} {unit[1]:.0f}%)")
        else:
            t = f",Tensor pipe {tensor:.0f}%" if tensor else ""
            parts.append(f"**计算侧受限**(SM {sm:.0f}%{t})")
    elif hi < 60:
        parts.append(f"**延迟受限**:计算 {sm:.0f}% 与内存 {mem:.0f}% 都没跑满,先看占用率与 stall 原因")
    elif abs(sm - mem) >= 15:
        side = "计算" if sm > mem else "内存"
        parts.append(f"**偏{side}侧、尚未撞墙**(计算 {sm:.0f}%、内存 {mem:.0f}%,都没到 80%):先看占用率与 stall 原因")
    else:
        parts.append(f"**接近均衡**:计算 {sm:.0f}%、内存 {mem:.0f}%,再往上要同时动两边")
    if occ is not None:
        parts.append(f"实测占用率 {occ:.0f}%")
    return ";".join(parts)


# ---------- 渲染 ----------


def render(run_dir: Path, cfg: TargetConfig, bench: dict | None = None) -> str:
    result = json.loads((run_dir / "result.json").read_text())
    rows = load_raw(run_dir / "ncu-raw.csv") if (run_dir / "ncu-raw.csv").exists() else []
    advisories = load_advisories(run_dir / "ncu-details.txt") if (run_dir / "ncu-details.txt").exists() else []
    dev = result["device"]
    cases = result.get("results", [])
    launches = max(int(c.get("launches", 1)) for c in cases) if cases else 1
    bench_by_case = {r["case"]: r for r in (bench or {}).get("results", [])}
    case_meta = {c["name"]: c for c in (result.get("cases") or (bench or {}).get("cases") or [])}

    md = [f"# 算子体检单 · {result['kernel']}", ""]
    md.append(f"后端 `{cfg.name}` · {dev['device']} (cc {dev['cc']}, {dev['sm_count']} SM) · torch {dev.get('torch')} · triton {dev.get('triton', '-')} · 运行 `{run_dir.name}`")
    md.append("")
    if not rows:
        md.append("**没有 NCU 数据**(ncu-raw.csv 缺失或为空)。")
        return "\n".join(md)

    peak_bw = cfg.peaks.get("peak_gbps")
    for i, c in enumerate(cases):
        idx = min(i * launches + launches - 1, len(rows) - 1)
        row = rows[idx]
        name = c["case"]
        dtype = case_meta.get(name, {}).get("dtype", "float32")
        peak_fp = cfg.peaks.get("peak_tflops_fp16") if dtype in ("float16", "bfloat16") else cfg.peaks.get("peak_tflops_fp32")
        b = bench_by_case.get(name)

        md.append(f"## case `{name}`")
        md.append("")
        md.append(f"**判定**:{verdict(row)}")
        md.append("")

        # 速度
        dur = _to_us(row, "gpu__time_duration.sum")
        md.append("### 速度")
        md.append("")
        md.append("| 项 | 值 | 对峰值 |")
        md.append("|---|---|---|")
        md.append(f"| NCU 内核时长 | {_fmt(dur, '.1f', ' us')} | 锁频 {cfg.extra.get('ncu_clock_control', 'base')} |")
        if b:
            md.append(f"| bench 中位数 | {b['median_ms']:.4f} ms(p10 {b['p10_ms']:.4f} / p90 {b['p90_ms']:.4f},{b['iters']} 次) | |")
            fp_pct = f"{100 * b['tflops'] / peak_fp:.0f}% of {peak_fp:.0f}" if peak_fp else "-"
            bw_pct = f"{100 * b['gbps'] / peak_bw:.0f}% of {peak_bw:.0f}" if peak_bw else "-"
            if b.get("ref_median_ms"):
                x = b["speedup_vs_ref"]
                tag = "快于 torch" if x >= 1.0 else ("接近 torch" if x >= 0.85 else "慢于 torch")
                md.append(f"| **相对 torch 参考** | **{x:.2f}×**({tag}) | torch 参考 {b['ref_median_ms']:.4f} ms,多半是 cuBLAS / torch 融合 kernel |")
            md.append(f"| 算力 | {b['tflops']:.1f} TFLOPS | {fp_pct} |")
            md.append(f"| 带宽 | {b['gbps']:.1f} GB/s | {bw_pct} |")
            if b.get("bytes") and peak_fp and peak_bw:
                ai = b["flops"] / b["bytes"]
                ridge = peak_fp * 1e12 / (peak_bw * 1e9)
                side = "计算侧" if ai > ridge else "带宽侧"
                md.append(f"| 算术强度 | {ai:.0f} flop/B | roofline 拐点 {ridge:.0f} flop/B → 理论上在{side} |")
        else:
            md.append("| bench | 没找到同 case 的 bench 结果,先跑一次 `klab bench` | |")
        md.append("")

        # 瓶颈定位
        md.append("### 各单元利用率(NCU Speed of Light,% of peak)")
        md.append("")
        md.append("| 单元 | 利用率 | 备注 |")
        md.append("|---|---|---|")
        md.append(f"| SM(计算总) | {_pct(_num(row, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'))} | 发射槽忙 {_pct(_num(row, 'sm__issue_active.avg.pct_of_peak_sustained_elapsed'))},IPC {_fmt(_num(row, 'sm__inst_executed.avg.per_cycle_active'), '.2f')} |")
        md.append(f"| Tensor pipe | {_pct(_num(row, 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'))} | hmma {_pct(_num(row, 'sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed'))},gmma {_pct(_num(row, 'sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_elapsed'))} 按指令数 |")
        md.append(f"| FMA / ALU / LSU pipe | {_pct(_num(row, 'sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed'))} / {_pct(_num(row, 'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed'))} / {_pct(_num(row, 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed'))} | 按指令数 |")
        md.append(f"| 内存(总) | {_pct(_num(row, 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'))} | 取 DRAM / L2 / L1 中最忙者 |")
        md.append(f"| DRAM | {_pct(_num(row, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'))} | {_fmt(_gbps(row, 'dram__bytes.sum.per_second'), '.0f', ' GB/s')} |")
        md.append(f"| L2 | {_pct(_num(row, 'lts__throughput.avg.pct_of_peak_sustained_elapsed'))} | 命中率 {_pct(_num(row, 'lts__t_sector_hit_rate.pct'))} |")
        md.append(f"| L1/TEX | {_pct(_num(row, 'l1tex__throughput.avg.pct_of_peak_sustained_active'))} | 命中率 {_pct(_num(row, 'l1tex__t_sector_hit_rate.pct'))} |")
        md.append("")

        # 发射与占用
        grid = _num(row, "launch__grid_size")
        block = _num(row, "launch__block_size")
        regs = _num(row, "launch__registers_per_thread")
        smem = _kb(row, "launch__shared_mem_per_block")
        waves = _num(row, "launch__waves_per_multiprocessor")
        occ_t = _num(row, "sm__maximum_warps_per_active_cycle_pct")
        occ_a = _num(row, "sm__warps_active.avg.pct_of_peak_sustained_active")
        limits = {
            "寄存器": _num(row, "launch__occupancy_limit_registers"),
            "共享内存": _num(row, "launch__occupancy_limit_shared_mem"),
            "warp 数": _num(row, "launch__occupancy_limit_warps"),
            "block 数": _num(row, "launch__occupancy_limit_blocks"),
        }
        valid = {k: v for k, v in limits.items() if v}  # 0 是 ncu 对超出默认 carveout 的动态共享内存的占位值,不当限制因子
        limiter = min(valid, key=valid.get) if valid else "-"
        md.append("### 发射与占用")
        md.append("")
        md.append("| 项 | 值 | 说明 |")
        md.append("|---|---|---|")
        md.append(f"| grid × block | {_fmt(grid, '.0f')} × {_fmt(block, '.0f')} | {_fmt(waves, '.2f')} 波/SM |")
        md.append(f"| 寄存器 / 线程 | {_fmt(regs, '.0f')} | 允许 {_fmt(limits['寄存器'], '.0f')} block/SM |")
        md.append(f"| 共享内存 / block | {_fmt(smem, '.1f', ' KB')} | 允许 {_fmt(limits['共享内存'], '.0f')} block/SM |")
        md.append(f"| 占用率 理论 / 实测 | {_pct(occ_t)} / {_pct(occ_a)} | 限制因子:{limiter}(允许 {_fmt(valid.get(limiter), '.0f')} block/SM) |")
        spill = _num(row, "derived__local_spilling_requests")
        md.append(f"| 本地内存溢出请求 | {_fmt(spill, '.0f')} | 非 0 说明寄存器不够、已溢出到 local |")
        md.append("")

        # 调度与 stall
        st = stall_reasons(row)
        elig = _num(row, "smsp__warps_eligible.avg.per_cycle_active")
        act = _num(row, "smsp__warps_active.avg.per_cycle_active")
        md.append("### 调度与 stall")
        md.append("")
        md.append("| 项 | 值 |")
        md.append("|---|---|")
        md.append(f"| 每调度器 活跃 / 可发射 warp | {_fmt(act, '.2f')} / {_fmt(elig, '.2f')} |")
        if st:
            md.append("| 主要 stall(每次发射平均等待的 warp 数) | " + ",".join(f"{k} {v:.2f}" for k, v in st) + " |")
        else:
            md.append("| 主要 stall | 未采集(需要 WarpStateStats section) |")
        md.append("")

        # NCU 建议
        md.append("### NCU 建议(原文摘录)")
        md.append("")
        if advisories:
            for a in advisories[:5]:
                md.append(f"- {a[:420] + (' …' if len(a) > 420 else '')}")
        else:
            md.append("- (无)")
        md.append("")
    return "\n".join(md)


def find_latest_bench(root: Path, kernel: str, target: str) -> dict | None:
    cands = sorted((root / "runs").glob(f"*-{kernel}-{target}-bench/result.json"))
    for p in reversed(cands):
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    return None
