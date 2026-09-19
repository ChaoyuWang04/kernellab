"""算子体检单:把 bench 的 result.json 与 NCU 的 raw CSV / details 文本压成一张固定模板。

NCU 信息全但慢,体检单只回答写算子的人最先要问的几件事:
  跑多快、离这张卡的上限多远、哪个部件忙哪个闲、卡子切得对不对、warp 在等什么。
措辞一律用人话,不留 NCU 的英文术语(原始指标名在括号里给,方便回查)。
模板固定,任何算子、任何后端都长一样,便于横着比。
"""
from __future__ import annotations

import csv
import json
import math
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


# warp 停下来在等什么。NCU 的英文名照搬没人看得懂,这里给一句人话。
STALL_ZH = {
    "math_pipe_throttle": "等计算单元排队(计算密集时的正常现象)",
    "long_scoreboard": "等显存把数据取回来",
    "short_scoreboard": "等共享内存或特殊功能单元",
    "barrier": "等其他线程到达同步点",
    "membar": "等内存屏障",
    "mio_throttle": "访存指令排不进队",
    "lg_throttle": "全局/局部访存指令排不进队",
    "tex_throttle": "访存单元排不进队",
    "imc_miss": "等常量内存",
    "wait": "等上一条指令的结果",
    "dispatch_stall": "调度器发不出去",
    "no_instruction": "等指令本身被取回",
    "branch_resolving": "等分支决定往哪走",
    "drain": "收尾,在排空",
    "sleeping": "线程在睡",
    "misc": "其他",
}


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
    hi = max(sm, mem)
    if hi >= 80:
        if mem >= sm:
            unit, uv = max([("显存", dram or 0), ("L2 缓存", l2 or 0), ("L1 缓存", l1 or 0)], key=lambda x: x[1])
            head = (f"**卡在搬数据上**:内存通道忙到 {mem:.0f}%,最挤的是{unit}({uv:.0f}%)。"
                    f"再快只能少搬字节 —— 融合、复用、换低精度存")
        else:
            t = f",其中 tensor core {tensor:.0f}%" if tensor else ""
            head = (f"**卡在算上**:计算单元忙到 {sm:.0f}%{t}。"
                    f"{'这是好状态,矩阵单元基本喂饱了' if (tensor or 0) >= 80 else '但 tensor core 没吃满,先看是不是走了标量指令'}")
    elif hi < 60:
        head = (f"**两头都没跑满,时间花在等上**:计算才 {sm:.0f}%、内存才 {mem:.0f}%。"
                f"往下看占用率和「warp 在等什么」")
    elif abs(sm - mem) >= 15:
        side = "算" if sm > mem else "搬数据"
        head = f"**偏{side}这一侧,但都还没撞墙**:计算 {sm:.0f}%、内存 {mem:.0f}%。往下看占用率和「warp 在等什么」"
    else:
        head = f"**两边吃得差不多**:计算 {sm:.0f}%、内存 {mem:.0f}%。再往上要同时动两边"
    if occ is not None:
        head += f";每个 SM 上只有 {occ:.0f}% 的 warp 位置在用"
    return head


# ---------- 渲染 ----------


def render(run_dir: Path, cfg: TargetConfig, bench: dict | None = None) -> str:
    result = json.loads((run_dir / "result.json").read_text())
    rows = load_raw(run_dir / "ncu-raw.csv") if (run_dir / "ncu-raw.csv").exists() else []
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
        md.append("### 跑多快")
        md.append("")
        md.append("| 项 | 值 | 什么意思 |")
        md.append("|---|---|---|")
        if b:
            if b.get("ref_median_ms"):
                x = b["speedup_vs_ref"]
                d = abs(1 - x) * 100
                tag = f"比 torch 快 {d:.0f}%" if x >= 1.005 else (f"比 torch 慢 {d:.0f}%" if x <= 0.995 else "与 torch 打平")
                md.append(f"| **和 torch 比** | **{x:.2f}×** | {tag}(torch 用 {b['ref_median_ms']:.4f} ms,背后是 cuBLAS) |")
            md.append(f"| 耗时 | {b['median_ms']:.4f} ms | 跑 {b['iters']} 次取中位数;最快 {b['p10_ms']:.4f},最慢 {b['p90_ms']:.4f} |")
            fp_pct = f"用掉这张卡 {100 * b['tflops'] / peak_fp:.0f}% 的算力(满值 {peak_fp:.0f} TFLOPS)" if peak_fp else "-"
            bw_pct = f"用掉这张卡 {100 * b['gbps'] / peak_bw:.0f}% 的显存带宽(满值 {peak_bw:.0f} GB/s)" if peak_bw else "-"
            md.append(f"| 算力 | {b['tflops']:.1f} TFLOPS | {fp_pct} |")
            md.append(f"| 显存带宽 | {b['gbps']:.1f} GB/s | {bw_pct} |")
            if b.get("bytes") and peak_fp and peak_bw:
                ai = b["flops"] / b["bytes"]
                ridge = peak_fp * 1e12 / (peak_bw * 1e9)
                side = "算得不够快" if ai > ridge else "搬得不够快"
                md.append(f"| 每搬 1 字节要算几次 | {ai:.0f} 次 | 这张卡的分水岭是 {ridge:.0f} 次:超过就该是**{side}**是瓶颈 |")
        else:
            md.append("| 耗时 | 没找到同 case 的测速结果 | 先跑一次 bench |")
        md.append(f"| NCU 量到的单次耗时 | {_fmt(dur, '.1f', ' us')} | 剖析时另测的,锁频 {cfg.extra.get('ncu_clock_control', 'base')},与上面的中位数会有出入 |")
        md.append("")

        # 瓶颈定位
        l2_hit = _num(row, "lts__t_sector_hit_rate.pct")
        l1_hit = _num(row, "l1tex__t_sector_hit_rate.pct")
        md.append("### 哪个部件忙,哪个闲")
        md.append("")
        md.append("| 部件 | 忙到几成 | 什么意思 |")
        md.append("|---|---|---|")
        md.append(f"| 计算单元(总) | {_pct(_num(row, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'))} | 指令发射槽只忙 {_pct(_num(row, 'sm__issue_active.avg.pct_of_peak_sustained_elapsed'))},平均每周期发 {_fmt(_num(row, 'sm__inst_executed.avg.per_cycle_active'), '.2f')} 条指令 |")
        md.append(f"| **tensor core** | {_pct(_num(row, 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'))} | 矩阵乘专用单元。矩阵类算子就看这一行 |")
        md.append(f"| 普通浮点 / 整数逻辑 / 访存指令 | {_pct(_num(row, 'sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed'))} / {_pct(_num(row, 'sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_elapsed'))} / {_pct(_num(row, 'sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed'))} | 整数逻辑高多半是地址算得太多;普通浮点高说明没走 tensor core |")
        md.append(f"| 内存通道(总) | {_pct(_num(row, 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed'))} | 取下面三级里最忙的那个 |")
        md.append(f"| ├ 显存 | {_pct(_num(row, 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed'))} | 实际从显存搬了 {_fmt(_gbps(row, 'dram__bytes.sum.per_second'), '.0f', ' GB/s')}。到 80% 就是带宽到头了 |")
        md.append(f"| ├ L2 缓存 | {_pct(_num(row, 'lts__throughput.avg.pct_of_peak_sustained_elapsed'))} | {_pct(l2_hit)} 的请求在这里就找到了{',没惊动显存' if (l2_hit or 0) >= 70 else ',其余要去显存取'} |")
        md.append(f"| └ L1 缓存 | {_pct(_num(row, 'l1tex__throughput.avg.pct_of_peak_sustained_active'))} | 命中 {_pct(l1_hit)}{'。很低是正常的:数据被你手动搬进共享内存了,不走 L1' if (l1_hit or 0) < 20 else ''} |")
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
            "每 SM 的 warp 上限": _num(row, "launch__occupancy_limit_warps"),
            "每 SM 的 block 上限": _num(row, "launch__occupancy_limit_blocks"),
        }
        valid = {k: v for k, v in limits.items() if v}  # 0 是 ncu 对超出默认 carveout 的动态共享内存的占位值,不当限制因子
        limiter = min(valid, key=valid.get) if valid else "-"
        md.append("### 卡子怎么切的,SM 喂饱了吗")
        md.append("")
        md.append("| 项 | 值 | 什么意思 |")
        md.append("|---|---|---|")
        md.append(f"| 切成多少块 × 每块多少线程 | {_fmt(grid, '.0f')} × {_fmt(block, '.0f')} | {_tail_wave(grid, waves, dev.get('sm_count'))} |")
        md.append(f"| 每个线程占几个寄存器 | {_fmt(regs, '.0f')} | 寄存器越多,每个 SM 同时放得下的块越少 —— 这里是 {_fmt(limits['寄存器'], '.0f')} 块 |")
        md.append(f"| 每块占多少共享内存 | {_fmt(smem, '.1f', ' KB')} | 按这个算,每个 SM 放得下 {_fmt(limits['共享内存'], '.0f')} 块 |")
        # 理论占用率已经接近满的时候,说「卡在某某上」是误导 —— 那时占用率根本不是瓶颈
        if occ_t is not None and occ_t >= 95:
            why = "warp 位置已经排满,占用率不是瓶颈。真跑不满就是在等,看下一节"
        else:
            why = (f"**卡在{limiter}上**:它只允许每个 SM 同时放 {_fmt(valid.get(limiter), '.0f')} 块。"
                   f"位置越满,越能用别的 warp 盖住等待")
        md.append(f"| warp 位置用了多少(理论/实测) | {_pct(occ_t)} / {_pct(occ_a)} | {why} |")
        spill = _num(row, "derived__local_spilling_requests")
        md.append(f"| 寄存器不够、溢出到显存 | {_fmt(spill, '.0f')} 次 | {'0 就好' if not spill else '**非 0 是硬伤**,先减寄存器压力'} |")
        md.append("")

        # 调度与 stall
        st = stall_reasons(row)
        elig = _num(row, "smsp__warps_eligible.avg.per_cycle_active")
        act = _num(row, "smsp__warps_active.avg.per_cycle_active")
        md.append("### warp 在等什么")
        md.append("")
        md.append("| 项 | 值 | 什么意思 |")
        md.append("|---|---|---|")
        idle = "手上的 warp 大多在等,发不出指令" if (act and elig is not None and elig < act * 0.25) else "多数时候有 warp 可发"
        md.append(f"| 每个调度器手上有几个 warp | {_fmt(act, '.2f')} 个,其中 {_fmt(elig, '.2f')} 个能立刻发 | {idle} |")
        if st:
            total = sum(v for _, v in st) or 1
            for name, v in st:
                md.append(f"| {STALL_ZH.get(name, name)} | {v:.2f} | 占主要等待的 {100 * v / total:.0f}%(NCU 里叫 `{name}`) |")
        else:
            md.append("| 未采集 | - | 需要 WarpStateStats section |")
        md.append("")
    return "\n".join(md)


def _tail_wave(grid: float | None, waves: float | None, sm_count: int | None) -> str:
    """尾波:块数不是「一波」的整数倍时,最后一波大半个 GPU 在空转。

    实际要跑 ceil(waves) 波,理想只需 waves 波,所以浪费 = (ceil - waves) / ceil。
    3.01 波是最坏情况(要跑 4 波,最后一波只有 1% 的位置在用,浪费 25%),
    3.98 波反而几乎没有浪费 —— 别被小数点骗了。
    """
    if not grid or not waves or waves <= 0:
        return "-"
    per_wave = grid / waves
    base = f"要分 {waves:.2f} 波跑完(一波 = 全卡同时能跑的块数,这里约 {per_wave:.0f} 块)"
    n = math.ceil(waves - 1e-9)
    waste = (n - waves) / n
    if waste < 0.02:
        return base + ",块数正好铺满,没有浪费"
    last = waves - (n - 1)
    if waste < 0.08:
        return base + f",最后一波用掉 {100 * last:.0f}% 的位置,浪费约 {100 * waste:.0f}%"
    return base + f",**最后一波只用掉 {100 * last:.0f}% 的位置,约浪费 {100 * waste:.0f}% 的时间**"


def find_latest_bench(root: Path, kernel: str, target: str) -> dict | None:
    cands = sorted((root / "runs").glob(f"*-{kernel}-{target}-bench/result.json"))
    for p in reversed(cands):
        try:
            return json.loads(p.read_text())
        except Exception:
            continue
    return None
