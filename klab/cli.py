"""klab:Mac 上写算子,远端 GPU 上跑。

    klab targets                       列出后端
    klab setup  --target T [--toolchain triton]   在后端装环境
    klab probe  --target T             探测后端(设备、实测带宽与 matmul 吞吐)
    klab check  <kernel_dir> --target T          正确性
    klab bench  <kernel_dir> --target T          测速
    klab ncu    <kernel_dir> --target T [--full] NCU 剖析,报告拉回 runs/
    klab open   <run_dir | .ncu-rep>   用本地 Nsight Compute 打开
    klab report <run_dir>              重新渲染体检单
    klab sh     --target T             进后端 shell
    klab exec   "<shell>" --target T   在后端执行一段命令(诊断)
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from klab import toolchains
from klab.config import TargetConfig, load_targets, repo_root
from klab.harness.spec import KernelSpec
from klab.report import print_result
from klab.targets import make_target
from klab.targets.base import Target

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

TargetOpt = typer.Option(..., "--target", "-t", help="targets.toml 里的后端名")


def _resolve(target: str) -> tuple[Path, TargetConfig, Target]:
    root = repo_root()
    cfgs = load_targets(root)
    if target not in cfgs:
        raise SystemExit(f"未知 target {target!r};可选:{list(cfgs)}")
    return root, cfgs[target], make_target(cfgs[target])


def _kernel_dir(root: Path, kernel: Path) -> Path:
    k = kernel if kernel.is_absolute() else (Path.cwd() / kernel)
    k = k.resolve()
    if k.is_file():
        k = k.parent
    if not (k / "meta.toml").is_file():
        raise SystemExit(f"{k} 不是算子目录(缺 meta.toml)")
    try:
        k.relative_to(root)
    except ValueError:
        raise SystemExit(f"{k} 不在仓库 {root} 内")
    return k


def _run_id(kernel: str, target: str, mode: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{kernel}-{target}-{mode}"


def _remote_run(root: Path, cfg: TargetConfig, tgt: Target, kdir: Path, mode: str,
                cases: list[str], extra: list[str], ncu_prefix: str = "") -> Path:
    spec = KernelSpec.load(kdir)
    rid = _run_id(spec.name, cfg.name, mode)
    rel = kdir.relative_to(root).as_posix()
    remote_run = f"{tgt.runs_dir}/{rid}"
    py = tgt.env_python(spec.toolchain)
    case_args = " ".join(f"--case {c}" for c in cases)
    cmd = (
        f"mkdir -p {remote_run} && {ncu_prefix}{py} -m klab.harness.runner --kernel {rel} --mode {mode} "
        f"--out {remote_run}/result.json {case_args} {' '.join(extra)}"
    )
    console.print(f"[dim]→ {cfg.name}: sync[/]")
    tgt.sync(root)
    console.print(f"[dim]→ {cfg.name}: {mode} {rel}[/]")
    proc = tgt.run(cmd, check=False)
    local = root / "runs"
    tgt.fetch(remote_run, local)
    out = local / rid
    if not (out / "result.json").exists():
        raise SystemExit(f"后端 {cfg.name} 没有产出结果(exit {proc.returncode}):看上面的输出")
    # 把 case 元数据并进结果,供报表查 dtype
    data = json.loads((out / "result.json").read_text())
    data["cases"] = spec.cases
    (out / "result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


@app.command()
def targets():
    """列出 targets.toml 里的后端。"""
    root = repo_root()
    for name, c in load_targets(root).items():
        console.print(f"{name:<12} kind={c.kind:<6} host={c.host or '-':<12} root={c.root}")


@app.command()
def setup(target: str = TargetOpt, toolchain: str = typer.Option("triton", help="工具链名")):
    """在后端装该工具链的环境(幂等,可重复跑)。"""
    root, cfg, tgt = _resolve(target)
    tc = toolchains.get(toolchain)
    tgt.sync(root)
    tgt.run(tc.setup_script(tgt.root, cfg.python, tgt.repo_dir))


@app.command()
def probe(target: str = TargetOpt, toolchain: str = typer.Option("triton")):
    """探测后端:设备属性、实测拷贝带宽与 fp16 matmul 吞吐,结果存 runs/。"""
    root, cfg, tgt = _resolve(target)
    rid = _run_id("probe", cfg.name, "probe")
    remote_run = f"{tgt.runs_dir}/{rid}"
    tgt.sync(root)
    tgt.run(f"mkdir -p {remote_run} && {tgt.env_python(toolchain)} -m klab.harness.probe --out {remote_run}/probe.json")
    tgt.fetch(remote_run, root / "runs")
    console.print(f"[dim]结果 {root / 'runs' / rid / 'probe.json'}[/]")


@app.command()
def check(
    kernel: Path,
    target: str = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c"),
    ignore_requires: bool = typer.Option(False, "--ignore-requires", help="架构要求不满足也强行跑"),
):
    """正确性:与 kernel.py 的 reference() 逐 case 比对。"""
    root, cfg, tgt = _resolve(target)
    extra = ["--ignore-requires"] if ignore_requires else []
    out = _remote_run(root, cfg, tgt, _kernel_dir(root, kernel), "check", case or [], extra)
    ok = print_result(out / "result.json", cfg)
    raise typer.Exit(0 if ok else 1)


@app.command()
def bench(
    kernel: Path,
    target: str = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c"),
    iters: int = typer.Option(100),
    warmup: int = typer.Option(10),
    no_flush: bool = typer.Option(False, "--no-flush", help="测速前不刷 L2"),
    skip_check: bool = typer.Option(False, "--skip-check", help="不先跑正确性"),
    ignore_requires: bool = typer.Option(False, "--ignore-requires", help="架构要求不满足也强行跑"),
):
    """测速:预热 → 每次迭代前刷 L2 → CUDA event 计时 → 中位数/分位数 → GB/s 与 TFLOPS。"""
    root, cfg, tgt = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    ign = ["--ignore-requires"] if ignore_requires else []
    if not skip_check:
        out = _remote_run(root, cfg, tgt, kdir, "check", case or [], ign)
        if not print_result(out / "result.json", cfg):
            raise SystemExit("正确性未通过,不测速(--skip-check 可跳过)")
    extra = [f"--iters {iters}", f"--warmup {warmup}"] + (["--no-flush"] if no_flush else []) + ign
    out = _remote_run(root, cfg, tgt, kdir, "bench", case or [], extra)
    print_result(out / "result.json", cfg)
    console.print(f"[dim]结果 {out}[/]")


NCU_BASIC = "--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats"


@app.command()
def ncu(
    kernel: Path,
    target: str = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c"),
    full: bool = typer.Option(False, "--full", help="--set full(慢很多,replay 次数多)"),
    launches: int = typer.Option(1, help="每个 case 启动次数;ncu 对每次启动都做一轮 replay,默认 1"),
    open_gui: bool = typer.Option(False, "--open", help="完成后用本地 Nsight Compute 打开"),
):
    """NCU 剖析:在后端跑 ncu,把 .ncu-rep 与文本报告拉回 runs/。"""
    root, cfg, tgt = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    spec = KernelSpec.load(kdir)
    kfilter = f"-k regex:{spec.kernel_regex} " if spec.kernel_regex else ""
    # 容器里通常锁不了 GPU 时钟(Modal 就是),targets.toml 用 ncu_clock_control = "none" 关掉;有权限的机器保持默认 base 以稳定数字
    clock = f"--clock-control {cfg.extra.get('ncu_clock_control', 'base')} "
    sections = ("--set full" if full else NCU_BASIC) + " " + clock.strip()

    def run_with_ncu() -> Path:
        rid = _run_id(spec.name, cfg.name, "ncu")
        rel = kdir.relative_to(root).as_posix()
        remote_run = f"{tgt.runs_dir}/{rid}"
        py = tgt.env_python(spec.toolchain)
        case_args = " ".join(f"--case {c}" for c in (case or []))
        cmd = (
            f"mkdir -p {remote_run} && ncu --target-processes all {kfilter}{sections} -f -o {remote_run}/ncu "
            f"{py} -m klab.harness.runner --kernel {rel} --mode ncu --launches {launches} --out {remote_run}/result.json {case_args} "
            f"&& ncu --import {remote_run}/ncu.ncu-rep --page details > {remote_run}/ncu-details.txt "
            f"&& ncu --import {remote_run}/ncu.ncu-rep --page raw --csv > {remote_run}/ncu-raw.csv"
        )
        console.print(f"[dim]→ {cfg.name}: sync[/]")
        tgt.sync(root)
        console.print(f"[dim]→ {cfg.name}: ncu {rel}[/]")
        tgt.run(cmd)
        tgt.fetch(remote_run, root / "runs")
        out = root / "runs" / rid
        data = json.loads((out / "result.json").read_text())
        data["cases"] = spec.cases  # 体检单要按 case 的 dtype 选峰值
        (out / "result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        return out

    out = run_with_ncu()
    _print_report(root, cfg, out)
    console.print(f"[dim]NCU 报告 {out / 'ncu.ncu-rep'} · 文本 {out / 'ncu-details.txt'} · 体检单 {out / 'report.md'}[/]")
    if open_gui:
        _open_ncu(out / "ncu.ncu-rep")


def _print_report(root: Path, cfg: TargetConfig, run_dir: Path) -> None:
    from rich.markdown import Markdown

    from klab.kreport import find_latest_bench, render

    data = json.loads((run_dir / "result.json").read_text())
    bench = find_latest_bench(root, data["kernel"], cfg.name)
    md = render(run_dir, cfg, bench)
    (run_dir / "report.md").write_text(md)
    console.print(Markdown(md))


@app.command()
def report(run_dir: Path, target: Optional[str] = typer.Option(None, "--target", "-t", help="默认从目录名推断")):
    """重新渲染某次 ncu 运行的体检单(runs/<id>/report.md)。"""
    root = repo_root()
    run_dir = run_dir.resolve()
    if not (run_dir / "result.json").exists():
        raise SystemExit(f"{run_dir} 不是一次 ncu 运行(缺 result.json)")
    cfgs = load_targets(root)
    if target is None:
        target = next((n for n in cfgs if f"-{n}-" in run_dir.name), None)
        if target is None:
            raise SystemExit("无法从目录名推断 target,请用 --target 指定")
    _print_report(root, cfgs[target], run_dir)


def _open_ncu(rep: Path) -> None:
    subprocess.run(["open", "-a", "NVIDIA Nsight Compute", str(rep)], check=False)


@app.command(name="open")
def open_cmd(path: Path):
    """用本地 Nsight Compute 打开 .ncu-rep(可给 run 目录)。"""
    rep = path / "ncu.ncu-rep" if path.is_dir() else path
    if not rep.exists():
        raise SystemExit(f"找不到 {rep}")
    _open_ncu(rep)


@app.command(name="exec")
def exec_cmd(script: str, target: str = TargetOpt):
    """在后端的 repo 目录里执行一段 shell(诊断用)。"""
    root, _, tgt = _resolve(target)
    tgt.sync(root)
    proc = tgt.run(script, check=False)
    raise typer.Exit(proc.returncode)


@app.command()
def sh(target: str = TargetOpt):
    """进后端 shell(已 cd 到远端 repo,PATH 含 uv 与 cuda)。"""
    _, _, tgt = _resolve(target)
    tgt.shell()


if __name__ == "__main__":
    app()
