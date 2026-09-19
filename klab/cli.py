"""klab:Mac 上写算子,远端 GPU 上跑。<算子> 可写名字、kernels/<名>、specs/<名> 或其中的文件;--target 不给用 targets.toml [defaults]。

    klab targets                       列出后端
    klab setup  --target T [--toolchain triton]   在后端装环境
    klab probe  --target T             探测后端(设备、实测带宽与 matmul 吞吐)
    klab run    <算子> [--target T]      一条龙 check → bench → ncu + 体检单(agent 默认用它)
    klab check  <kernel_dir> --target T          正确性
    klab bench  <kernel_dir> --target T          测速
    klab ncu    <kernel_dir> --target T [--full] NCU 剖析,报告拉回 runs/
    klab ptx    <算子> --target T        dump PTX,按世代统计 mma.sync / wgmma / cp.async / tcgen05 等指令
    klab open   <run_dir | .ncu-rep>   用本地 Nsight Compute 打开
    klab report <run_dir>              重新渲染体检单
    klab compare [算子...] [-t 后端]     跨算子 / 跨后端的 bench 对比表
    klab sweep  <kernel_dir> --target T          按 meta.toml [sweep] 扫参
    klab baseline <kernel_dir> --target T        把最新 bench 钉成基线,之后 bench 自动报差
    klab web    [--port 8777]          本地面板(算子版 LeetCode):左题面右编辑器,Run / Submit
    klab sh     --target T             进后端 shell
    klab exec   "<shell>" --target T   在后端执行一段命令(诊断)
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from klab import toolchains
from klab.config import TargetConfig, default_target, load_targets, repo_root
from klab.harness.spec import KernelSpec
from klab.report import print_result
from klab.targets import make_target
from klab.targets.base import Target

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)
console = Console()

TargetOpt = typer.Option(None, "--target", "-t", help="targets.toml 里的后端名;不给用 [defaults] target")


def _resolve(target: str | None) -> tuple[Path, TargetConfig, Target]:
    root = repo_root()
    cfgs = load_targets(root)
    target = target or default_target(root)
    if target not in cfgs:
        raise SystemExit(f"未知 target {target!r};可选:{list(cfgs)}")
    return root, cfgs[target], make_target(cfgs[target])


def _kernel_dir(root: Path, kernel: Path) -> Path:
    """接受算子名、kernels/<名>、specs/<名>、或其中任一文件的路径,统一返回 specs/<名>。"""
    raw = str(kernel)
    if "/" not in raw and not Path(raw).exists():
        name = raw
    else:
        k = kernel if kernel.is_absolute() else (Path.cwd() / kernel)
        k = k.resolve()
        if k.is_file():
            k = k.parent
        try:
            rel = k.relative_to(root)
        except ValueError:
            raise SystemExit(f"{k} 不在仓库 {root} 内")
        parts = rel.parts
        if len(parts) < 2 or parts[0] not in ("kernels", "specs"):
            raise SystemExit(f"{rel} 既不在 kernels/ 也不在 specs/ 下")
        name = parts[1]
    spec_dir = root / "specs" / name
    if not (spec_dir / "meta.toml").is_file():
        hint = "kernels/" + name + " 存在,但还没有接线" if (root / "kernels" / name).exists() else "kernels/ 与 specs/ 下都没有它"
        raise SystemExit(f"specs/{name}/meta.toml 不存在({hint});按 docs/01-AGENT-PLAYBOOK.md 生成 specs/{name}/")
    return spec_dir


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
    os.environ["KLAB_TOOLCHAIN"] = spec.toolchain  # Modal 按它选镜像;SSH 后端靠 env_python(toolchain)
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
def setup(target: Optional[str] = TargetOpt, toolchain: str = typer.Option("triton", help="工具链名")):
    """在后端装该工具链的环境(幂等,可重复跑)。"""
    root, cfg, tgt = _resolve(target)
    tc = toolchains.get(toolchain)
    os.environ["KLAB_TOOLCHAIN"] = toolchain
    if hasattr(tc, "local_prepare"):
        tc.local_prepare(root)
    tgt.sync(root)
    if cfg.kind == "modal":
        # Modal 的环境就是镜像,第一次调用时构建;这里只触发构建并打印版本
        tgt.run("python -c \"import torch,importlib; print('[setup] torch', torch.__version__, torch.cuda.get_device_name(0)); "
                "[print('[setup]', m, importlib.import_module(m).__version__) for m in ('triton','tilelang') if importlib.util.find_spec(m)]\"")
        return
    tgt.run(tc.setup_script(tgt.root, cfg.python, tgt.repo_dir))


@app.command()
def probe(target: Optional[str] = TargetOpt, toolchain: str = typer.Option("triton")):
    """探测后端:设备属性、实测拷贝带宽与 fp16 matmul 吞吐,结果存 runs/。"""
    root, cfg, tgt = _resolve(target)
    os.environ["KLAB_TOOLCHAIN"] = toolchain
    rid = _run_id("probe", cfg.name, "probe")
    remote_run = f"{tgt.runs_dir}/{rid}"
    tgt.sync(root)
    tgt.run(f"mkdir -p {remote_run} && {tgt.env_python(toolchain)} -m klab.harness.probe --out {remote_run}/probe.json")
    tgt.fetch(remote_run, root / "runs")
    console.print(f"[dim]结果 {root / 'runs' / rid / 'probe.json'}[/]")


@app.command()
def check(
    kernel: Path,
    target: Optional[str] = TargetOpt,
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
    target: Optional[str] = TargetOpt,
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
    _print_baseline_delta(kdir, cfg, out / "result.json")
    console.print(f"[dim]结果 {out}[/]")


@app.command()
def run(
    kernel: Path,
    target: Optional[str] = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c", help="不给则 check/bench 跑全部 case,ncu 只抓第一个"),
    iters: int = typer.Option(50),
    ignore_requires: bool = typer.Option(False, "--ignore-requires"),
    open_gui: bool = typer.Option(False, "--open"),
):
    """一条龙:check → bench → ncu,最后打印体检单路径。agent 接到「测一下这个 kernel」默认用它。"""
    root, cfg, tgt = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    spec = KernelSpec.load(kdir)
    ign = ["--ignore-requires"] if ignore_requires else []
    out = _remote_run(root, cfg, tgt, kdir, "check", case or [], ign)
    if not print_result(out / "result.json", cfg):
        raise SystemExit("正确性未通过,停在 check;先修算子再测速")
    out = _remote_run(root, cfg, tgt, kdir, "bench", case or [], [f"--iters {iters}", "--warmup 10"] + ign)
    print_result(out / "result.json", cfg)
    _print_baseline_delta(kdir, cfg, out / "result.json")
    ncu_case = case or [spec.cases[0]["name"]]
    console.print(f"[dim]→ ncu 只抓 case {ncu_case}(全部 case 用 klab ncu 单独跑)[/]")
    _ncu_impl(root, cfg, tgt, kdir, ncu_case, False, 1, open_gui)


@app.command()
def sweep(
    kernel: Path,
    target: Optional[str] = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c"),
    iters: int = typer.Option(20),
    warmup: int = typer.Option(5),
    top: int = typer.Option(8, help="每个 case 显示前几名"),
    ignore_requires: bool = typer.Option(False, "--ignore-requires"),
):
    """扫参:按 meta.toml 的 [sweep] 笛卡尔积逐组 check + bench,按 case 列出最快的几组。"""
    from rich.table import Table

    root, cfg, tgt = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    extra = [f"--iters {iters}", f"--warmup {warmup}"] + (["--ignore-requires"] if ignore_requires else [])
    out = _remote_run(root, cfg, tgt, kdir, "sweep", case or [], extra)
    data = json.loads((out / "result.json").read_text())
    rows = data["results"]
    failed = [r for r in rows if "error" in r]
    by_case: dict[str, list] = {}
    for r in rows:
        if "error" not in r:
            by_case.setdefault(r["case"], []).append(r)
    for name, rs in by_case.items():
        rs.sort(key=lambda r: r["median_ms"])
        t = Table(title=f"{data['kernel']} · {name} · {cfg.name}(共 {len(rs)} 组可用,{len(failed)} 组失败)", expand=True)
        t.add_column("#", width=3); t.add_column("参数", overflow="fold", ratio=3); t.add_column("median ms"); t.add_column("TFLOPS"); t.add_column("GB/s")
        for i, r in enumerate(rs[:top]):
            t.add_row(str(i + 1), ",".join(f"{k}={v}" for k, v in r["config"].items()), f"{r['median_ms']:.4f}", f"{r['tflops']:.1f}", f"{r['gbps']:.0f}")
        console.print(t)
    if failed:
        console.print("[dim]失败的组:[/]")
        for r in failed[:10]:
            console.print(f"  [dim]{','.join(f'{k}={v}' for k, v in r['config'].items())}: {r['error']}[/]")
    console.print(f"[dim]结果 {out}[/]")


@app.command()
def ptx(
    kernel: Path,
    target: Optional[str] = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c", help="不给则用第一个 case;PTX 只随 tile 常量变,换 case 拿到的是同一份"),
    ignore_requires: bool = typer.Option(False, "--ignore-requires"),
):
    """dump PTX 并按世代统计指令:确认这份 kernel 到底降到了哪一代的 tensor core 与异步拷贝指令。

    体检单只说 Tensor pipe 用了多少,不说走的是 mma.sync 还是 wgmma;练「某一代的特性」时用这个查。
    """
    root, cfg, tgt = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    spec = KernelSpec.load(kdir)
    extra = ["--ignore-requires"] if ignore_requires else []
    out = _remote_run(root, cfg, tgt, kdir, "ptx", case or [spec.cases[0]["name"]], extra)
    _print_ptx(out)
    console.print(f"[dim]PTX 原文 {out / 'ptx'}[/]")


def _print_ptx(run_dir: Path) -> None:
    from rich.table import Table

    from klab.harness.ptxdump import ALWAYS_SHOW, PTX_FAMILIES

    data = json.loads((run_dir / "result.json").read_text())
    for r in data["results"]:
        counts = r["counts"]
        gens = r["generations"]
        t = Table(title=f"{data['kernel']} · {r['kernel']} · {data['device']['device']}(cc {data['device']['cc']})", expand=True)
        t.add_column("指令族", width=14); t.add_column("最低架构", width=9); t.add_column("条数", justify="right", width=6); t.add_column("说明", overflow="fold", ratio=1)
        for prefix, arch, marker, note in PTX_FAMILIES:
            n = counts.get(prefix, 0)
            if n == 0 and prefix not in ALWAYS_SHOW:
                continue
            t.add_row(prefix, arch, str(n), note, style="dim" if n == 0 else ("bold" if marker else None))
        console.print(t)
        hit = " + ".join(gens) if gens else "没有任何标志指令(没走 tensor core / 异步拷贝)"
        console.print(f"命中世代:[bold]{hit}[/bold]  ·  PTX {r['ptx_lines']} 行 · {r['file']}")


@app.command()
def baseline(kernel: Path, target: Optional[str] = TargetOpt, run: Optional[Path] = typer.Option(None, help="指定某次 bench 运行目录;默认取最新")):
    """把某次 bench 钉成基线(specs/<名>/baselines/<后端>.json,进 git);之后 bench 自动报与基线的差。"""
    from klab.kreport import find_latest_bench

    root, cfg, _ = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    spec = KernelSpec.load(kdir)
    data = json.loads((run / "result.json").read_text()) if run else find_latest_bench(root, spec.name, cfg.name)
    if not data or data.get("mode") != "bench":
        raise SystemExit("没找到 bench 结果,先跑 klab bench")
    dst = kdir / "baselines" / f"{cfg.name}.json"
    dst.parent.mkdir(exist_ok=True)  # kdir 现在是 specs/<名>
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    console.print(f"基线已写入 {dst.relative_to(root)}(设备 {data['device']['device']})")


def _print_baseline_delta(kdir: Path, cfg: TargetConfig, result_json: Path) -> None:
    from rich.table import Table

    base_path = kdir / "baselines" / f"{cfg.name}.json"
    if not base_path.exists():
        return
    base = {r["case"]: r for r in json.loads(base_path.read_text())["results"]}
    now = json.loads(result_json.read_text())["results"]
    t = Table(title=f"与基线对比(specs/{kdir.name}/baselines/{cfg.name}.json)")
    for c in ("case", "现在 ms", "基线 ms", "变化"):
        t.add_column(c)
    for r in now:
        b = base.get(r["case"])
        if not b:
            continue
        d = 100 * (r["median_ms"] - b["median_ms"]) / b["median_ms"]
        color = "green" if d <= -3 else ("red" if d >= 3 else "dim")
        t.add_row(r["case"], f"{r['median_ms']:.4f}", f"{b['median_ms']:.4f}", f"[{color}]{d:+.1f}%[/]")
    console.print(t)


NCU_BASIC = "--section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy --section LaunchStats --section ComputeWorkloadAnalysis --section SchedulerStats --section WarpStateStats"


@app.command()
def ncu(
    kernel: Path,
    target: Optional[str] = TargetOpt,
    case: Optional[list[str]] = typer.Option(None, "--case", "-c"),
    full: bool = typer.Option(False, "--full", help="--set full(慢很多,replay 次数多)"),
    launches: int = typer.Option(1, help="每个 case 启动次数;ncu 对每次启动都做一轮 replay,默认 1"),
    open_gui: bool = typer.Option(False, "--open", help="完成后用本地 Nsight Compute 打开"),
):
    """NCU 剖析:在后端跑 ncu,把 .ncu-rep 与文本报告拉回 runs/,自动渲染体检单。"""
    root, cfg, tgt = _resolve(target)
    kdir = _kernel_dir(root, kernel)
    _ncu_impl(root, cfg, tgt, kdir, case or [], full, launches, open_gui)


def _ncu_impl(root: Path, cfg: TargetConfig, tgt: Target, kdir: Path, case: list[str], full: bool, launches: int, open_gui: bool) -> Path:
    import shlex

    spec = KernelSpec.load(kdir)
    kfilter = f"-k {shlex.quote('regex:' + spec.kernel_regex)} " if spec.kernel_regex else ""  # 正则里可能有 |
    # 容器里通常锁不了 GPU 时钟(Modal 就是),targets.toml 用 ncu_clock_control = "none" 关掉;有权限的机器保持默认 base 以稳定数字
    clock = f"--clock-control {cfg.extra.get('ncu_clock_control', 'base')} "
    sections = ("--set full" if full else NCU_BASIC) + " " + clock.strip()
    rid = _run_id(spec.name, cfg.name, "ncu")
    rel = kdir.relative_to(root).as_posix()
    remote_run = f"{tgt.runs_dir}/{rid}"
    py = tgt.env_python(spec.toolchain)
    case_args = " ".join(f"--case {c}" for c in case)
    cmd = (
        f"mkdir -p {remote_run} && ncu --target-processes all {kfilter}{sections} -f -o {remote_run}/ncu "
        f"{py} -m klab.harness.runner --kernel {rel} --mode ncu --launches {launches} --out {remote_run}/result.json {case_args} "
        f"&& ncu --import {remote_run}/ncu.ncu-rep --page details > {remote_run}/ncu-details.txt "
        f"&& ncu --import {remote_run}/ncu.ncu-rep --page raw --csv > {remote_run}/ncu-raw.csv"
    )
    os.environ["KLAB_TOOLCHAIN"] = spec.toolchain
    console.print(f"[dim]→ {cfg.name}: sync[/]")
    tgt.sync(root)
    console.print(f"[dim]→ {cfg.name}: ncu {rel}[/]")
    tgt.run(cmd)
    tgt.fetch(remote_run, root / "runs")
    out = root / "runs" / rid
    data = json.loads((out / "result.json").read_text())
    data["cases"] = spec.cases  # 体检单要按 case 的 dtype 选峰值
    (out / "result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    _print_report(root, cfg, out)
    console.print(f"[dim]NCU 报告 {out / 'ncu.ncu-rep'} · 文本 {out / 'ncu-details.txt'} · 体检单 {out / 'report.md'}[/]")
    if open_gui:
        _open_ncu(out / "ncu.ncu-rep")
    return out


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


@app.command()
def compare(
    kernel: Optional[list[str]] = typer.Argument(None, help="算子名,可多个;不给则全部"),
    target: Optional[list[str]] = typer.Option(None, "--target", "-t", help="只看这些后端"),
    all_runs: bool = typer.Option(False, "--all", help="不只取每个 (算子, 后端) 最新一次"),
):
    """跨算子、跨后端的 bench 对比表(读 runs/ 里的结果,不联网)。"""
    from klab.compare import collect, print_table

    root = repo_root()
    print_table(collect(root, kernel or None, target or None, not all_runs), load_targets(root))


@app.command(name="exec")
def exec_cmd(script: str, target: Optional[str] = TargetOpt):
    """在后端的 repo 目录里执行一段 shell(诊断用)。"""
    root, _, tgt = _resolve(target)
    tgt.sync(root)
    proc = tgt.run(script, check=False)
    raise typer.Exit(proc.returncode)


@app.command()
def web(
    port: int = typer.Option(8777, help="监听端口,只绑 127.0.0.1"),
    no_open: bool = typer.Option(False, "--no-open", help="不自动打开浏览器"),
):
    """本地面板(算子版 LeetCode):左边题面与优化路线,右边写算子;Run 看对不对,Submit 出体检单。"""
    from klab.web import serve

    serve(repo_root(), port=port, open_browser=not no_open)


@app.command()
def sh(target: Optional[str] = TargetOpt):
    """进后端 shell(已 cd 到远端 repo,PATH 含 uv 与 cuda)。"""
    _, _, tgt = _resolve(target)
    tgt.shell()


if __name__ == "__main__":
    app()
