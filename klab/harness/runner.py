"""在执行后端上跑的 harness。只依赖 torch 与标准库,不依赖 CLI 那边的 typer/rich。

    python -m klab.harness.runner --kernel kernels/vector_add --mode check|bench|ncu --out <json>
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from klab.harness.spec import KernelSpec

L2_FLUSH_BYTES = 256 * 1024 * 1024  # 5090 L2 为 96 MB,256 MB 足够冲干净


def device_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    info = {
        "device": p.name,
        "cc": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "mem_gb": round(p.total_memory / 2**30, 1),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": platform.node(),
    }
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        pass
    return info


# 各代架构提供的硬件特性,是 meta.toml 里 requires.features 的匹配表。
# 注意 cc 数字不是超集关系:sm_120(5090)数字最大,却没有 Hopper 的 wgmma,也没有数据中心 Blackwell 的 tcgen05。
ARCH_FEATURES = {
    8: {"mma_sync", "async_copy"},                                                   # Ampere / Ada
    9: {"mma_sync", "async_copy", "wgmma", "tma", "cluster", "dsmem"},               # Hopper
    10: {"mma_sync", "async_copy", "tma", "cluster", "dsmem", "tcgen05", "tmem"},    # 数据中心 Blackwell(B200)
    12: {"mma_sync", "async_copy", "block_scaled_mma"},                              # 消费级 Blackwell(5090);TMA 支持待实测
}


def check_requirements(spec: KernelSpec, ignore: bool = False) -> None:
    major, minor = torch.cuda.get_device_capability(0)
    cc = major + minor / 10
    have = ARCH_FEATURES.get(major, set())
    missing = [f for f in spec.features if f not in have]
    problems = []
    if cc < spec.min_cc:
        problems.append(f"要求 cc >= {spec.min_cc},当前 cc {cc}")
    if missing:
        problems.append(f"要求特性 {spec.features},当前架构 sm_{major}{minor} 缺 {missing}")
    if not problems:
        return
    msg = f"[requires] {torch.cuda.get_device_name(0)} 不满足算子 {spec.name} 的要求:" + ";".join(problems)
    if ignore:
        print(msg + "(--ignore-requires,继续)", flush=True)
        return
    raise SystemExit(msg + "。换一个 --target,或加 --ignore-requires 强行跑")


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    out_f, ref_f = out.float(), ref.float()
    diff = (out_f - ref_f).abs()
    ok = bool(torch.allclose(out_f, ref_f, atol=atol, rtol=rtol))
    return {
        "ok": ok,
        "max_abs_err": float(diff.max()) if diff.numel() else 0.0,
        "mean_abs_err": float(diff.mean()) if diff.numel() else 0.0,
        "shape": list(out.shape),
        "dtype": str(out.dtype).replace("torch.", ""),
    }


def do_check(spec: KernelSpec, mod, cases: list[dict]) -> list[dict]:
    results = []
    for case in cases:
        inputs = mod.make_inputs(case, "cuda")
        out = mod.run(**inputs)
        torch.cuda.synchronize()
        ref = mod.reference(**inputs)
        torch.cuda.synchronize()
        r = {"case": case["name"], **compare(out, ref, spec.atol, spec.rtol)}
        results.append(r)
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"[check] {case['name']:<12} {flag}  max_abs_err={r['max_abs_err']:.3e}", flush=True)
    return results


def do_bench(spec: KernelSpec, mod, cases: list[dict], warmup: int, iters: int, flush: bool) -> list[dict]:
    results = []
    scratch = torch.empty(L2_FLUSH_BYTES, dtype=torch.uint8, device="cuda") if flush else None
    for case in cases:
        inputs = mod.make_inputs(case, "cuda")
        for _ in range(warmup):
            mod.run(**inputs)
        torch.cuda.synchronize()
        times_ms: list[float] = []
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            if scratch is not None:
                scratch.zero_()
            start.record()
            mod.run(**inputs)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        times_ms.sort()
        med = statistics.median(times_ms)
        w = mod.workload(case, **inputs)
        r = {
            "case": case["name"],
            "iters": iters,
            "median_ms": med,
            "min_ms": times_ms[0],
            "p10_ms": times_ms[int(0.1 * (iters - 1))],
            "p90_ms": times_ms[int(0.9 * (iters - 1))],
            "flops": int(w.get("flops", 0)),
            "bytes": int(w.get("bytes", 0)),
            "tflops": (w.get("flops", 0) / (med * 1e-3)) / 1e12 if med else 0.0,
            "gbps": (w.get("bytes", 0) / (med * 1e-3)) / 1e9 if med else 0.0,
        }
        results.append(r)
        print(
            f"[bench] {case['name']:<12} median {med:8.4f} ms  p10 {r['p10_ms']:8.4f}  p90 {r['p90_ms']:8.4f}"
            f"  {r['gbps']:8.1f} GB/s  {r['tflops']:7.2f} TFLOPS",
            flush=True,
        )
    return results


def do_ncu(spec: KernelSpec, mod, cases: list[dict], launches: int) -> list[dict]:
    """给 ncu 用:每个 case 跑 launches 次,不刷 L2(ncu 自己会 replay 并控制缓存)。"""
    results = []
    for case in cases:
        inputs = mod.make_inputs(case, "cuda")
        for _ in range(launches):
            mod.run(**inputs)
        torch.cuda.synchronize()
        results.append({"case": case["name"], "launches": launches})
        print(f"[ncu] {case['name']:<12} launched x{launches}", flush=True)
    return results


def do_sweep(spec: KernelSpec, mod, cases: list[dict], warmup: int, iters: int, flush: bool) -> list[dict]:
    """扫参:meta.toml 的 [sweep] 给参数候选,kernel.py 的 configure(**params) 应用一组;每组先 check 再 bench。"""
    import itertools
    import traceback

    grid = spec.sweep
    if not grid:
        raise SystemExit("meta.toml 没有 [sweep] 段(参数名 = [候选列表])")
    if not hasattr(mod, "configure"):
        raise SystemExit("kernel.py 缺少 configure(**params),扫参需要它把一组参数应用到算子上")
    names = list(grid)
    combos = [dict(zip(names, vals)) for vals in itertools.product(*(grid[n] for n in names))]
    print(f"[sweep] {len(combos)} 组参数 × {len(cases)} 个 case", flush=True)
    results = []
    for i, cfg in enumerate(combos):
        tag = ",".join(f"{k}={v}" for k, v in cfg.items())
        try:
            mod.configure(**cfg)
            chk = do_check(spec, mod, cases[:1])
            if not chk[0]["ok"]:
                results.append({"config": cfg, "error": f"check 未通过 max_abs_err={chk[0]['max_abs_err']:.3e}"})
                print(f"[sweep] {i + 1}/{len(combos)} {tag}: 结果错误", flush=True)
                continue
            for r in do_bench(spec, mod, cases, warmup, iters, flush):
                results.append({"config": cfg, **r})
        except Exception as e:  # 编译失败、共享内存超限、寄存器溢出……都算这组不可用
            msg = str(e).strip().splitlines()[-1][:200] if str(e).strip() else type(e).__name__
            results.append({"config": cfg, "error": msg})
            print(f"[sweep] {i + 1}/{len(combos)} {tag}: 失败 {msg}", flush=True)
            torch.cuda.synchronize()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--mode", choices=["check", "bench", "ncu", "sweep"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--no-flush", action="store_true", help="测速时不刷 L2")
    ap.add_argument("--launches", type=int, default=3, help="ncu 模式下每个 case 启动次数")
    ap.add_argument("--ignore-requires", action="store_true", help="架构要求不满足也继续")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda 不可用:检查驱动或 venv 里的 torch 是否为 CUDA 版")
    spec = KernelSpec.load(Path(args.kernel))
    check_requirements(spec, args.ignore_requires)
    mod = spec.load_module()
    cases = spec.select_cases(args.case)

    t0 = time.time()
    if args.mode == "check":
        results = do_check(spec, mod, cases)
    elif args.mode == "bench":
        results = do_bench(spec, mod, cases, args.warmup, args.iters, not args.no_flush)
    elif args.mode == "sweep":
        results = do_sweep(spec, mod, cases, args.warmup, args.iters, not args.no_flush)
    else:
        results = do_ncu(spec, mod, cases, args.launches)

    payload = {
        "kernel": spec.name,
        "toolchain": spec.toolchain,
        "mode": args.mode,
        "device": device_info(),
        "elapsed_s": round(time.time() - t0, 2),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    failed = [r for r in results if r.get("ok") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
