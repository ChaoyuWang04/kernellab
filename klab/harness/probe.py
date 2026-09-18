"""探测执行后端:设备属性 + 实测拷贝带宽 + 实测 fp16 matmul 吞吐。输出 JSON。"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

from klab.harness.runner import device_info


def time_ms(fn, warmup=5, iters=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda 不可用")

    info = device_info()
    n = 512 * 2**20  # 512 MB
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    ms = time_ms(lambda: b.copy_(a))
    info["measured_copy_gbps"] = round(2 * n / (ms * 1e-3) / 1e9, 1)  # 读 + 写

    m = 8192
    x = torch.randn(m, m, dtype=torch.float16, device="cuda")
    y = torch.randn(m, m, dtype=torch.float16, device="cuda")
    ms = time_ms(lambda: x @ y, warmup=3, iters=10)
    info["measured_fp16_matmul_tflops"] = round(2 * m**3 / (ms * 1e-3) / 1e12, 1)

    for k, v in info.items():
        print(f"{k:<28} {v}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
