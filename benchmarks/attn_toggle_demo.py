from __future__ import annotations

import argparse
import os
import sys
import time

import mlx.core as mx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mlx_qwen.attention_patch import KVCacheToggle


def run_demo(d: int, steps: int, use_quant: bool) -> tuple[float, float]:
    scale = 1.0 / (d ** 0.5)
    kv = KVCacheToggle(d=d, use_quant=use_quant, bits=3.5, unbiased=True)
    q = mx.random.normal(shape=(d,))
    t0 = time.time()
    ttft = None
    for _ in range(steps):
        k = mx.random.normal(shape=(d,))
        v = mx.random.normal(shape=(d,))
        _ = kv.step(q, k, v, scale)
        if ttft is None:
            ttft = (time.time() - t0) * 1000
    elapsed = time.time() - t0
    return steps / max(elapsed, 1e-6), ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--steps", type=int, default=2048)
    args = ap.parse_args()

    for flag in [False, True]:
        tps, ttft = run_demo(args.d, args.steps, flag)
        label = "FP16_KV" if not flag else "TurboQuantprod_3p5bpc"
        print(f"{label}: tps={tps:.2f}, ttft_ms={ttft:.1f}")


if __name__ == "__main__":
    main()

