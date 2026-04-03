from __future__ import annotations

import argparse
import time
import os
import sys
from dataclasses import dataclass
from typing import List

# Ensure repo root is on path when running as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mlx.core as mx

from mlx_qwen.kv_quant_integration import CompressedKVStore


def dummy_tokenizer(text: str) -> List[int]:
    # Placeholder: replace with actual tokenizer for Qwen
    return [1] * len(text)


@dataclass
class Metrics:
    tokens_per_sec: float
    ttft_ms: float
    peak_bytes: int


def run_decode(seq_len: int, d: int, unbiased: bool) -> Metrics:
    store = CompressedKVStore(d=d, bits=3.5, unbiased=unbiased, seed=0)
    q = mx.random.normal(shape=(d,))
    start = time.time()
    ttft = None
    for t in range(seq_len):
        # Synthesize K,V vectors
        k = mx.random.normal(shape=(d,))
        v = mx.random.normal(shape=(d,))
        store.append(k, v)
        scores = store.attention_scores(q)
        if ttft is None:
            ttft = (time.time() - start) * 1000
        _ = scores
    elapsed = time.time() - start
    tps = seq_len / max(elapsed, 1e-6)
    # Rough memory via Python: not accurate; placeholder
    peak_bytes = 0
    return Metrics(tokens_per_sec=tps, ttft_ms=ttft, peak_bytes=peak_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--d", type=int, default=128)
    args = ap.parse_args()

    for unbiased in [False, True]:
        m = run_decode(args.seq_len, args.d, unbiased=unbiased)
        label = "TurboQuantmse" if not unbiased else "TurboQuantprod"
        print(f"{label}: tps={m.tokens_per_sec:.2f}, ttft_ms={m.ttft_ms:.1f}, peak_bytes≈{m.peak_bytes}")


if __name__ == "__main__":
    main()
