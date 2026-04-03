"""
Measure real KV-cache memory:
  - Baseline:   mlx_lm KVCache (full bf16)
  - TurboQuant: TurboQuantKVCache (3.5 bpc compressed)

Both use generate_step so the cache objects accumulate real tokens.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

from mlx_turboquant import make_tq_cache


def _sum_cache_nbytes(cache) -> int:
    total = 0
    for c in cache:
        nb = getattr(c, "nbytes", 0)
        total += int(nb) if nb else 0
    return total


def run_and_measure(model, tokenizer, prompt: str, max_new_tokens: int, cache):
    toks = mx.array(tokenizer.encode(prompt))
    t0 = time.time()
    token_ids = []
    for token_id, _logprobs in generate_step(toks, model, max_tokens=max_new_tokens, prompt_cache=cache):
        token_ids.append(token_id)
    elapsed = time.time() - t0
    nb = _sum_cache_nbytes(cache)
    tps = len(token_ids) / max(elapsed, 1e-6)
    text = tokenizer.decode(token_ids)
    return nb, tps, len(token_ids), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Write a short paragraph about the ocean.")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--bits", type=float, default=3.5)
    args = ap.parse_args()

    candidates = [
        "mlx-community/Qwen2.5-3B-Instruct-bf16",
        "mlx-community/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]
    model_id = None
    for mid in candidates:
        try:
            print(f"[load] trying {mid} ...", flush=True)
            model, tokenizer = load(mid)
            model_id = mid
            print(f"[load] ok: {mid}", flush=True)
            break
        except Exception as e:
            print(f"[load] failed: {e}", flush=True)

    if model_id is None:
        raise RuntimeError("Could not load any model")

    prompt_str = args.prompt

    # ── Baseline: full bf16 KVCache ────────────────────────────────────────
    print("\n[run] BASELINE (bf16 KVCache) ...", flush=True)
    baseline_cache = make_prompt_cache(model)
    bf16_bytes, bf16_tps, bf16_toks, bf16_text = run_and_measure(
        model, tokenizer, prompt_str, args.tokens, baseline_cache
    )

    # ── TurboQuant: compressed KVCache ─────────────────────────────────────
    print(f"[run] TURBOQUANT ({args.bits} bpc) ...", flush=True)
    tq_cache = make_tq_cache(model, bits=args.bits)
    tq_bytes, tq_tps, tq_toks, tq_text = run_and_measure(
        model, tokenizer, prompt_str, args.tokens, tq_cache
    )

    # ── Report ─────────────────────────────────────────────────────────────
    ratio = bf16_bytes / max(tq_bytes, 1) if tq_bytes > 0 else float("inf")
    sep = "=" * 60
    print()
    print(sep)
    print(f"Prompt: {repr(prompt_str)}")
    print(sep)
    print()
    print("── Generated text ──────────────────────────────────────")
    print(f"[bf16]       {bf16_text.strip()}")
    print()
    print(f"[TurboQuant] {tq_text.strip()}")
    print()
    print(sep)
    print(f"{'':30s}  {'bf16':>12}  {'TurboQuant':>12}")
    print(f"{'Tokens generated':30s}  {bf16_toks:>12}  {tq_toks:>12}")
    print(f"{'KV cache nbytes':30s}  {bf16_bytes:>12,}  {tq_bytes:>12,}")
    print(f"{'KV cache MB':30s}  {bf16_bytes/1e6:>11.2f}  {tq_bytes/1e6:>11.2f}")
    print(f"{'Throughput tok/s':30s}  {bf16_tps:>12.2f}  {tq_tps:>12.2f}")
    print()
    if tq_bytes > 0:
        print(f"Memory reduction: {ratio:.2f}× smaller  ({(1-1/ratio)*100:.1f}% saved)")
    print(sep)


if __name__ == "__main__":
    main()
