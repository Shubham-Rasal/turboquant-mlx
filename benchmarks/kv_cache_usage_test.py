"""
Measure real KV-cache memory:
  - Baseline:   mlx_lm KVCache (full bf16)
  - TurboQuant: TurboQuantKVCache (3.5 bpc compressed)

Both use generate_step so the cache objects accumulate real tokens.

Pass a real document via --prompt or --prompt-file for honest benchmarking.

Note: ``--tokens`` is the number of *decode* steps. For long-context KV benchmarks
pass a long --prompt-file rather than a huge --tokens value.
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

from mlx_turboquant import HAS_METAL_KERNELS, make_tq_cache


def _sum_cache_nbytes(cache) -> int:
    total = 0
    for c in cache:
        nb = getattr(c, "nbytes", 0)
        total += int(nb) if nb else 0
    return total


def run_and_measure(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    cache,
    *,
    prefill_step_size: int = 2048,
):
    toks = mx.array(tokenizer.encode(prompt))
    t0 = time.time()
    token_ids = []
    for token_id, _logprobs in generate_step(
        toks,
        model,
        max_tokens=max_new_tokens,
        prompt_cache=cache,
        prefill_step_size=prefill_step_size,
    ):
        token_ids.append(token_id)
    elapsed = time.time() - t0
    nb = _sum_cache_nbytes(cache)
    tps = len(token_ids) / max(elapsed, 1e-6)
    text = tokenizer.decode(token_ids)
    return nb, tps, len(token_ids), text


def main():
    ap = argparse.ArgumentParser(
        description="KV cache memory benchmark: bf16 baseline vs TurboQuant compressed cache."
    )
    ap.add_argument(
        "--prompt",
        default="Write a short paragraph about the ocean.",
        help="Prompt string passed directly to the model.",
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        metavar="PATH",
        help="Read prompt from a text file (overrides --prompt). Use a real document for "
        "honest long-context benchmarks.",
    )
    ap.add_argument(
        "--tokens",
        type=int,
        default=64,
        metavar="N",
        help="Number of new tokens to decode after the prompt.",
    )
    ap.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="mlx_lm prefill chunk size (default 2048).",
    )
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument(
        "--fp16-layers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Keep the first and last N layers in full FP16 (uncompressed). "
            "Improves quality on small models (<=7B). Recommended: 2-4 for 3B, 1-2 for 32B. "
            "Default: 0 (all layers compressed)."
        ),
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Hugging Face model id. If omitted, tries Qwen2.5-3B-Instruct variants.",
    )
    args = ap.parse_args()

    if args.model:
        print(f"[load] loading {args.model} ...", flush=True)
        model, tokenizer = load(args.model)
        model_id = args.model
        print(f"[load] ok: {args.model}", flush=True)
    else:
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

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_str = f.read()
        print(f"[prompt] loaded from {args.prompt_file}", flush=True)
    else:
        prompt_str = args.prompt
    prompt_toks = len(tokenizer.encode(prompt_str))

    print(f"[config] Metal TurboQuant: {HAS_METAL_KERNELS}", flush=True)
    if args.fp16_layers == 0:
        print(
            "[config] tip: add --fp16-layers 2 (or 4 for 3B models) to improve quality "
            "at the cost of ~15-30% less compression",
            flush=True,
        )

    # ── Baseline: full bf16 KVCache ────────────────────────────────────────
    print("\n[run] BASELINE (bf16 KVCache) ...", flush=True)
    baseline_cache = make_prompt_cache(model)
    bf16_bytes, bf16_tps, bf16_toks, bf16_text = run_and_measure(
        model,
        tokenizer,
        prompt_str,
        args.tokens,
        baseline_cache,
        prefill_step_size=args.prefill_step_size,
    )

    # ── TurboQuant: compressed KVCache ─────────────────────────────────────
    fp16_layers = args.fp16_layers
    fp16_tag = f", fp16_layers={fp16_layers}" if fp16_layers > 0 else ""
    print(f"[run] TURBOQUANT ({args.bits} bpc{fp16_tag}) ...", flush=True)
    tq_cache = make_tq_cache(model, bits=args.bits, fp16_layers=fp16_layers)
    tq_bytes, tq_tps, tq_toks, tq_text = run_and_measure(
        model,
        tokenizer,
        prompt_str,
        args.tokens,
        tq_cache,
        prefill_step_size=args.prefill_step_size,
    )

    # ── Report ─────────────────────────────────────────────────────────────
    ratio = bf16_bytes / max(tq_bytes, 1) if tq_bytes > 0 else float("inf")
    sep = "=" * 60
    print()
    print(sep)
    if len(prompt_str) > 240:
        print(f"Prompt: {repr(prompt_str[:120])}... ({prompt_toks} tokens, truncated display)")
    else:
        print(f"Prompt: {repr(prompt_str)} ({prompt_toks} tokens)")
    print(f"Decode tokens (--tokens): {args.tokens}")
    approx_ctx = prompt_toks + bf16_toks
    print(f"Approx KV sequence length: {prompt_toks} prefill + {bf16_toks} decode ≈ {approx_ctx}")
    print(sep)
    print()
    print("── Generated text ──────────────────────────────────────")
    print(f"[bf16]       {bf16_text.strip()}")
    print()
    print(f"[TurboQuant{fp16_tag}] {tq_text.strip()}")
    print()
    print(sep)
    tq_label = f"TQ{args.bits:g}bpc" + (f"+fp16×{fp16_layers}" if fp16_layers > 0 else "")
    print(f"{'':30s}  {'bf16':>12}  {tq_label:>12}")
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
