from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mlx_qwen.auto_patch import auto_patch_model


def try_load_model(model_ids):
    from mlx_lm import load
    last_err = None
    for mid in model_ids:
        try:
            print(f"[load] trying: {mid}")
            model, tokenizer = load(mid)
            print(f"[load] ok: {mid}")
            return model, tokenizer, mid
        except Exception as e:
            print(f"[load] failed: {mid}: {e}")
            last_err = e
    raise last_err


def gen_tokens(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    from mlx_lm import generate
    # Greedy decode for stable timing
    t0 = time.time()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens)
    elapsed = time.time() - t0
    ntoks = max_new_tokens
    tps = ntoks / max(elapsed, 1e-6)
    return out, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="Write a short haiku about the moon.")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--quant-bits", type=float, default=3.5)
    args = ap.parse_args()

    # Prefer MLX community bf16 weights; fallback to HF conversion
    candidates = [
        "mlx-community/Qwen2.5-3B-Instruct-bf16",
        "mlx-community/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    model, tokenizer, used = try_load_model(candidates)

    # Baseline timing
    print(f"[baseline] model={used}")
    _, tps_base = gen_tokens(model, tokenizer, args.prompt, max_new_tokens=args.tokens)
    print(f"[baseline] tokens/sec={tps_base:.2f}")

    # Apply auto patch for TurboQuant KV (unbiased). Some models wrap attention differently,
    # so it's normal if zero modules are patched; we'll still report baseline numbers.
    n = auto_patch_model(model, use_quant=True, bits=args.quant_bits, unbiased=True)
    print(f"[patch] attention blocks patched: {n}")

    # Quantized timing
    _, tps_quant = gen_tokens(model, tokenizer, args.prompt, max_new_tokens=args.tokens)
    print(f"[quantized] tokens/sec={tps_quant:.2f}")
    print(f"[delta] speedup x{(tps_quant / max(tps_base, 1e-6)):.2f}")


if __name__ == "__main__":
    main()
