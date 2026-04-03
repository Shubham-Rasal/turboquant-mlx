from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mlx_qwen.auto_patch import auto_patch_model
import mlx.core as mx


def load_model(model_ids):
    from mlx_lm import load
    last = None
    for mid in model_ids:
        try:
            print(f"[load] trying: {mid}")
            m, tok = load(mid)
            print(f"[load] ok: {mid}")
            return m, tok, mid
        except Exception as e:
            print(f"[load] failed: {mid}: {e}")
            last = e
    raise last


def make_long_prompt(tokenizer, target_tokens: int) -> str:
    # Repeat a simple sentence until tokenized length >= target_tokens
    chunk = "The ocean is vast and deep. " * 50
    s = chunk
    # Use tokenizer to count tokens; mlx-lm tokenizers follow HF fast tokenizer API
    while True:
        n = len(tokenizer.encode(s))
        if n >= target_tokens:
            break
        s += chunk
    return s


def measure_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> float:
    from mlx_lm import generate
    t0 = time.time()
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=max_new_tokens)
    dt = time.time() - t0
    return max_new_tokens / max(dt, 1e-6)


def kv_bytes_per_token(model) -> tuple[int, int, int]:
    # Inspect first layer attention for shapes
    attn = model.model.layers[0].self_attn
    n_heads = int(getattr(attn, "n_heads", getattr(attn, "num_heads", 0)))
    n_kv = int(getattr(attn, "n_kv_heads", getattr(attn, "num_kv_heads", n_heads)))
    # head_dim: infer from q_proj out features / n_heads
    w = attn.q_proj.weight
    head_dim = int(w.shape[0] // n_heads)
    n_layers = int(getattr(model.model, "num_hidden_layers", len(model.model.layers)))
    # Baseline bf16: 2 bytes per element, K and V each: bytes/token/layer = 2*(n_kv*head_dim)*2
    bf16_per_layer = 2 * n_kv * head_dim * 2
    bf16_per_token = bf16_per_layer * n_layers
    # TurboQuantprod @ 3.5 bpc total (keys+values → 7.0 bpc): bits/token/layer = 7 * n_kv * head_dim
    # plus two scalars mu per (K,V) per head per layer → ~ 2 * 4 bytes
    tq_bits_per_layer = 7.0 * n_kv * head_dim
    tq_bytes_per_layer = int(tq_bits_per_layer // 8)
    mu_overhead = 2 * 4
    tq_per_layer = tq_bytes_per_layer + mu_overhead
    tq_per_token = tq_per_layer * n_layers
    return bf16_per_token, tq_per_token, n_layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8192, help="Target prompt length in tokens")
    ap.add_argument("--tokens", type=int, default=64, help="New tokens to generate")
    ap.add_argument("--bits", type=float, default=3.5)
    args = ap.parse_args()

    candidates = [
        "mlx-community/Qwen2.5-3B-Instruct-bf16",
        "mlx-community/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ]

    model, tokenizer, used = load_model(candidates)
    long_prompt = make_long_prompt(tokenizer, args.ctx)

    print(f"[info] model={used}; context≈{args.ctx} toks; gen={args.tokens} toks")
    # Baseline: measure tokens/sec and record peak RSS via ps if available
    bf16_tps = measure_generate(model, tokenizer, long_prompt, args.tokens)
    print(f"[baseline] tokens/sec={bf16_tps:.2f}")

    patched = auto_patch_model(model, use_quant=True, bits=args.bits, unbiased=True)
    print(f"[patch] attention blocks patched: {patched}")
    # After patching, access kv stores for rough byte counting
    tq_tps = measure_generate(model, tokenizer, long_prompt, args.tokens)
    # Compute measured KV bytes across layers/heads if exposed
    measured_bytes = 0
    try:
        for lyr in model.model.layers:
            attn = getattr(lyr, "self_attn", None)
            stores = getattr(attn, "tq_kvstores", None)
            if not stores:
                continue
            for s in stores:
                qs = getattr(s, "qstore", None)
                if qs is None:
                    continue
                for cv in qs.K + qs.V:
                    if cv.idx_packed is not None:
                        measured_bytes += int(cv.idx_packed.nbytes)
                    if cv.idx is not None:
                        measured_bytes += int(cv.idx.nbytes)
                    if cv.qjl_signs is not None:
                        measured_bytes += int(cv.qjl_signs.nbytes)
                    if cv.mu is not None:
                        measured_bytes += int(cv.mu.nbytes)
    except Exception:
        measured_bytes = -1
    print(f"[quantized] tokens/sec={tq_tps:.2f}; speedup x{(tq_tps/max(bf16_tps,1e-6)):.2f}")

    bf16_bpt, tq_bpt, n_layers = kv_bytes_per_token(model)
    total_tokens = args.ctx
    print(f"[kv-theory] per-token bytes: bf16={bf16_bpt}, tq≈{tq_bpt}; layers={n_layers}")
    print(f"[kv-theory] total KV for {total_tokens} toks: bf16≈{bf16_bpt*total_tokens/1e6:.2f} MB; tq≈{tq_bpt*total_tokens/1e6:.2f} MB")
    if measured_bytes >= 0:
        print(f"[kv-measured] compressed bytes in stores≈{measured_bytes/1e6:.2f} MB (excludes allocator overhead)")


if __name__ == "__main__":
    main()
