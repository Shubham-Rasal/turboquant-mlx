#!/usr/bin/env python
"""
serve_tq.py — OpenAI-compatible MLX server with TurboQuant KV-cache compression.

Accepts the same arguments as `mlx_lm.server.main` (i.e. `mlx_lm serve`).
The only difference: every new prompt cache is created with TurboQuantKVCache
(3.5 bpc) instead of the default full-precision KVCache.

Usage
-----
conda run -n mlx python serve_tq.py \
    --model mlx-community/Qwen2.5-3B-Instruct-bf16 \
    --port 8080

Then query it exactly like the standard mlx_lm server:
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "Write a haiku about the sea."}],
        "max_tokens": 64
      }'

How it works
------------
mlx_lm.server calls make_prompt_cache(model) each time it needs a fresh KV
cache. We replace that function with one that returns TurboQuantKVCache objects
so the server machinery (LRU prompt caching, nbytes accounting, streaming) works
unchanged. Batching is disabled because our cache has no `merge` attribute; the
server falls back to sequential-request mode which is fine for single-GPU use.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Patch must happen BEFORE mlx_lm.server is imported ──────────────────────
# mlx_lm.server does:  from .models.cache import make_prompt_cache
# That creates a local binding; we must replace it in the server module too.

import mlx_lm.models.cache as _cache_mod      # noqa: E402
import mlx_lm.server as _srv                  # noqa: E402  (imports make_prompt_cache)

from mlx_turboquant import make_tq_cache       # noqa: E402

_original_make_prompt_cache = _cache_mod.make_prompt_cache


def _tq_make_prompt_cache(model, max_kv_size=None):
    """
    Return a TurboQuant KV cache if the model has the expected layer structure,
    otherwise fall back to the standard cache gracefully.
    """
    try:
        cache = make_tq_cache(model)
        print(
            f"[turboquant] created {len(cache)} TurboQuantKVCache objects "
            f"(3.5 bpc, ~4× smaller than bf16)",
            flush=True,
        )
        return cache
    except Exception as e:
        print(f"[turboquant] falling back to standard cache: {e}", flush=True)
        return _original_make_prompt_cache(model, max_kv_size=max_kv_size)


# Patch both locations so the server picks up the new factory
_cache_mod.make_prompt_cache = _tq_make_prompt_cache
_srv.make_prompt_cache = _tq_make_prompt_cache


# ── Run the standard server ──────────────────────────────────────────────────
if __name__ == "__main__":
    _srv.main()
