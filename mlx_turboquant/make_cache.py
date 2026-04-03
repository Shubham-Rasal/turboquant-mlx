"""
make_tq_cache(model) — build a list of TurboQuantKVCache objects (one per
transformer layer) ready to pass to mlx_lm's generate_step as prompt_cache.
"""
from __future__ import annotations

from typing import List

from .kv_cache import TurboQuantKVCache


def _get_kv_shape(attn) -> tuple[int, int]:
    """Extract (n_kv_heads, head_dim) from a Qwen2-style attention module."""
    n_heads = getattr(attn, "n_heads", None) or getattr(attn, "num_attention_heads", None)
    n_kv = getattr(attn, "n_kv_heads", None) or getattr(attn, "num_key_value_heads", None) or n_heads
    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and hasattr(attn, "q_proj"):
        head_dim = attn.q_proj.weight.shape[0] // n_heads
    if n_kv is None or head_dim is None:
        raise AttributeError(
            f"Cannot infer (n_kv_heads, head_dim) from {type(attn).__name__}; "
            "set them explicitly via make_tq_cache(model, n_kv_heads=..., head_dim=...)"
        )
    return int(n_kv), int(head_dim)


def make_tq_cache(
    model,
    bits: float = 3.5,
    seed: int = 0,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
) -> List[TurboQuantKVCache]:
    """
    Build one TurboQuantKVCache per transformer layer.

    Args:
        model:       The loaded mlx_lm model (must expose model.model.layers).
        bits:        Bits per coordinate for TurboQuantMSE (default 3.5).
        seed:        Base RNG seed.
        n_kv_heads:  Override if auto-detection fails.
        head_dim:    Override if auto-detection fails.

    Returns:
        List[TurboQuantKVCache] with len == number of layers, suitable for
        passing as `prompt_cache` to mlx_lm.generate_step.
    """
    layers = model.model.layers
    caches: List[TurboQuantKVCache] = []
    for i, lyr in enumerate(layers):
        attn = getattr(lyr, "self_attn", None)
        if attn is None:
            attn = getattr(lyr, "attention", getattr(lyr, "attn", None))
        if attn is None:
            raise AttributeError(f"Layer {i} has no recognizable attention attribute")

        nkv = n_kv_heads
        hd = head_dim
        if nkv is None or hd is None:
            nkv_auto, hd_auto = _get_kv_shape(attn)
            nkv = nkv or nkv_auto
            hd = hd or hd_auto

        caches.append(TurboQuantKVCache(n_kv_heads=nkv, head_dim=hd, bits=bits, seed=seed + i))
    return caches
