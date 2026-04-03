"""
make_tq_cache(model) — build a list of KVCache objects (one per transformer layer)
ready to pass to mlx_lm's generate_step as prompt_cache.

Layer-adaptive mode (fp16_layers > 0):
    The first and last fp16_layers layers use the standard mlx_lm KVCache (full
    precision). Middle layers use TurboQuantKVCache. This preserves quality on
    smaller models where the first/last layers are most sensitive to quantization.
"""
from __future__ import annotations

from typing import List, Union

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


def _make_fp16_cache(model):
    """Build a standard mlx_lm KVCache list (full precision) for all layers."""
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(model)


def make_tq_cache(
    model,
    bits: float = 3.5,
    seed: int = 0,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    fp16_layers: int = 0,
) -> List[Union[TurboQuantKVCache, object]]:
    """
    Build one cache per transformer layer for use as ``prompt_cache``.

    Args:
        model:        The loaded mlx_lm model (must expose model.model.layers).
        bits:         Bits per coordinate for TurboQuantMSE (default 3.5).
        seed:         Base RNG seed.
        n_kv_heads:   Override auto-detection.
        head_dim:     Override auto-detection.
        fp16_layers:  Number of layers at the *start* and *end* of the network
                      to keep in full FP16. Set to 1-4 to improve quality on
                      smaller models (<= 7B). Default: 0 (all layers compressed).

    Returns:
        List of cache objects (TurboQuantKVCache or KVCache) suitable for
        passing as ``prompt_cache`` to ``mlx_lm.generate_step``.
    """
    layers = model.model.layers
    n_layers = len(layers)

    fp16_set: set[int] = set()
    if fp16_layers > 0:
        for i in range(min(fp16_layers, n_layers)):
            fp16_set.add(i)
        for i in range(max(0, n_layers - fp16_layers), n_layers):
            fp16_set.add(i)

    fp16_baseline = _make_fp16_cache(model) if fp16_set else None

    caches = []
    for i, lyr in enumerate(layers):
        if i in fp16_set:
            caches.append(fp16_baseline[i])
            continue

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
