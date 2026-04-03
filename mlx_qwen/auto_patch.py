from __future__ import annotations

import inspect
from typing import Any, Callable, Set

import mlx.core as mx

from .qwen_wrapper import patch_qwen_block


def _is_attn_module(obj: Any) -> bool:
    # Support MLX Qwen Attention with attributes n_heads/n_kv_heads and q_proj/... submodules
    has_proj = all(hasattr(obj, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    has_heads = any(hasattr(obj, n) for n in ("num_heads", "n_heads"))
    return has_proj and has_heads


def _iter_modules(obj: Any, seen: Set[int] | None = None):
    if seen is None:
        seen = set()
    if obj is None:
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    yield obj
    # Recurse lists/tuples of modules
    if isinstance(obj, (list, tuple)):
        for itm in obj:
            yield from _iter_modules(itm, seen)
        return
    # Recurse attributes conservatively
    for name in ("model", "layers", "blocks", "modules", "transformer", "self_attn", "attention", "attn"):
        if hasattr(obj, name):
            try:
                child = getattr(obj, name)
            except Exception:
                continue
            yield from _iter_modules(child, seen)


def _make_forward_wrapper(attn, patcher) -> Callable:
    orig = getattr(attn, "forward", None)

    def forward(x, *args, **kwargs):
        # Decode path: single token common cases
        try:
            if isinstance(x, mx.array):
                if x.ndim == 1:
                    return patcher.attn.quant_step(x)
                if x.ndim == 2 and x.shape[0] == 1:
                    y = patcher.attn.quant_step(x[0])
                    return y[None, :]
        except Exception:
            pass
        # Fallback: original forward
        if orig is None:
            raise RuntimeError("Attention module has no forward and cannot be wrapped")
        return orig(x, *args, **kwargs)

    return forward


def auto_patch_model(model, use_quant: bool = True, bits: float = 3.5, unbiased: bool = True, seed: int = 0) -> int:
    """
    Find Qwen-style attention modules in `model`, attach `quant_step`, and monkey-patch
    their `forward` to use TurboQuant for single-token decode. Returns number of patched
    modules. Multi-token inputs fall back to the original forward.
    """
    count = 0
    for m in _iter_modules(model):
        if _is_attn_module(m):
            patcher = patch_qwen_block(m, use_quant=use_quant, bits=bits, unbiased=unbiased, seed=seed + count)
            try:
                m._orig_forward = getattr(m, "forward", None)
                m.forward = _make_forward_wrapper(m, patcher)
                count += 1
            except Exception:
                # Attach quant_step even if we can't replace forward
                count += 0
    return count
