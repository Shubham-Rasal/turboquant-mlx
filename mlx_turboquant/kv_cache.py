"""
TurboQuantKVCache — a drop-in replacement for mlx_lm's KVCache that stores
K and V in compressed form (3.5 bpc TurboQuantMSE) and decompresses on every
attention step using fully-batched MLX operations (no Python token loops).

Design
------
* The TurboQuantMSE codebook assumes unit-sphere vectors. Real K/V vectors have
  arbitrary L2 norms, so we store a per-token per-head float16 scale alongside
  the packed indices and normalise before compress / rescale after decompress.
* One shared rotation + quantizer for all KV heads.
* Storage: two pre-allocated uint8 tensors [n_kv_heads, capacity, d_pack] for
  indices, and two float16 tensors [n_kv_heads, capacity] for scales.
  All buffers grow in `step`-sized increments (mirrors KVCache).
* Compress path:  scale → normalize → rotate → encode → pack  (batched).
* Decompress path: unpack → gather_centroids → inv-rotate → rescale (batched).

Memory
------
At 3.5 bpc (head_dim=128): 64 bytes indices + 2 bytes scale = 66 bytes/token/head
vs 256 bytes bf16 → ~3.9× per head.  KVCache pre-allocates 256 slots, so at
short contexts the real ratio is much higher.
"""
from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx

from .turboquant import (
    StructuredHadamardRotation,
    ScalarQuantizerBeta,
    _Pack35,
    _next_pow2,
)


# ---------------------------------------------------------------------------
# Batched compress / decompress helpers
# ---------------------------------------------------------------------------

def _compress_batch(
    rot: StructuredHadamardRotation,
    sq: ScalarQuantizerBeta,
    x: mx.array,
) -> Tuple[mx.array, mx.array, int]:
    """
    Compress a batch of K or V vectors.

    x: [n_kv_heads, L, head_dim]

    Returns
    -------
    packed : [n_kv_heads, L, d_pack]  uint8
    scales : [n_kv_heads, L]          float16  (L2 norm of each vector)
    d_eff  : int                       padded-d stored by Pack35
    """
    # Per-vector L2 norm for normalisation; keepdims for broadcast
    scales = mx.sqrt(mx.sum(x * x, axis=-1))          # [n_kv_heads, L]
    x_norm = x / (scales[..., None] + 1e-8)           # unit-norm, [n_kv_heads, L, head_dim]

    z = rot.rotate(x_norm)                            # [n_kv_heads, L, d_pad]
    idx = sq.encode_indices(z)                        # [n_kv_heads, L, d_pad] uint16

    if abs(sq.bits - 3.5) < 1e-6:
        packed, d_eff = _Pack35.pack(idx)             # [n_kv_heads, L, d_pack] uint8
    else:
        packed, d_eff = idx, idx.shape[-1]

    return packed, scales.astype(mx.float16), d_eff


def _decompress_batch(
    rot: StructuredHadamardRotation,
    sq: ScalarQuantizerBeta,
    packed: mx.array,
    scales: mx.array,
    d_eff: int,
    is_packed35: bool,
) -> mx.array:
    """
    Decompress a batch of stored K or V vectors.

    packed : [n_kv_heads, T, d_pack]  uint8
    scales : [n_kv_heads, T]          float16

    Returns: [n_kv_heads, T, head_dim]
    """
    if is_packed35:
        idx = _Pack35.unpack(packed, d_eff)                     # [n_kv_heads, T, d_eff]
    else:
        idx = packed

    z_hat = sq.gather_centroids(idx).astype(mx.float32)         # [n_kv_heads, T, d_eff]

    # Pad last dim to d_pad for the inverse rotation
    if z_hat.shape[-1] != rot.d_pad:
        pad_widths = [(0, 0)] * (z_hat.ndim - 1) + [(0, rot.d_pad - z_hat.shape[-1])]
        z_hat = mx.pad(z_hat, pad_widths)

    x_hat = rot.inv(z_hat)                                       # [n_kv_heads, T, head_dim]

    # Rescale back to original magnitude
    x_hat = x_hat * scales[..., None].astype(mx.float32)
    return x_hat


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------

class TurboQuantKVCache:
    """
    Drop-in replacement for mlx_lm's KVCache.

    Usage::

        from mlx_turboquant import make_tq_cache
        from mlx_lm.generate import generate_step

        cache = make_tq_cache(model)
        for token, logprobs in generate_step(prompt, model, prompt_cache=cache):
            ...

    Compressed persistent storage: ~3.5 bpc + 2-byte scale per vector.
    Decompression is fully batched over heads and tokens via MLX ops.
    """

    step = 256  # pre-allocation granularity, matching mlx_lm KVCache

    def __init__(
        self,
        n_kv_heads: int,
        head_dim: int,
        bits: float = 3.5,
        seed: int = 0,
    ):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self._tq_bits = bits  # underscore avoids mlx_lm QuantizedKVCache detection
        self.offset = 0

        # Shared rotation + quantizer (different seeds for K vs V)
        self._rot_k = StructuredHadamardRotation(head_dim, seed=seed)
        self._sq_k  = ScalarQuantizerBeta(head_dim, bits)
        self._rot_v = StructuredHadamardRotation(head_dim, seed=seed + 1)
        self._sq_v  = ScalarQuantizerBeta(head_dim, bits)

        self._is_packed35 = abs(bits - 3.5) < 1e-6
        self._d_pad  = _next_pow2(head_dim)
        self._d_pack = self._d_pad // 2 if self._is_packed35 else self._d_pad
        self._d_eff: int = 0  # set on first compress call

        # Packed-index buffers:   [n_kv_heads, capacity, d_pack]  uint8
        # Scale buffers:          [n_kv_heads, capacity]           float16
        self._k_packed: Optional[mx.array] = None
        self._v_packed: Optional[mx.array] = None
        self._k_scales: Optional[mx.array] = None
        self._v_scales: Optional[mx.array] = None

        self._dtype = mx.bfloat16  # must match model weights

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _ensure_capacity(self, n_kv_heads: int, need: int) -> None:
        if self._k_packed is not None and self.offset + need <= self._k_packed.shape[1]:
            return
        new_steps = ((need + self.step - 1) // self.step) * self.step
        new_idx = mx.zeros((n_kv_heads, new_steps, self._d_pack), dtype=mx.uint8)
        new_sc  = mx.zeros((n_kv_heads, new_steps), dtype=mx.float16)
        if self._k_packed is not None:
            self._k_packed = mx.concatenate([self._k_packed, new_idx], axis=1)
            self._v_packed = mx.concatenate([self._v_packed, new_idx], axis=1)
            self._k_scales = mx.concatenate([self._k_scales, new_sc],  axis=1)
            self._v_scales = mx.concatenate([self._v_scales, new_sc],  axis=1)
        else:
            self._k_packed = new_idx
            self._v_packed = mx.zeros_like(new_idx)
            self._k_scales = new_sc
            self._v_scales = mx.zeros_like(new_sc)

    # ------------------------------------------------------------------
    # mlx_lm KVCache protocol
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        keys:   [1, n_kv_heads, L, head_dim]
        values: [1, n_kv_heads, L, head_dim]
        Returns full history [1, n_kv_heads, offset, head_dim] decompressed.
        """
        mx.eval(keys, values)
        _B, n_kv_heads, L, _hd = keys.shape

        # ── Compress new L tokens ──────────────────────────────────────
        k_new, k_sc, d_eff = _compress_batch(self._rot_k, self._sq_k, keys[0])
        v_new, v_sc, _     = _compress_batch(self._rot_v, self._sq_v, values[0])
        if self._d_eff == 0:
            self._d_eff = d_eff

        # ── Write into pre-allocated slices ───────────────────────────
        prev = self.offset
        self._ensure_capacity(n_kv_heads, L)
        self._k_packed[:, prev : prev + L, :] = k_new
        self._v_packed[:, prev : prev + L, :] = v_new
        self._k_scales[:, prev : prev + L]    = k_sc
        self._v_scales[:, prev : prev + L]    = v_sc
        mx.eval(self._k_packed, self._v_packed, self._k_scales, self._v_scales)
        self.offset += L

        # ── Decompress full history in one batched call ────────────────
        k_hist   = self._k_packed[:, : self.offset, :]
        v_hist   = self._v_packed[:, : self.offset, :]
        k_sc_all = self._k_scales[:, : self.offset]
        v_sc_all = self._v_scales[:, : self.offset]

        k_out = _decompress_batch(self._rot_k, self._sq_k, k_hist, k_sc_all, self._d_eff, self._is_packed35)
        v_out = _decompress_batch(self._rot_v, self._sq_v, v_hist, v_sc_all, self._d_eff, self._is_packed35)

        return (
            k_out[None].astype(self._dtype),   # [1, n_kv_heads, T, head_dim]
            v_out[None].astype(self._dtype),
        )

    # ------------------------------------------------------------------
    # Properties required by mlx_lm / cache protocol
    # ------------------------------------------------------------------

    def is_trimmable(self) -> bool:
        return False

    def empty(self) -> bool:
        return self.offset == 0

    def size(self) -> int:
        return self.offset

    def make_mask(self, N: int, return_array: bool = False, window_size=None):
        from mlx_lm.models.cache import create_attention_mask as _cam
        return _cam(N, self.offset, return_array, window_size)

    @property
    def nbytes(self) -> int:
        """Bytes used by the occupied portion of the compressed buffers."""
        if self._k_packed is None:
            return 0
        idx_bytes   = self.offset * self.n_kv_heads * self._d_pack * 2   # K + V uint8
        scale_bytes = self.offset * self.n_kv_heads * 2 * 2              # K + V float16
        return idx_bytes + scale_bytes

    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        pass

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        pass
