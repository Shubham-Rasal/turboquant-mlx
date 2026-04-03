from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from .kv_quant_integration import CompressedKVStore


@dataclass
class KVCacheToggle:
    """
    Minimal attention helper that toggles between full-precision KV and
    TurboQuant-based compressed KV at decode time.

    This does not monkey-patch a specific Qwen class; instead it provides a
    step() method you can call from an attention wrapper. It is designed to be
    simple to wire into an MLX Qwen attention implementation where you already
    have per-step query/key/value tensors.
    """

    d: int
    use_quant: bool = True
    bits: float = 3.5
    unbiased: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.use_quant:
            self.qstore = CompressedKVStore(d=self.d, bits=self.bits, unbiased=self.unbiased, seed=self.seed)
            self.k_full = None
            self.v_full = None
        else:
            self.qstore = None
            self.k_full: list[mx.array] = []
            self.v_full: list[mx.array] = []

    def step(self, q: mx.array, k: mx.array, v: mx.array, scale: Optional[float] = None) -> mx.array:
        """
        One decode step.
        - q: [d]
        - k: [d]
        - v: [d]
        - scale: optional softmax scale (e.g., 1/sqrt(d))
        Returns: attention output vector [d]
        """
        if self.use_quant:
            # Append compressed KV and compute scores via unbiased estimator
            self.qstore.append(k, v)
            scores = self.qstore.attention_scores(q)  # [T]
            if scale is not None:
                scores = scores * scale
            weights = mx.softmax(scores, axis=-1)  # [T]
            # Dequantize values in one go (MSE decoding is fine for aggregation)
            # Build stacked dequantized V
            V_list = []
            for cv in self.qstore.V:
                if cv.idx_packed is not None:
                    z_hat = self.qstore.mse.dequantize_rotated({"idx_packed": cv.idx_packed, "d": cv.d})
                else:
                    z_hat = self.qstore.mse.dequantize_rotated({"idx": cv.idx})
                # Pad and invert rotation
                if z_hat.shape[-1] != self.qstore.mse.rot.d_pad:
                    pad = self.qstore.mse.rot.d_pad - z_hat.shape[-1]
                    z_hat = mx.pad(z_hat, ((0, pad),))
                v_hat = self.qstore.mse.rot.inv(z_hat)  # [d]
                V_list.append(v_hat)
            V = mx.stack(V_list, axis=0) if len(V_list) else mx.zeros((0, self.d))
            out = weights @ V  # [d]
            return out
        else:
            # Full-precision fallback path
            self.k_full.append(k)
            self.v_full.append(v)
            K = mx.stack(self.k_full, axis=0)  # [T, d]
            V = mx.stack(self.v_full, axis=0)  # [T, d]
            scores = (K @ q).reshape((-1,))  # [T]
            if scale is not None:
                scores = scores * scale
            weights = mx.softmax(scores, axis=-1)
            out = weights @ V
            return out


def example_usage():  # pragma: no cover
    d = 128
    scale = 1.0 / (d ** 0.5)
    kv = KVCacheToggle(d=d, use_quant=True, bits=3.5, unbiased=True)
    q = mx.random.normal(shape=(d,))
    for _ in range(8):
        k = mx.random.normal(shape=(d,))
        v = mx.random.normal(shape=(d,))
        _ = kv.step(q, k, v, scale)

