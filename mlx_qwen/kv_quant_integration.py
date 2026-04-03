from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

import mlx.core as mx

from mlx_turboquant import TurboQuantMSE, TurboQuantProd


@dataclass
class CompressedVec:
    idx_packed: Optional[mx.array]  # uint8 pairs (Pack35) or None
    d: int
    qjl_signs: Optional[mx.array] = None  # int8 {-1,+1}
    mu: Optional[mx.array] = None         # fp16/fp32
    idx: Optional[mx.array] = None        # uint16 [d] if not packed


@dataclass
class CompressedKVStore:
    d: int
    bits: float = 3.5
    unbiased: bool = True
    seed: int = 0

    def __post_init__(self):
        self.prod = TurboQuantProd(self.d, self.bits, self.seed) if self.unbiased else None
        self.mse = TurboQuantMSE(self.d, self.bits, self.seed) if not self.unbiased else self.prod.mse
        # buffers: lists of per-token compressed vectors per head
        self.K = []
        self.V = []

    def append(self, k: mx.array, v: mx.array):
        if self.unbiased:
            qk = self.prod.quantize(k)
            qv = self.prod.quantize(v)
            if "idx_packed" in qk["mse"]:
                d_k = qk["mse"].get("d", int(qk["mse"]["idx_packed"].shape[-1] * 2))
                self.K.append(CompressedVec(qk["mse"]["idx_packed"], d_k, qk["qjl_signs"], qk["mu"], idx=None))
            else:
                self.K.append(CompressedVec(None, self.d, qk["qjl_signs"], qk["mu"], idx=qk["mse"]["idx"]))
            if "idx_packed" in qv["mse"]:
                d_v = qv["mse"].get("d", int(qv["mse"]["idx_packed"].shape[-1] * 2))
                self.V.append(CompressedVec(qv["mse"]["idx_packed"], d_v, qv["qjl_signs"], qv["mu"], idx=None))
            else:
                self.V.append(CompressedVec(None, self.d, qv["qjl_signs"], qv["mu"], idx=qv["mse"]["idx"]))
        else:
            qk = self.mse.quantize(k)
            qv = self.mse.quantize(v)
            if "idx_packed" in qk:
                d_k = qk.get("d", int(qk["idx_packed"].shape[-1] * 2))
                self.K.append(CompressedVec(qk["idx_packed"], d_k))
            else:
                self.K.append(CompressedVec(None, self.d, idx=qk["idx"]))
            if "idx_packed" in qv:
                d_v = qv.get("d", int(qv["idx_packed"].shape[-1] * 2))
                self.V.append(CompressedVec(qv["idx_packed"], d_v))
            else:
                self.V.append(CompressedVec(None, self.d, idx=qv["idx"]))

    def attention_scores(self, q: mx.array) -> mx.array:
        # Return attention scores vs all past keys in store using unbiased estimator if enabled
        if self.unbiased:
            # Assemble a batched qobj for keys
            has_packed = all(kv.idx_packed is not None for kv in self.K)
            has_idx = all(kv.idx is not None for kv in self.K)
            if not (has_packed or has_idx):
                raise ValueError("Mixed or empty KV encoding not supported in dummy runner")
            sgn_arrs = [mx.array(kv.qjl_signs) for kv in self.K]
            mu_arrs = [mx.array(kv.mu) for kv in self.K]
            if has_packed:
                idx_arrs = [mx.array(kv.idx_packed) for kv in self.K]
                mse = {"idx_packed": mx.stack(idx_arrs, axis=0), "d": self.d}
            else:
                idx_arrs = [mx.array(kv.idx) for kv in self.K]
                mse = {"idx": mx.stack(idx_arrs, axis=0)}
            qobjs = {"mse": mse,
                     "qjl_signs": mx.stack(sgn_arrs, axis=0),
                     "mu": mx.stack(mu_arrs, axis=0)}
            return self.prod.estimate_inner_product(qobjs, q)
        else:
            # Full dequant for MSE path
            z_hats = []
            for kv in self.K:
                z_hat = self.mse.dequantize_rotated({"idx_packed": kv.idx_packed, "d": kv.d})
                z_hats.append(z_hat)
            Z = mx.stack(z_hats)  # [T, d]
            y_rot = self.mse.rot.rotate(q)
            return mx.sum(Z * y_rot, axis=-1)
