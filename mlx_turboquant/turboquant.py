"""
MLX implementation of TurboQuant for KV-cache compression.

This module provides:
- StructuredHadamardRotation: random sign + FWHT rotation with optional padding.
- ScalarQuantizerBeta: Beta-prior scalar quantizer (boundaries + centroids).
- TurboQuantMSE: MSE-optimized per-coordinate quantization.
- TurboQuantProd: adds a 1-bit QJL residual for unbiased inner products.

The design targets head_dim ~ 64/128/160 padded to power of two.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn  # noqa: F401  (may be useful later)
except Exception as e:  # pragma: no cover
    raise RuntimeError("MLX is required for mlx_turboquant; install ml-explore/mlx.") from e


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _fwht_inplace(x: mx.array) -> mx.array:
    """In-place FWHT along last dimension; assumes length is a power of two.
    Returns x transformed (not normalized).
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        # reshape pairs of size 2h
        x_reshaped = x.reshape((*x.shape[:-1], n // (2 * h), 2, h))
        a = x_reshaped[..., 0, :]
        b = x_reshaped[..., 1, :]
        x_reshaped[..., 0, :] = a + b
        x_reshaped[..., 1, :] = a - b
        x = x_reshaped.reshape((*x.shape[:-1], n))
        h *= 2
    return x


@dataclass
class StructuredHadamardRotation:
    d: int
    seed: int = 0

    def __post_init__(self):
        self.d_pad = _next_pow2(self.d)
        # Rademacher ±1 mask
        rng = np.random.default_rng(self.seed)
        self.signs = mx.array(rng.choice([-1.0, 1.0], size=(self.d_pad,), replace=True))
        self.norm = math.sqrt(self.d_pad)

    def rotate(self, x: mx.array) -> mx.array:
        """Apply z = H D x with padding if needed; normalized so ‖x‖≈‖z‖."""
        if x.shape[-1] != self.d:
            raise ValueError(f"Expected last dim {self.d}, got {x.shape[-1]}")
        if self.d_pad != self.d:
            pad = self.d_pad - self.d
            x = mx.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, pad),)) if x.ndim > 1 else mx.pad(x, ((0, pad),))
        x = x * self.signs  # D
        x = _fwht_inplace(x)  # H (unnormalized)
        return x / self.norm

    def inv(self, z: mx.array) -> mx.array:
        """Apply x̂ = D H z / √d_pad and crop to d."""
        if z.shape[-1] != self.d_pad:
            raise ValueError(f"Expected last dim {self.d_pad}, got {z.shape[-1]}")
        x = _fwht_inplace(z) / self.norm
        x = x * self.signs
        return x[..., : self.d]


def _beta_pdf_unnorm(x: mx.array, d: int) -> mx.array:
    # Coordinate distribution on the sphere: proportional to (1 - x^2)^{(d-3)/2} on [-1,1]
    alpha = (d - 1) / 2.0
    return mx.maximum(1.0 - x * x, 0.0) ** (alpha - 1.0)


@dataclass
class ScalarQuantizerBeta:
    d: int
    bits: float  # allow 2.5, 3.5, 4.0
    grid_size: int = 4096
    refine_steps: int = 3

    def __post_init__(self):
        if self.bits <= 0:
            raise ValueError("bits must be > 0")
        # Effective levels: for 3.5 bpc we target 11 roughly equal-mass bins
        if abs(self.bits - 3.5) < 1e-6:
            self.n_levels = 11
        elif abs(self.bits - 2.5) < 1e-6:
            self.n_levels = 6  # ~2.58 bits
        else:
            self.n_levels = int(2 ** round(self.bits))

        grid = mx.linspace(-1.0, 1.0, self.grid_size)
        pdf = _beta_pdf_unnorm(grid, self.d)
        cdf = mx.cumsum(pdf)
        cdf = cdf / cdf[-1]

        # equal-mass boundaries between levels (searchsorted replacement)
        targets = mx.linspace(1 / self.n_levels, 1 - 1 / self.n_levels, self.n_levels - 1)
        # idx_k = count of cdf elements < target_k
        idx = mx.sum(cdf[None, :] < targets[:, None], axis=-1).astype(mx.int32)
        self.boundaries = grid[idx]

        # initialize centroids as midpoints; refine by Lloyd steps on prior
        edges = mx.concatenate([mx.array([-1.0]), self.boundaries, mx.array([1.0])], axis=0)
        cents = 0.5 * (edges[:-1] + edges[1:])
        # refinement: weighted mean under prior per bin using the grid
        for _ in range(self.refine_steps):
            # assign each grid point to a bin
            # bin index for each grid value: count of boundaries less than grid point
            bin_idx = mx.sum(grid[:, None] > self.boundaries[None, :], axis=-1).astype(mx.int32)
            new_c = []
            for k in range(self.n_levels):
                mask = bin_idx == k
                w = pdf * mask
                denom = mx.sum(w) + 1e-12
                num = mx.sum(grid * w)
                new_c.append(num / denom)
            cents = mx.stack(new_c)
        self.centroids = cents.astype(mx.float32)

    def encode_indices(self, z: mx.array) -> mx.array:
        # z in [-1,1] after rotation normalization. Compute bin index per value.
        # idx_j = number of boundaries less than z_j
        idx = mx.sum(z[..., None] > self.boundaries[None, :], axis=-1)
        return idx.astype(mx.uint16)

    def gather_centroids(self, idx: mx.array) -> mx.array:
        return self.centroids[idx]


class _Pack35:
    """Pack two base-11 indices (0..10) into a single uint8 using base-11 pairing.
    Each pair value = a + 11*b fits in [0,120] < 2^7, so 1 byte per pair → 3.5 bpc.
    """

    levels = 11

    @staticmethod
    def pack(idx: mx.array) -> Tuple[mx.array, int]:
        # idx shape [..., d], values in [0,10]
        d = idx.shape[-1]
        if d % 2 != 0:
            # pad with zero to even length
            idx = mx.concatenate([idx, mx.zeros_like(idx[..., :1])], axis=-1)
            d += 1
        a = idx[..., 0::2]
        b = idx[..., 1::2]
        packed = (a + b * 11).astype(mx.uint8)  # 0..120
        return packed, d

    @staticmethod
    def unpack(packed: mx.array, d: int) -> mx.array:
        b = (packed // 11).astype(mx.uint16)
        a = (packed - b * 11).astype(mx.uint16)
        idx_pairs = mx.stack([a, b], axis=-1).reshape((*packed.shape[:-1], -1))
        return idx_pairs[..., :d]


@dataclass
class TurboQuantMSE:
    d: int
    bits: float = 3.5
    seed: int = 0

    def __post_init__(self):
        self.rot = StructuredHadamardRotation(self.d, self.seed)
        self.q = ScalarQuantizerBeta(self.d, self.bits)

    def quantize(self, x: mx.array) -> dict:
        z = self.rot.rotate(x)
        idx = self.q.encode_indices(z)
        if abs(self.q.bits - 3.5) < 1e-6:
            idx_packed, d_eff = _Pack35.pack(idx)
            return {"idx_packed": idx_packed, "d": d_eff}
        else:
            return {"idx": idx}

    def dequantize_rotated(self, qobj: dict) -> mx.array:
        if "idx_packed" in qobj:
            idx = _Pack35.unpack(qobj["idx_packed"], qobj["d"])
        else:
            idx = qobj["idx"]
        z_hat = self.q.gather_centroids(idx)
        return z_hat

    def dequantize(self, qobj: dict) -> mx.array:
        z_hat = self.dequantize_rotated(qobj)
        # ensure last dim equals rot.d_pad for inverse
        if z_hat.shape[-1] != self.rot.d_pad:
            pad = self.rot.d_pad - z_hat.shape[-1]
            z_hat = mx.pad(z_hat, ((0, 0),) * (z_hat.ndim - 1) + ((0, pad),)) if z_hat.ndim > 1 else mx.pad(z_hat, ((0, pad),))
        x_hat = self.rot.inv(z_hat)
        return x_hat


@dataclass
class TurboQuantProd:
    d: int
    bits: float = 3.5  # total; uses (bits-1) for MSE + 1 bit for QJL
    seed: int = 0

    def __post_init__(self):
        # MSE with (b-1) bits approximated by adjusting n_levels
        b_mse = max(self.bits - 1.0, 1.0)
        self.mse = TurboQuantMSE(self.d, b_mse, self.seed)
        self.rot = self.mse.rot  # share rotation
        # QJL structured projector shares Hadamard with different sign mask
        rng = np.random.default_rng(self.seed + 1)
        self.qjl_signs = mx.array(rng.choice([-1.0, 1.0], size=(self.rot.d_pad,), replace=True))

    def quantize(self, x: mx.array) -> dict:
        # First stage
        q_mse = self.mse.quantize(x)
        # Compute residual sign(S r) without forming r explicitly:
        # r = x - x_hat_mse; we approximate x_hat_mse via rotated z_hat centroids.
        z = self.rot.rotate(x)
        z_hat = self.mse.dequantize_rotated(q_mse)
        r_rot = z - z_hat
        # S is sign mask + FWHT; use separate sign pattern
        s_r = r_rot * self.qjl_signs
        s_r = _fwht_inplace(s_r) / self.rot.norm
        signs = mx.sign(s_r)  # {-1, +1}
        mu = mx.mean(mx.abs(s_r), axis=-1, keepdims=False)
        return {"mse": q_mse, "qjl_signs": signs.astype(mx.int8), "mu": mu}

    def estimate_inner_product(self, qobj: dict, y: mx.array) -> mx.array:
        # MSE part: <y, x_hat_mse>
        y_rot = self.rot.rotate(y)  # [..., d_pad]
        z_hat = self.mse.dequantize_rotated(qobj["mse"])  # [..., d]
        # Pad z_hat to d_pad for dot product if needed
        if z_hat.shape[-1] != y_rot.shape[-1]:
            pad = y_rot.shape[-1] - z_hat.shape[-1]
            z_hat = mx.pad(z_hat, ((0, 0),) * (z_hat.ndim - 1) + ((0, pad),)) if z_hat.ndim > 1 else mx.pad(z_hat, ((0, pad),))
        ip_mse = mx.sum(y_rot * z_hat, axis=-1)
        # QJL correction: (mu/d) * sum_i sign_i * t_i, where t = S y
        t = y_rot * self.qjl_signs
        t = _fwht_inplace(t) / self.rot.norm
        # Broadcast t (length d_pad) against per-token signs [T, d_pad]
        correction = (qobj["mu"] / self.rot.d_pad) * mx.sum(qobj["qjl_signs"] * t, axis=-1)
        return ip_mse + correction
