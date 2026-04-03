"""
Fused Metal kernels for TurboQuant KV compress/decompress (3.5 bpc, base-11 pack).

Uses mx.fast.metal_kernel with one threadgroup per (head, token) vector and
threadgroup shared memory for FWHT butterflies + L2 norm reduction.

Falls back to pure-MLX Python when Metal is unavailable (HAS_METAL_KERNELS=False).
"""
from __future__ import annotations

import math
from typing import Tuple

try:
    import mlx.core as mx
except ImportError as e:  # pragma: no cover
    raise RuntimeError("MLX is required for mlx_turboquant.") from e

_QUANT_KERNEL_CACHE: dict[tuple, object] = {}
_DEQUANT_KERNEL_CACHE: dict[tuple, object] = {}


def _metal_supported() -> bool:
    """True when MLX is using the GPU backend (Metal on Apple Silicon)."""
    try:
        return "gpu" in str(mx.default_device()).lower()
    except Exception:
        return False


# False on CPU-only MLX; callers fall back to pure-MLX Python ops.
HAS_METAL_KERNELS: bool = _metal_supported()


def _build_dequant_source(d_pad: int, d_pack: int, d_orig: int) -> str:
    inv_sqrt_d = 1.0 / math.sqrt(float(d_pad))
    return f"""
    uint vec_idx = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    threadgroup float shmem[{d_pad}];

    uint pair = lane / 2u;
    uint b_byte = packed[vec_idx * {d_pack}u + pair];
    uint idx = (lane % 2u == 0u) ? (b_byte % 11u) : (b_byte / 11u);
    float c = centroids[idx];
    shmem[lane] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 1u; h < {d_pad}u; h <<= 1u) {{
        if ((lane & h) == 0u) {{
            float a = shmem[lane];
            float b = shmem[lane ^ h];
            shmem[lane] = a + b;
            shmem[lane ^ h] = a - b;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float val = shmem[lane] * float({inv_sqrt_d}) * signs[lane];
    float sc = float(scales[vec_idx]);
    if (lane < {d_orig}u) {{
        out[vec_idx * {d_orig}u + lane] = val * sc;
    }}
    """


def _build_quant_source(d_pad: int, d_pack: int, d_orig: int, n_bound: int) -> str:
    inv_sqrt_d = 1.0 / math.sqrt(float(d_pad))
    return f"""
    uint vec_idx = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    threadgroup float shmem[{d_pad}];

    float x0 = (lane < {d_orig}u) ? x[vec_idx * {d_orig}u + lane] : 0.0f;
    shmem[lane] = x0 * x0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = {d_pad}u / 2u; s > 0u; s >>= 1u) {{
        if (lane < s) {{
            shmem[lane] += shmem[lane + s];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float norm = sqrt(shmem[0]) + 1e-8f;
    x0 = (lane < {d_orig}u) ? x[vec_idx * {d_orig}u + lane] : 0.0f;
    shmem[lane] = (x0 / norm) * signs[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 1u; h < {d_pad}u; h <<= 1u) {{
        if ((lane & h) == 0u) {{
            float a = shmem[lane];
            float b = shmem[lane ^ h];
            shmem[lane] = a + b;
            shmem[lane ^ h] = a - b;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float z = shmem[lane] * float({inv_sqrt_d});
    uint idx = 0u;
    for (uint k = 0u; k < {n_bound}u; k++) {{
        idx += (z > boundaries[k]) ? 1u : 0u;
    }}

    threadgroup ushort enc[{d_pad}];
    enc[lane] = ushort(idx);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if ((lane % 2u == 0u) && (lane < {d_pad}u)) {{
        uint pair = lane / 2u;
        if (pair < {d_pack}u) {{
            ushort a = enc[lane];
            ushort b = enc[lane + 1u];
            packed_out[vec_idx * {d_pack}u + pair] = uchar(a + b * 11u);
        }}
    }}

    if (lane == 0u) {{
        scales_out[vec_idx] = half(norm);
    }}
    """


def _get_dequant_kernel(d_pad: int, d_pack: int, d_orig: int):
    key = ("dequant", d_pad, d_pack, d_orig)
    if key not in _DEQUANT_KERNEL_CACHE:
        src = _build_dequant_source(d_pad, d_pack, d_orig)
        _DEQUANT_KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"tq_dequant_d{d_pad}",
            input_names=["packed", "scales", "centroids", "signs"],
            output_names=["out"],
            source=src,
        )
    return _DEQUANT_KERNEL_CACHE[key]


def _get_quant_kernel(d_pad: int, d_pack: int, d_orig: int, n_bound: int):
    key = ("quant", d_pad, d_pack, d_orig, n_bound)
    if key not in _QUANT_KERNEL_CACHE:
        src = _build_quant_source(d_pad, d_pack, d_orig, n_bound)
        _QUANT_KERNEL_CACHE[key] = mx.fast.metal_kernel(
            name=f"tq_quant_d{d_pad}_b{n_bound}",
            input_names=["x", "boundaries", "signs"],
            output_names=["packed_out", "scales_out"],
            source=src,
        )
    return _QUANT_KERNEL_CACHE[key]


def metal_dequantize(
    packed: mx.array,
    scales: mx.array,
    centroids: mx.array,
    signs: mx.array,
    d_orig: int,
    d_pad: int,
    d_pack: int,
) -> mx.array:
    """
    Decompress packed uint8 indices + fp16 scales to float32 vectors.

    packed: [H, T, d_pack] uint8
    scales: [H, T] float16
    centroids: [n_levels] float32
    signs: [d_pad] float32 (Rademacher mask)
    """
    if not HAS_METAL_KERNELS:
        raise RuntimeError("Metal dequantize not available")

    n_h, t, _ = packed.shape
    vec_count = n_h * t
    total_threads = vec_count * d_pad
    kernel = _get_dequant_kernel(d_pad, d_pack, d_orig)
    out = kernel(
        inputs=[packed, scales, centroids.reshape(-1), signs.reshape(-1)],
        grid=(total_threads, 1, 1),
        threadgroup=(d_pad, 1, 1),
        output_shapes=[(n_h, t, d_orig)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def metal_quantize(
    x: mx.array,
    boundaries: mx.array,
    signs: mx.array,
    d_orig: int,
    d_pad: int,
    d_pack: int,
) -> Tuple[mx.array, mx.array]:
    """
    Compress float32 vectors to packed uint8 + fp16 scales.

    x: [H, L, d_orig] float32 (caller should cast)
    boundaries: [n_levels-1] float32
    signs: [d_pad] float32
    """
    if not HAS_METAL_KERNELS:
        raise RuntimeError("Metal quantize not available")

    n_h, ell, _ = x.shape
    vec_count = n_h * ell
    total_threads = vec_count * d_pad
    n_bound = int(boundaries.shape[0])
    kernel = _get_quant_kernel(d_pad, d_pack, d_orig, n_bound)
    packed_out, scales_out = kernel(
        inputs=[x, boundaries.reshape(-1), signs.reshape(-1)],
        grid=(total_threads, 1, 1),
        threadgroup=(d_pad, 1, 1),
        output_shapes=[(n_h, ell, d_pack), (n_h, ell)],
        output_dtypes=[mx.uint8, mx.float16],
    )
    return packed_out, scales_out
