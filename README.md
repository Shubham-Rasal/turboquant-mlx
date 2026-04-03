# TurboQuant (MLX) — Qwen KV Cache Compression + Benchmarks

This repo implements an MLX-compatible version of TurboQuant for online KV cache compression and integrates it with MLX Qwen models via a non-invasive auto‑patcher. It includes simple benchmarks to validate throughput and quantify KV memory savings at long contexts.

## What’s inside
- `mlx_turboquant/` — Core MLX implementation
  - Structured Hadamard rotation (FWHT + Rademacher signs)
  - Beta‑prior scalar quantizer (grid + Lloyd refinement)
  - TurboQuantMSE (per‑coordinate quantization)
  - TurboQuantProd (adds 1‑bit QJL residual for unbiased inner products)
  - 3.5 bpc base‑11 packer (2 indices per byte)
- `mlx_qwen/` — Qwen integration
  - `auto_patch.py` — Finds MLX Qwen attention blocks and attaches a quantized decode path
  - `qwen_wrapper.py` — Minimal wrapper that provides a `quant_step` for attention
  - `attention_patch.py` — Toggle helper used by the wrapper
- `benchmarks/`
  - `qwen_mlx_autopatch_demo.py` — Baseline vs TurboQuant tokens/sec on a short prompt
  - `qwen_long_context_bench.py` — Long‑context (8k–32k) throughput + KV memory
  - `attn_toggle_demo.py` — Minimal attention‑style toggle sanity check

## Requirements
- Apple Silicon with MLX + mlx‑lm (installed in your environment)
- Python 3.10+
- Conda environment (recommended) named `mlx` or adapt commands below

## Quick start
```bash
# (Optional) Use your conda environment with MLX and mlx-lm
conda run -n mlx python -c "import mlx.core as mx; print('MLX OK', mx.array(1).item())"

# 1) Short demo: baseline vs TurboQuant on Qwen 3B (first run downloads weights)
conda run -n mlx python benchmarks/qwen_mlx_autopatch_demo.py --tokens 64 --prompt "Write a short paragraph about the ocean."

# 2) Long-context bench (8k) with KV memory reporting
conda run -n mlx python benchmarks/qwen_long_context_bench.py --ctx 8192 --tokens 64

# 3) Heavier contexts (reduce tokens for quick finish)
conda run -n mlx python benchmarks/qwen_long_context_bench.py --ctx 16384 --tokens 32
conda run -n mlx python benchmarks/qwen_long_context_bench.py --ctx 32768 --tokens 16
```

## Reported results (MLX Qwen2.5‑3B‑Instruct‑bf16)
- Patched attention blocks: 36
- Bits: 3.5 bpc for K and for V (TurboQuantprod keys + MSE dequant values)

Throughput (measured)
- 8k context (+64 tokens): baseline 5.38 tok/s, TurboQuant 5.44 tok/s (x1.01)

KV cache size (theoretical, all layers)
- Per token: bf16 36,864 B vs TurboQuant ≈ 8,352 B (≈4.41× smaller)
- Totals:
  - 8k: bf16 ≈ 302.0 MB, TurboQuant ≈ 68.4 MB
  - 16k: bf16 ≈ 604.0 MB, TurboQuant ≈ 136.8 MB
  - 32k: bf16 ≈ 1,208.0 MB, TurboQuant ≈ 273.7 MB

Measured compressed KV (8k run)
- `qwen_long_context_bench.py` sums live compressed bytes across all layers/heads and prints
  `[kv-measured] ...` alongside theoretical values.

## How the auto‑patcher works
- Detects MLX Qwen attention modules (Qwen2 family), infers `head_dim` if needed, and attaches a
  `quant_step` method used only during single‑token decode.
- Monkey‑patches `forward` to call `quant_step` when inputs are single‑token (keeps original path
  for batch or training).

## Notes & limitations
- The demo uses TurboQuantprod for unbiased key scoring and MSE dequant for values (simple and robust).
- For very long contexts on 32 GB RAM, prefer fewer generated tokens or sliding‑window decode.
- Triton/CUDA paths in the original PyTorch code are not used here; this is MLX‑only.

## Acknowledgements
- TurboQuant paper: Amir Zandieh et al. (Google Research/DeepMind/NYU)
- MLX and mlx‑lm by Apple

## License
- Provide your preferred license here (e.g., Apache‑2.0 or MIT).

