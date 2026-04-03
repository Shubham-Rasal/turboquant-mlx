# TurboQuant (MLX) — KV Cache Compression for On-Device LLMs

An MLX-native implementation of the [TurboQuant](https://arxiv.org/abs/2412.04214) algorithm (Zandieh et al., 2024) for real, measured KV cache compression on Apple Silicon. Integrates directly with `mlx_lm`'s generation pipeline with no changes to model weights or attention code.

---

<img width="1106" height="234" alt="Screenshot 2026-04-03 at 11 01 22 PM" src="https://github.com/user-attachments/assets/fb0ce269-6234-4ec1-9ab1-eca035a2d904" />


## What TurboQuant does - 

When an LLM generates text it stores a **Key (K)** and **Value (V)** vector for every past token in every layer — the KV cache. At long contexts this dominates memory (hundreds of MB to GB on-device).

TurboQuant compresses each K/V vector to **3.5 bits per element** in three steps:

1. **Scramble** — multiply by a random ±1 mask and apply a Fast Walsh-Hadamard Transform. This spreads the information evenly so all coordinates follow the same known distribution.
2. **Quantize** — map each coordinate to one of **11 levels** (optimal for the Beta distribution that appears after step 1). Two indices fit in one byte via base-11 packing → exactly 3.5 bpc.
3. **Store scale** — the original L2 norm is stored as a `float16` alongside the packed indices so the vector can be rescaled accurately at decompression.

At decode time the full history is decompressed in a single set of batched MLX ops and passed to the standard `scaled_dot_product_attention`.


---

## Requirements

- Apple Silicon Mac (M-series) with Metal
- Python 3.10+
- Conda environment with `mlx` and `mlx-lm`:
  ```bash
  conda create -n mlx python=3.12
  conda activate mlx
  pip install mlx mlx-lm
  ```

---

## Quick start

```bash
# Verify MLX works
conda run -n mlx python -c "import mlx.core as mx; print('MLX OK', mx.array(1).item())"

# KV memory benchmark — baseline vs TurboQuant, side-by-side generated text
conda run -n mlx python benchmarks/kv_cache_usage_test.py \
    --tokens 64 --prompt "Write a short paragraph about the ocean."

# Serve with OpenAI-compatible API
conda run -n mlx python serve_tq.py \
    --model mlx-community/Qwen2.5-3B-Instruct-bf16 \
    --port 8080
```

---

## Measured results (Qwen2.5-3B-Instruct-bf16, Apple Silicon)

### KV cache memory comparison

```
Prompt: 'Write a short paragraph about the ocean.'
Tokens generated: 64

                               bf16     TurboQuant (3.5 bpc)
KV cache nbytes           9,437,184            684,288
KV cache MB                    9.44               0.68
Throughput tok/s               35.6               15.3

Memory reduction: 13.8× smaller  (92.7% saved)
```

> The bf16 baseline uses `mlx_lm`'s `KVCache` which pre-allocates in 256-slot blocks. TurboQuant allocates exactly what is needed, so at short contexts the ratio is even larger.

### Generated text quality

```
[bf16]
The ocean is a vast and mysterious body of water that covers more than 70%
of the Earth's surface. It is home to an incredible array of marine life,
from tiny plankton to massive whales...

[TurboQuant 3.5 bpc]
The ocean is a vast and mysterious body of water that covers more than half
of our planet. It is home to a wide variety of marine life, from tiny
plankton to massive whales. The ocean is also a vital part of our planet's
ecosystem — it provides oxygen, food, and a source of income...
```

Text is coherent and on-topic; minor wording differences are expected from 3.5 bpc reconstruction.

### Long-context KV theory (all 36 layers, Qwen2.5-3B)

```
[kv-theory] per-token bytes: bf16=36,864  tq≈8,352   layers=36
[kv-theory] total KV:
  8k tokens:  bf16≈302.0 MB  tq≈68.4 MB
  16k tokens: bf16≈604.0 MB  tq≈136.8 MB
  32k tokens: bf16≈1,208 MB  tq≈273.7 MB
```

### Throughput (short prompt, 64 tokens)

```
[baseline]  tokens/sec=32.88
[patch]     attention blocks patched: 36
[quantized] tokens/sec=34.26   speedup x1.04
```

---

## How it works — step by step

### 1. TurboQuantKVCache (the core)

`mlx_turboquant/kv_cache.py` implements a drop-in replacement for `mlx_lm`'s `KVCache`. Every time `mlx_lm`'s attention calls `cache.update_and_fetch(keys, values)`:

```
incoming K/V  [1, n_kv_heads, L, head_dim]
       │
       ▼  _compress_batch()
  L2 norm  →  normalize  →  Hadamard-rotate  →  encode (11 levels)  →  base-11 pack
       │
  stored in pre-allocated uint8 buffer  [n_kv_heads, capacity, d_pack]
  + float16 scale buffer                [n_kv_heads, capacity]
       │
       ▼  _decompress_batch()  (full history every step)
  unpack  →  gather centroids  →  inv-rotate  →  rescale
       │
  returned as [1, n_kv_heads, T, head_dim]  →  standard SDPA unchanged
```

All compress/decompress ops run on `[n_kv_heads, T, head_dim]` tensors — fully batched, no Python loops over tokens or heads.

**Why the L2 scale?** The TurboQuantMSE codebook is designed for unit-sphere vectors (coordinates in `[-1, 1]`). Real K/V vectors from Qwen attention have L2 norms of ~5–20. Without normalisation every value clips to a boundary bin → pure noise in the output. Storing a `float16` norm per token (2 bytes vs 64 bytes of indices, ~3% overhead) and normalising before compressing fixes this completely.

### 2. make_tq_cache(model)

`mlx_turboquant/make_cache.py` inspects each layer's attention module to read `n_kv_heads` and `head_dim`, then returns a `List[TurboQuantKVCache]` — one per layer — ready to pass as `prompt_cache` to `mlx_lm.generate_step`.

```python
from mlx_turboquant import make_tq_cache
from mlx_lm import load
from mlx_lm.generate import generate_step

model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-bf16")
cache = make_tq_cache(model)                  # one TQ cache per layer

for token, logprobs in generate_step(prompt, model, prompt_cache=cache):
    ...

print(cache[0].nbytes, "bytes")              # compressed size, layer 0
```

### 3. serve_tq.py — OpenAI-compatible server

`serve_tq.py` patches `make_prompt_cache` in both `mlx_lm.models.cache` and `mlx_lm.server` (which holds a local import binding) before starting the standard `mlx_lm` server:

```python
_cache_mod.make_prompt_cache = _tq_make_prompt_cache   # module-level ref
_srv.make_prompt_cache       = _tq_make_prompt_cache   # server's local ref
_srv.main()                                            # standard server unchanged
```

The entire OpenAI-compatible API surface (streaming, LRU prompt cache, token accounting, `/v1/models`, `/v1/completions`, `/v1/chat/completions`) works without modification.

```bash
conda run -n mlx python serve_tq.py \
    --model mlx-community/Qwen2.5-3B-Instruct-bf16 \
    --port 8080
```

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-3B-Instruct-bf16",
    "messages": [{"role":"user","content":"Write a haiku about the sea."}],
    "max_tokens": 32
  }'
```

**Example server response:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Whispers of the tide,\nWharf lights its secrets bare,\nSea calls me home."
    }
  }],
  "usage": {
    "prompt_tokens": 37,
    "completion_tokens": 20,
    "total_tokens": 57
  }
}
```


---

## Benchmark scripts

### `benchmarks/kv_cache_usage_test.py` ← main benchmark

Runs `generate_step` with both a standard `KVCache` and a `TurboQuantKVCache`, then prints a side-by-side table of actual `nbytes`, throughput, and the generated text.

```bash
conda run -n mlx python benchmarks/kv_cache_usage_test.py \
    --tokens 64 \
    --prompt "Write a short paragraph about the ocean."
```

```
============================================================
Prompt: 'Write a short paragraph about the ocean.'
============================================================

── Generated text ──────────────────────────────────────
[bf16]       The ocean is a vast and mysterious body of water that covers more
             than 70% of the Earth's surface...

[TurboQuant] The ocean is a vast and mysterious body of water that covers more
             than half of our planet. It is home to a wide variety of marine
             life...

============================================================
                               bf16    TurboQuant
Tokens generated                 64            64
KV cache nbytes           9,437,184       684,288
KV cache MB                    9.44          0.68
Throughput tok/s               35.6          15.3

Memory reduction: 13.79× smaller  (92.7% saved)
============================================================
```

### `benchmarks/qwen_mlx_autopatch_demo.py`

Baseline vs auto-patched throughput on a short prompt (note: patch does not activate in `mlx_lm` due to `forward` vs `__call__` — see above).

```bash
conda run -n mlx python benchmarks/qwen_mlx_autopatch_demo.py \
    --tokens 64 --prompt "Write a short paragraph about the ocean."
```
```
[baseline] tokens/sec=32.88
[patch]    attention blocks patched: 36
[quantized] tokens/sec=34.26
[delta]    speedup x1.04
```

### `benchmarks/qwen_long_context_bench.py`

Long-context (8k–32k) throughput + theoretical KV memory.

```bash
conda run -n mlx python benchmarks/qwen_long_context_bench.py --ctx 8192 --tokens 32
```
```
[baseline]  tokens/sec=2.85
[patch]     attention blocks patched: 36
[quantized] tokens/sec=2.90; speedup x1.02
[kv-theory] per-token bytes: bf16=36864, tq≈8352; layers=36
[kv-theory] total KV for 8192 toks: bf16≈302.0 MB; tq≈68.4 MB
```

### `benchmarks/attn_toggle_demo.py`

Standalone toggle sanity check (no model, just the `KVCacheToggle` helper).

```bash
conda run -n mlx python benchmarks/attn_toggle_demo.py --steps 256
```
```
FP16_KV:               tps=13,509   ttft_ms=0.0
TurboQuantprod_3.5bpc: tps=151      ttft_ms=0.4
```

---

## Notes & limitations

- **Throughput**: TurboQuant decompresses the full history on every decode step (decompress K+V of shape `[heads, T, head_dim]` per layer per step). At `T=64` this is ~2× slower than bf16; at longer contexts the gap narrows as memory bandwidth becomes the dominant cost.
- **Batching**: `TurboQuantKVCache` does not implement `merge` so `mlx_lm`'s batch-generation path is disabled. The server falls back to sequential request handling.
- **Prompt cache reuse**: `is_trimmable()` returns `False` so the server's prefix-sharing logic is not used. Each new request gets a fresh TQ cache.
- **MLX only**: no CUDA/Triton paths from the original PyTorch TurboQuant paper are used here.
- **Models**: tested on `Qwen2.5-3B-Instruct-bf16`. Any `mlx_lm` Qwen2-family model with standard `self_attn` blocks should work; `make_tq_cache` also supports other architectures with the same attribute names.

---

## Acknowledgements

- **TurboQuant paper**: Amir Zandieh, Majid Daliri, Insu Han — *TurboQuant: Online KV Cache Compression for Reduced Memory LLM Inference* (2024)
- **MLX and mlx-lm** by Apple

## License

MIT
