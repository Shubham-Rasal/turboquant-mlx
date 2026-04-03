from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import math
import mlx.core as mx

from .attention_patch import KVCacheToggle


@dataclass
class QwenAttentionPatcher:
    """
    Minimal, non-invasive wrapper to add TurboQuant KV caching to an MLX Qwen
    attention block at decode time, behind a flag.

    Expectations for `attn` (common in Qwen-style modules):
      - callable linear projections: `q_proj(x)`, `k_proj(x)`, `v_proj(x)`, `o_proj(x)`
      - attributes: `num_heads`, `head_dim`, and optional `num_kv_heads` (defaults to num_heads)

    This patcher does NOT monkey-patch `forward`; instead it attaches an attribute
    `attn.quant_step(x: mx.array) -> mx.array` that you can call from your decode loop
    (token-by-token). Keep your original path for unquantized mode if desired.
    """

    attn: object
    use_quant: bool = True
    bits: float = 3.5
    unbiased: bool = True
    seed: int = 0

    def __post_init__(self):
        # Discover shapes (support multiple attribute names)
        nh = getattr(self.attn, "num_heads", None) or getattr(self.attn, "n_heads", None)
        hd = getattr(self.attn, "head_dim", None) or getattr(self.attn, "hidden_size_per_head", None)
        # For MLX Qwen2 Attention, head_dim is typically q_proj.weight.shape[1] // n_heads
        if hd is None and hasattr(self.attn, "q_proj"):
            try:
                w = self.attn.q_proj.weight
                # q_proj maps hidden_size -> n_heads*head_dim (1D weight per MLX Linear)
                out = w.shape[0]
                if nh is None and hasattr(self.attn, "n_heads"):
                    nh = int(getattr(self.attn, "n_heads"))
                if nh:
                    hd = int(out // int(nh))
            except Exception:
                pass
        nkv = getattr(self.attn, "num_kv_heads", None) or getattr(self.attn, "n_kv_heads", None) or nh
        if nh is None or hd is None:
            raise AttributeError("Attention module missing num_heads/head_dim (or equivalents)")
        self.nh = int(nh)
        self.hd = int(hd)
        self.nkv = int(nkv)

        # Projections: support separate q/k/v/o or fused wqkv + wo
        self.q_proj: Callable[[mx.array], mx.array]
        self.k_proj: Callable[[mx.array], mx.array]
        self.v_proj: Callable[[mx.array], mx.array]
        self.o_proj: Callable[[mx.array], mx.array]

        if hasattr(self.attn, "q_proj") and hasattr(self.attn, "k_proj") and hasattr(self.attn, "v_proj"):
            self.q_proj = self.attn.q_proj
            self.k_proj = self.attn.k_proj
            self.v_proj = self.attn.v_proj
            self.o_proj = getattr(self.attn, "o_proj", getattr(self.attn, "wo", None))
            if self.o_proj is None:
                raise AttributeError("Attention module missing o_proj/wo")
        elif hasattr(self.attn, "wqkv") and (hasattr(self.attn, "wo") or hasattr(self.attn, "o_proj")):
            wqkv = self.attn.wqkv
            self.o_proj = getattr(self.attn, "wo", getattr(self.attn, "o_proj"))

            def split_qkv(x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
                qkv = wqkv(x)  # shape [hidden]
                total = qkv.shape[-1]
                # Expect nh*hd + 2*nkv*hd
                q_sz = self.nh * self.hd
                kv_sz = self.nkv * self.hd
                if q_sz + 2 * kv_sz != total:
                    # fallback: equal thirds
                    third = total // 3
                    q = qkv[..., :third]
                    k = qkv[..., third : 2 * third]
                    v = qkv[..., 2 * third : 3 * third]
                else:
                    q = qkv[..., :q_sz]
                    k = qkv[..., q_sz : q_sz + kv_sz]
                    v = qkv[..., q_sz + kv_sz : q_sz + 2 * kv_sz]
                return q, k, v

            self.q_proj = lambda x: split_qkv(x)[0]
            self.k_proj = lambda x: split_qkv(x)[1]
            self.v_proj = lambda x: split_qkv(x)[2]
        else:
            raise AttributeError("Unsupported attention projections: expected q/k/v/o or fused wqkv + wo")

        # Make one KV toggle per KV head
        self.kv: List[KVCacheToggle] = [
            KVCacheToggle(d=self.hd, use_quant=self.use_quant, bits=self.bits, unbiased=self.unbiased, seed=self.seed + i)
            for i in range(self.nkv)
        ]
        self.scale = 1.0 / math.sqrt(self.hd)
        # Expose kv stores on the attention module for external accounting
        setattr(self.attn, "tq_kvstores", self.kv)

        def quant_step(x: mx.array) -> mx.array:
            """
            x: [hidden_size] — single-token hidden state for this layer
            Returns: [hidden_size] attention output for this layer using the
            requested KV caching mode (quantized or fp path).
            """
            # Linear projections
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            # Reshape to heads
            q = q.reshape(self.nh, self.hd)
            # For grouped-KV, share k/v among groups
            kv_ratio = self.nh // self.nkv
            k = k.reshape(self.nkv, kv_ratio, self.hd)
            v = v.reshape(self.nkv, kv_ratio, self.hd)

            # Compute per-head outputs
            head_outs = []
            for h in range(self.nh):
                kvh = h // kv_ratio
                # average the group k/v to a single vector for the new token
                k_h = mx.mean(k[kvh], axis=0)
                v_h = mx.mean(v[kvh], axis=0)
                out_h = self.kv[kvh].step(q[h], k_h, v_h, self.scale)
                head_outs.append(out_h)

            y = mx.concatenate(head_outs, axis=0)  # [hidden_size]
            y = self.o_proj(y)
            return y

        # Attach as a method-like attribute for the caller to use
        setattr(self.attn, "quant_step", quant_step)


def patch_qwen_block(attn_module, use_quant: bool = True, bits: float = 3.5, unbiased: bool = True, seed: int = 0) -> QwenAttentionPatcher:
    """Convenience: create and attach a `quant_step` callable to an attention module."""
    return QwenAttentionPatcher(attn=attn_module, use_quant=use_quant, bits=bits, unbiased=unbiased, seed=seed)
