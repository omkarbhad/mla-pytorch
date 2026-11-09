"""Multi-head Latent Attention (DeepSeek-V2, https://arxiv.org/abs/2405.04434) and a standard MHA baseline.

Keys and values are not cached directly. Each token is compressed to a small latent
c_kv = W_DKV h (size d_c), and keys / values are rebuilt from it when needed:

    k_C = W_UK c_kv      v_C = W_UV c_kv

Rotary embeddings cannot sit inside this low-rank path (the rotation would be stuck between W_Q and
W_UK), so position information travels in a separate small "decoupled" part: per-head rotary queries
and ONE rotary key shared by all heads. Per head, q = [q_C ; q_R] and k = [k_C ; k_R], and the
softmax scale is 1 / sqrt(d_h + d_h^R). Only c_kv and k_R are cached: d_c + d_h^R numbers per token.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MLAConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim: int, max_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start: int = 0) -> torch.Tensor:
    """Rotary embedding on the last dim; the sequence axis is dim -2."""
    t = x.shape[-2]
    c = cos[start : start + t].to(x.dtype)
    s = sin[start : start + t].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class Cache:
    """Per-layer cache of two tensors, joined along the sequence axis (dim -2).

    MHA stores (k, v), each (b, h, s, d_h).  MLA stores (c_kv (b, s, d_c), k_rope (b, 1, s, d_h^R)).
    """

    def __init__(self):
        self.first: Optional[torch.Tensor] = None
        self.second: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return 0 if self.first is None else self.first.shape[-2]

    def numel_per_token(self) -> int:
        """Elements held per cached token, per batch row."""
        if self.first is None:
            return 0
        n = self.length
        batch = self.first.shape[0]
        return (self.first.numel() + self.second.numel()) // (batch * n)

    def update(self, first: torch.Tensor, second: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.first = first if self.first is None else torch.cat([self.first, first], dim=-2)
        self.second = second if self.second is None else torch.cat([self.second, second], dim=-2)
        return self.first, self.second


def causal_mask(t: int, s: int, device) -> Optional[torch.Tensor]:
    """(t, s) boolean mask: with s keys of which the last t are the queries' own positions,
    query i sees key j if j <= s - t + i. A single new query sees everything."""
    if t == 1:
        return None
    return torch.arange(s, device=device)[None, :] <= torch.arange(s - t, s, device=device)[:, None]


class MultiHeadLatentAttention(nn.Module):
    """Paper eq. 37-47 (Appendix C), plus the weight-absorption trick of section C for decoding.

    Like the reference implementation, the two latents are RMS-normalised (section 3.1 of the
    paper mentions "additional RMS Norm layers after the compressed latent vectors").
    """

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        d, nh, dh, dr = cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.rope_head_dim
        self.nh, self.dh, self.dr, self.dc, self.absorb = nh, dh, dr, cfg.kv_lora_rank, cfg.absorb
        q_in = cfg.q_lora_rank
        if q_in is None:
            self.wdq = self.q_norm = None
            self.wuq = nn.Linear(d, nh * dh, bias=False)
            self.wqr = nn.Linear(d, nh * dr, bias=False)
        else:
            self.wdq = nn.Linear(d, q_in, bias=False)                  # W_DQ
            self.q_norm = RMSNorm(q_in, cfg.norm_eps)
            self.wuq = nn.Linear(q_in, nh * dh, bias=False)            # W_UQ
            self.wqr = nn.Linear(q_in, nh * dr, bias=False)            # W_QR
        self.wdkv = nn.Linear(d, self.dc, bias=False)                  # W_DKV
        self.kv_norm = RMSNorm(self.dc, cfg.norm_eps)
        self.wuk = nn.Linear(self.dc, nh * dh, bias=False)             # W_UK
        self.wuv = nn.Linear(self.dc, nh * dh, bias=False)             # W_UV
        self.wkr = nn.Linear(d, dr, bias=False)                        # W_KR (shared by all heads)
        self.wo = nn.Linear(nh * dh, d, bias=False)                    # W_O

    def forward(self, x, cos, sin, cache: Optional[Cache] = None):
        b, t, _ = x.shape
        nh, dh, dr = self.nh, self.dh, self.dr

        c_q = x if self.wdq is None else self.q_norm(self.wdq(x))      # eq. 37
        q_c = self.wuq(c_q).view(b, t, nh, dh).transpose(1, 2)         # eq. 38  (b, h, t, d_h)
        q_r = self.wqr(c_q).view(b, t, nh, dr).transpose(1, 2)         # eq. 39  (b, h, t, d_hR)
        c_kv = self.kv_norm(self.wdkv(x))                              # eq. 41  (b, t, d_c)
        k_r = self.wkr(x).view(b, t, 1, dr).transpose(1, 2)            # eq. 43  (b, 1, t, d_hR)

        start = cache.length if cache is not None else 0
        q_r, k_r = apply_rope(q_r, cos, sin, start), apply_rope(k_r, cos, sin, start)
        if cache is not None:
            c_kv, k_r = cache.update(c_kv, k_r)                        # only these two are cached
        s = c_kv.shape[1]
        mask = causal_mask(t, s, x.device)
        scale = (dh + dr) ** -0.5                                      # eq. 46: 1 / sqrt(d_h + d_hR)

        if not self.absorb:
            # Naive path: rebuild per-head keys and values from the latent.
            k_c = self.wuk(c_kv).view(b, s, nh, dh).transpose(1, 2)    # eq. 42
            v = self.wuv(c_kv).view(b, s, nh, dh).transpose(1, 2)      # eq. 45
            q = torch.cat([q_c, q_r], dim=-1)                          # eq. 40
            k = torch.cat([k_c, k_r.expand(-1, nh, -1, -1)], dim=-1)   # eq. 44
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        else:
            # Absorbed path: q_C . (W_UK c) = (W_UK^T q_C) . c, so scores are taken directly against the
            # latent, and W_UV is applied once after the weighted sum. Keys and values are never built.
            w_uk = self.wuk.weight.view(nh, dh, self.dc)
            w_uv = self.wuv.weight.view(nh, dh, self.dc)
            q_lat = torch.einsum("bhtd,hdk->bhtk", q_c, w_uk)          # (b, h, t, d_c)
            q = torch.cat([q_lat, q_r], dim=-1)                        # (b, h, t, d_c + d_hR)
            k = torch.cat([c_kv.unsqueeze(1), k_r], dim=-1)            # (b, 1, s, d_c + d_hR), shared by heads
            out_lat = F.scaled_dot_product_attention(
                q, k.expand(-1, nh, -1, -1), c_kv.unsqueeze(1).expand(-1, nh, -1, -1), attn_mask=mask, scale=scale
            )                                                          # (b, h, t, d_c)
            out = torch.einsum("bhtk,hdk->bhtd", out_lat, w_uv)
        return self.wo(out.transpose(1, 2).reshape(b, t, nh * dh))


class MultiHeadAttention(nn.Module):
    """Standard causal multi-head attention. Caches full keys and values: 2 * n_h * d_h per token."""

    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.nh, self.dh = cfg.n_heads, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, self.nh * self.dh, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.nh * self.dh, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.nh * self.dh, bias=False)
        self.wo = nn.Linear(self.nh * self.dh, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: Optional[Cache] = None):
        b, t, _ = x.shape
        q, k, v = (w(x).view(b, t, self.nh, self.dh).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        start = cache.length if cache is not None else 0
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if cache is not None:
            k, v = cache.update(k, v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask(t, k.shape[-2], x.device))
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))
