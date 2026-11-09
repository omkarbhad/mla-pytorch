"""Decoder-only language model: pre-RMSNorm, (latent) attention, SwiGLU, tied embeddings."""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Cache, MultiHeadAttention, MultiHeadLatentAttention, RMSNorm, precompute_rope
from .config import MLAConfig


def new_cache(cfg: MLAConfig) -> List[Cache]:
    return [Cache() for _ in range(cfg.n_layers)]


class SwiGLU(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w_down = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = (MultiHeadLatentAttention if cfg.attention == "mla" else MultiHeadAttention)(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class MLALanguageModel(nn.Module):
    def __init__(self, cfg: MLAConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # The rotary table is sized for the part that carries RoPE: d_h^R for MLA, d_h for MHA.
        rope_dim = cfg.rope_head_dim if cfg.attention == "mla" else cfg.head_dim
        cos, sin = precompute_rope(rope_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "mlp.w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, cache: Optional[List[Cache]] = None):
        """Returns (logits, loss). targets are the next tokens, same shape as idx (no shifting here)."""
        start = cache[0].length if cache is not None else 0
        if start + idx.shape[1] > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {start + idx.shape[1]} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, self.rope_cos, self.rope_sin, cache[i] if cache is not None else None)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sampling with a KV cache. temperature=0 means greedy."""
        was_training = self.training
        self.eval()
        max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - idx.shape[1])
        cache = new_cache(self.cfg)
        logits, _ = self(idx, cache=cache)
        for _ in range(max_new_tokens):
            last = logits[:, -1, :]
            if temperature == 0:
                nxt = last.argmax(dim=-1, keepdim=True)
            else:
                last = last / temperature
                if top_k:
                    kth = torch.topk(last, min(top_k, last.size(-1))).values[:, -1:]
                    last = last.masked_fill(last < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(last, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            logits, _ = self(nxt, cache=cache)
        self.train(was_training)
        return idx
