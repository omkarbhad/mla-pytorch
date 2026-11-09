"""Model configuration."""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MLAConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4              # n_h
    head_dim: int = 64            # d_h
    attention: str = "mla"        # "mla" or "mha" (the baseline that caches full keys and values)
    kv_lora_rank: int = 128       # d_c: size of the cached latent
    q_lora_rank: Optional[int] = 192  # d_c': query compression (saves activations only); None = off
    rope_head_dim: int = 32       # d_h^R: size of the decoupled rotary query/key part
    absorb: bool = False          # inference trick: fold W_UK, W_UV into the query / output side
    ffn_dim: Optional[int] = None  # default: 8/3 * d_model, rounded up to a multiple of 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.attention not in ("mla", "mha"):
            raise ValueError("attention must be 'mla' or 'mha'")
        if self.ffn_dim is None:
            self.ffn_dim = (int(8 / 3 * self.d_model) + 63) // 64 * 64
        if self.rope_head_dim % 2 or self.head_dim % 2:
            raise ValueError("head_dim and rope_head_dim must be even for rotary embeddings")
        if self.absorb and self.attention != "mla":
            raise ValueError("absorb only applies to attention='mla'")

    @property
    def cache_per_token(self) -> int:
        """Cached elements per token per layer (paper, Table 1)."""
        if self.attention == "mha":
            return 2 * self.n_heads * self.head_dim
        return self.kv_lora_rank + self.rope_head_dim

    def to_dict(self) -> dict:
        return asdict(self)
