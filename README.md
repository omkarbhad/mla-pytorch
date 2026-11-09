# mla-pytorch

A from-scratch PyTorch implementation of **Multi-head Latent Attention (MLA)** from
[DeepSeek-V2](https://arxiv.org/abs/2405.04434), with a standard multi-head attention baseline next to it.

This is an unofficial implementation written from the paper (section 2.1 and Appendix C).

## The idea

During generation, a Transformer keeps the keys and values of every past token in a cache. For
multi-head attention that is `2 · n_heads · head_dim` numbers per token per layer, and the cache
limits batch size and context length.

MLA compresses each token into one small latent vector and caches only that:

```
c_kv = W_DKV h          (size d_c, this is what is cached)
k_C  = W_UK c_kv        v_C = W_UV c_kv      (rebuilt from the latent when needed)
```

Rotary position embeddings cannot sit inside this low-rank path: the rotation would end up between
`W_Q` and `W_UK` and block the compression. So the paper adds a small **decoupled** part that
carries position: per-head rotary queries `q_R` and **one** rotary key `k_R` shared by all heads.
Each head attends with `q = [q_C ; q_R]` and `k = [k_C ; k_R]`, and the softmax scale is
`1 / sqrt(d_h + d_hR)`. The cache holds `c_kv` and `k_R`: **`d_c + d_hR` numbers per token per layer**.

| Attention | Cache per token per layer | At DeepSeek-V2 sizes |
| --- | --- | --- |
| MHA | `2 n_h d_h` | 32,768 |
| GQA (8 groups) | `2 n_g d_h` | 2,048 |
| MQA | `2 d_h` | 256 |
| MLA | `d_c + d_hR` | **576** (1.8% of MHA) |

(`n_h = d_h = 128`, `d_c = 512`, `d_hR = 64`, the sizes the paper reports. Run
`python examples/cache_size.py` for the table.)

## Install

```bash
git clone https://github.com/omkarbhad/mla-pytorch
cd mla-pytorch
pip install -e ".[dev]"
```

## Usage

```python
import torch
from mla_pytorch import MLAConfig, MLALanguageModel

cfg = MLAConfig(vocab_size=256, d_model=256, n_layers=4, n_heads=4, head_dim=64,
                kv_lora_rank=128, q_lora_rank=192, rope_head_dim=32)
model = MLALanguageModel(cfg)

idx = torch.randint(0, 256, (2, 64))
logits, loss = model(idx, targets=idx.roll(-1, 1))
out = model.generate(idx[:, :8], max_new_tokens=32, temperature=0.8)   # decodes from the latent cache

print(cfg.cache_per_token)        # 160 numbers per token per layer
baseline = MLALanguageModel(MLAConfig(vocab_size=256, d_model=256, n_layers=4, n_heads=4,
                                      head_dim=64, attention="mha"))    # cache: 512
```

Set `absorb=True` for the paper's decoding trick (see below).

## What is implemented

| Paper | Here |
| --- | --- |
| Eq. 9-11 and 41-42, 45: latent compression of keys and values | `MultiHeadLatentAttention` |
| Eq. 12-13 and 37-38: query compression | `q_lora_rank` (set `None` to turn it off) |
| Eq. 14-17 and 39-40, 43-44: decoupled RoPE, one shared key | `wqr`, `wkr` |
| Eq. 18 and 46: scale `1/sqrt(d_h + d_hR)` | `scale` in `forward` |
| Section C: absorb `W_UK` into the query and `W_UV` into the output | `absorb=True` |

As in the paper's practice (section 3.1) the two latent vectors get an RMSNorm. Rotary embeddings,
tied embeddings, SwiGLU and pre-RMSNorm follow LLaMA.

**Weight absorption.** Because `q_C · (W_UK c) = (W_UKᵀ q_C) · c`, attention scores can be taken directly
against the cached latent, and `W_UV` can be applied once after the weighted sum. Keys and values are then
never built per head. The result is identical to the plain path (a test checks this, also step by step
through the cache). This repo implements it with PyTorch's attention call and treats the latent as a single
shared key and value; it does not include a custom kernel, so it shows the algebra, not the speedup.

## Tests

```bash
pytest -q
```

34 tests. The main ones:

- MLA equals Appendix C eq. 37-47 written out literally, one head and one sequence at a time.
- The absorbed path equals the plain path.
- The cache really holds `d_c + d_hR` numbers per token (measured on the cache tensors), and `2 n_h d_h` for MHA.
- The paper's DeepSeek-V2 sizes give 576 versus 32,768.
- No token sees the future, and decoding with the cache matches a full forward pass, including a chunk after a cached prefix.
- Greedy generation matches a naive loop, and the model can overfit one sequence.

I checked that the tests fail when I break the code on purpose: use `1/sqrt(d_h)` as the scale, ignore the
rotary position offset, skip rotating the shared key, drop the rotary scores from the absorbed path, or
remove the causal mask.

## A small experiment

`examples/train_shakespeare.py` trains a character-level model on tiny Shakespeare. Same seed, data order and schedule.

```bash
python examples/train_shakespeare.py --attn mla
python examples/train_shakespeare.py --attn mha
```

Settings: 4 layers, `d_model` 128, 4 heads of size 32, context 128, 2,000 steps. MLA uses `d_c = 64`,
`d_c' = 96`, `d_hR = 16`. Validation loss at the end, 3 seeds:

| Attention | Seed 0 | Seed 1 | Seed 2 | Mean (nats) | Cache / token / layer | Params |
| --- | --- | --- | --- | --- | --- | --- |
| MHA | 1.5078 | 1.5045 | 1.5055 | **1.5059** | 256 | 861,440 |
| MLA | 1.5159 | 1.5076 | 1.5062 | 1.5099 | **80** (31%) | 894,848 |

MLA reaches nearly the same loss with 69% less cache. It is slightly worse in all three seeds, by 0.004 nats
on average, and it has 3.9% more parameters here. The paper reports MLA matching or beating MHA at much larger
scale (MoE models of 16B and 250B parameters). This run is small and short, so it shows that the latent
bottleneck costs little, not that MLA wins.

## Layout

```
mla_pytorch/
  config.py       MLAConfig, cache_per_token
  attention.py    MultiHeadLatentAttention, MultiHeadAttention, RoPE, Cache
  model.py        MLALanguageModel, generate
tests/            34 tests
examples/         train_shakespeare.py, cache_size.py
```

## Citation

```bibtex
@article{deepseekv2,
    title   = {DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model},
    author  = {DeepSeek-AI},
    journal = {arXiv preprint arXiv:2405.04434},
    year    = {2024}
}
```

## License

MIT
