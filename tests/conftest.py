import pytest
import torch

from mla_pytorch import MLAConfig, MLALanguageModel


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def cfg(**kw):
    base = dict(vocab_size=50, d_model=64, n_layers=3, n_heads=4, head_dim=16, kv_lora_rank=24,
                q_lora_rank=32, rope_head_dim=8, max_seq_len=64)
    base.update(kw)
    return MLAConfig(**base)


def make(**kw):
    return MLALanguageModel(cfg(**kw)).eval()


VARIANTS = [
    dict(attention="mla"),
    dict(attention="mla", absorb=True),
    dict(attention="mla", q_lora_rank=None),
    dict(attention="mha"),
]


@pytest.fixture(params=VARIANTS, ids=["mla", "mla-absorbed", "mla-no-q-compression", "mha"])
def model(request):
    return make(**request.param)
