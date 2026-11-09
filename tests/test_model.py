import pytest
import torch

from mla_pytorch import MLAConfig, new_cache
from conftest import cfg, make


def test_cache_elements_per_token_match_paper_table_1():
    """MHA: 2 n_h d_h.  MLA: d_c + d_hR.  Measured on the real cache tensors."""
    idx = torch.randint(0, 50, (3, 12))
    for kw, want in [(dict(attention="mha"), 2 * 4 * 16), (dict(attention="mla"), 24 + 8)]:
        m = make(**kw)
        cache = new_cache(m.cfg)
        m(idx, cache=cache)
        assert cache[0].numel_per_token() == want == m.cfg.cache_per_token
    assert make(attention="mla").cfg.cache_per_token < make(attention="mha").cfg.cache_per_token / 3


def test_paper_deepseek_v2_numbers():
    """n_h = d_h = 128, d_c = 512, d_hR = 64: the cache holds 576 numbers per token per layer instead of 32768."""
    big = MLAConfig(n_heads=128, head_dim=128, kv_lora_rank=512, rope_head_dim=64)
    assert big.cache_per_token == 576
    assert MLAConfig(n_heads=128, head_dim=128, attention="mha").cache_per_token == 32768


def test_invalid_config():
    with pytest.raises(ValueError):
        MLAConfig(attention="nope")
    with pytest.raises(ValueError):
        MLAConfig(attention="mha", absorb=True)
    with pytest.raises(ValueError):
        MLAConfig(rope_head_dim=7)


def test_shapes_and_loss(model):
    idx = torch.randint(0, 50, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, 50) and loss.ndim == 0 and torch.isfinite(loss)


def test_no_future_leakage(model):
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 50, (4,))
    la, _ = model(a)
    lb, _ = model(b)
    torch.testing.assert_close(la[:, :8], lb[:, :8], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:])


def test_cache_matches_full_forward(model):
    idx = torch.randint(0, 50, (2, 20))
    full, _ = model(idx)
    cache = new_cache(model.cfg)
    steps = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(20)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-4, rtol=1e-4)


def test_chunk_after_cached_prefix(model):
    idx = torch.randint(0, 50, (2, 24))
    full, _ = model(idx)
    cache = new_cache(model.cfg)
    model(idx[:, :10], cache=cache)
    tail, _ = model(idx[:, 10:], cache=cache)
    torch.testing.assert_close(tail, full[:, 10:], atol=1e-4, rtol=1e-4)


def test_greedy_generate_matches_naive_loop(model):
    prompt = torch.randint(0, 50, (1, 5))
    got = model.generate(prompt, 12, temperature=0)
    idx = prompt
    for _ in range(12):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(got, idx)


def test_length_limit(model):
    with pytest.raises(ValueError):
        model(torch.randint(0, 50, (1, 65)))
    assert model.generate(torch.randint(0, 50, (1, 60)), 100, temperature=0).shape[1] == 64


@pytest.mark.parametrize("attention", ["mla", "mha"])
def test_overfits_one_sequence(attention):
    m = make(attention=attention).train()
    idx = torch.randint(0, 50, (1, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        _, loss = m(idx, idx.roll(-1, 1))
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first
