import math

import torch

from mla_pytorch import MultiHeadLatentAttention, new_cache
from mla_pytorch.attention import apply_rope, precompute_rope
from conftest import cfg, make


def reference(attn, x, cos, sin):
    """Appendix C eq. 37-47 written out literally, one head and one sequence at a time."""
    b, t, _ = x.shape
    nh, dh, dr = attn.nh, attn.dh, attn.dr
    mask = torch.arange(t)[None, :] <= torch.arange(t)[:, None]
    out = torch.zeros(b, t, nh * dh)
    for bi in range(b):
        c_q = attn.q_norm(attn.wdq(x[bi]))                       # (37) + latent RMSNorm
        q_c = attn.wuq(c_q)                                      # (38)  (t, nh*dh)
        q_r = attn.wqr(c_q)                                      # (39, before RoPE)
        c_kv = attn.kv_norm(attn.wdkv(x[bi]))                    # (41)
        k_c = attn.wuk(c_kv)                                     # (42)
        v_c = attn.wuv(c_kv)                                     # (45)
        k_r = apply_rope(attn.wkr(x[bi]), cos, sin)              # (43)  one key, shared
        for h in range(nh):
            qh = torch.cat([q_c[:, h * dh:(h + 1) * dh], apply_rope(q_r[:, h * dr:(h + 1) * dr], cos, sin)], -1)  # (40)
            kh = torch.cat([k_c[:, h * dh:(h + 1) * dh], k_r], -1)                                                # (44)
            s = (qh @ kh.T) / math.sqrt(dh + dr)                 # (46)
            p = s.masked_fill(~mask, -torch.inf).softmax(-1)
            out[bi, :, h * dh:(h + 1) * dh] = p @ v_c[:, h * dh:(h + 1) * dh]
    return attn.wo(out)                                          # (47)


def test_matches_paper_equations_written_naively():
    c = cfg()
    attn = MultiHeadLatentAttention(c)
    x = torch.randn(2, 7, c.d_model)
    cos, sin = precompute_rope(c.rope_head_dim, 16, 10000.0)
    torch.testing.assert_close(attn(x, cos, sin), reference(attn, x, cos, sin), atol=1e-5, rtol=1e-5)


def test_absorbed_equals_naive():
    c = cfg()
    a, b = MultiHeadLatentAttention(c), MultiHeadLatentAttention(cfg(absorb=True))
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 9, c.d_model)
    cos, sin = precompute_rope(c.rope_head_dim, 16, 10000.0)
    torch.testing.assert_close(a(x, cos, sin), b(x, cos, sin), atol=1e-5, rtol=1e-5)


def test_absorbed_equals_naive_through_the_cache():
    a, b = make(), make(absorb=True)
    b.load_state_dict(a.state_dict())
    idx = torch.randint(0, 50, (2, 14))
    ca, cb = new_cache(a.cfg), new_cache(b.cfg)
    for i in range(14):
        la, _ = a(idx[:, i : i + 1], cache=ca)
        lb, _ = b(idx[:, i : i + 1], cache=cb)
        torch.testing.assert_close(la, lb, atol=1e-4, rtol=1e-4)


def test_decoupled_key_is_shared_by_all_heads():
    attn = MultiHeadLatentAttention(cfg())
    assert attn.wkr.weight.shape[0] == attn.dr  # one key of size d_hR, not n_heads of them


def test_rope_part_is_relative():
    cos, sin = precompute_rope(8, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)
    def score(pq, pk):
        return (apply_rope(q, cos, sin, pq) * apply_rope(k, cos, sin, pk)).sum()
    torch.testing.assert_close(score(10, 4), score(40, 34), atol=1e-4, rtol=1e-4)
