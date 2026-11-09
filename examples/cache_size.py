"""Print KV-cache size per token for MHA, GQA, MQA and MLA (paper, Table 1), for the paper's sizes."""

n_h, d_h, d_c, d_hr, layers = 128, 128, 512, 64, 60   # DeepSeek-V2 sizes (60 layers)
rows = {
    "MHA": 2 * n_h * d_h,
    "GQA (8 groups)": 2 * 8 * d_h,
    "MQA": 2 * d_h,
    "MLA": d_c + d_hr,
}
base = rows["MHA"]
print(f"{'':16}{'numbers / token / layer':>26}{'vs MHA':>10}{'bf16 KB / token (all layers)':>32}")
for name, n in rows.items():
    print(f"{name:16}{n:>26,}{n / base:>10.1%}{n * layers * 2 / 1024:>32,.1f}")
