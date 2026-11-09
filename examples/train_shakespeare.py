"""Train a character-level LM on tiny Shakespeare, with latent (MLA) or standard multi-head (MHA) attention.

    python examples/train_shakespeare.py --attn mla
    python examples/train_shakespeare.py --attn mha

Same seed, data order, schedule and (almost) parameter count, so the two runs are comparable.
Reports validation loss in bits per character.
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import torch

from mla_pytorch import MLAConfig, MLALanguageModel

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_text(path: Path) -> str:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    return path.read_text()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["mla", "mha"], default="mla")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--kv-rank", type=int, default=64, help="MLA latent size d_c")
    ap.add_argument("--q-rank", type=int, default=96, help="MLA query compression d_c'")
    ap.add_argument("--rope-dim", type=int, default=16, help="MLA decoupled rotary size d_h^R")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    text = load_text(Path("data/shakespeare.txt"))
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    n = int(0.9 * len(data))
    train, val = data[:n], data[n:]

    cfg = MLAConfig(vocab_size=len(chars), d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                    head_dim=args.head_dim, attention=args.attn, kv_lora_rank=args.kv_rank, q_lora_rank=args.q_rank,
                    rope_head_dim=args.rope_dim, max_seq_len=args.seq)
    model = MLALanguageModel(cfg).to(device)
    print(f"{args.attn}: {model.num_params():,} params, cache {cfg.cache_per_token} numbers/token/layer, device {device}")

    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def batch(split, g):
        ix = torch.randint(len(split) - args.seq - 1, (args.batch,), generator=g)
        x = torch.stack([split[i : i + args.seq] for i in ix])
        y = torch.stack([split[i + 1 : i + args.seq + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def evaluate():
        model.eval()
        g = torch.Generator().manual_seed(1234)  # same validation windows every time and for both models
        loss = sum(model(*batch(val, g))[1].item() for _ in range(40)) / 40
        model.train()
        return loss

    g = torch.Generator().manual_seed(args.seed)
    warmup, t0 = 100, time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * step / warmup if step <= warmup else \
            0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
        for group in opt.param_groups:
            group["lr"] = lr
        _, loss = model(*batch(train, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate()
            print(f"step {step:5d}  train {loss.item():.4f}  val {v:.4f}  ({v / math.log(2):.4f} bits/char)  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
