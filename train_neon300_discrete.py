"""Training script for Neon300 with Discrete Halving LR Schedule.
Identical to train_neon300.py except the last 3k steps use discrete halvings
instead of linear cooldown. This preserves the exact same training for steps
0-27000 to allow direct comparison with the baseline neon300 run.
"""
import argparse
import os
import sys
import math
import time
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from tokenizers import Tokenizer

torch._dynamo.config.cache_size_limit = 64

sys.path.append(os.getcwd())
from models.neon300 import Neon300

# ============================================================
# 1. Muon Optimizer (identical to train_neon300.py)
# ============================================================
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.no_grad()
def zeropower_polar_express(G: torch.Tensor, steps: int = 5):
    X = G.to(torch.float32)
    transpose_needed = X.size(-2) > X.size(-1)
    if transpose_needed: X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for a, b, c in coeffs_list[:steps]:
        A = X @ X.mT
        A2 = A @ A
        B = b * A + c * A2
        X = a * X + B @ X
    if transpose_needed: X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"])
                g = zeropower_polar_express(g, steps=group["ns_steps"])
                g = g.to(p.dtype)
                scale = max(1, p.size(-2) / p.size(-1))**0.5
                p.add_(g.view_as(p), alpha=-group["lr"] * scale)

# ============================================================
# 2. Discrete Halving LR Schedule
# ============================================================
def get_lr_multiplier(step: int, max_steps: int):
    """Trapezoid + Discrete Halving.
    - 10% warmup (steps 0-3000)
    - Constant plateau (steps 3000-27000)
    - Last 3k steps: discrete halving every 750 steps
      27000: 0.5, 27750: 0.25, 28500: 0.125, 29250: 0.0625
    """
    warmup_steps = int(0.10 * max_steps)
    decay_start = max_steps - 3000  # Hard-coded: last 3k steps

    if step < warmup_steps:
        return step / warmup_steps
    elif step < decay_start:
        return 1.0
    else:
        # Discrete halvings every 750 steps
        steps_into_decay = step - decay_start
        n_halvings = steps_into_decay // 750
        return 0.5 ** (n_halvings + 1)

# ============================================================
# 3. TurboSampler
# ============================================================
class TurboSampler:
    def __init__(self, data_path, batch_size, seq_len, device):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.n_total = len(self.data)
        self.train_data = self.data[:int(self.n_total * 0.99)]
        self.val_data = self.data[int(self.n_total * 0.99):]
        print(f"Loaded {self.n_total:,} tokens ({len(self.train_data):,} train, {len(self.val_data):,} val)")

    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.seq_len, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

# ============================================================
# 4. Evaluation
# ============================================================
@torch.no_grad()
def estimate_loss(model, sampler, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = sampler.get_batch('val')
        with autocast('cuda'):
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

# ============================================================
# 5. Main
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description="Train Neon300 (Discrete Halving)")
    parser.add_argument("--data", type=str, default="data/fineweb/fineweb_tok6.bin")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/fineweb_tok6.json")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon300_discrete")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/neon300_discrete_training_log.txt"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size}")

    sampler = TurboSampler(args.data, batch_size=args.batch_size, seq_len=args.seq_len, device=DEVICE)

    config = {
        'vocab_size': vocab_size,
        'd_model': 512,
        'n_layers': 8,
        'n_head': 8,
        'd_ff': 2048,
        'block_size': args.seq_len,
    }

    print(f"Initializing Neon300 (Discrete Halving Experiment)...")
    print(f"Config: {config}")
    model = Neon300(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print("Compiling model with torch.compile...")
    model = torch.compile(model)

    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "token_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    print(f"Muon params: {sum(p.numel() for p in muon_params):,}")
    print(f"AdamW params: {sum(p.numel() for p in adam_params):,}")

    optimizer_muon = Muon(muon_params, lr=args.muon_lr)
    optimizer_adam = torch.optim.AdamW(adam_params, lr=args.adam_lr, weight_decay=0.1)
    scaler = GradScaler()

    with open(log_path, "w") as f:
        f.write(f"Neon300 Discrete Halving Training Log\n")
        f.write(f"Config: {config}\n")
        f.write(f"Parameters: {n_params:,}\n")
        f.write(f"Muon LR: {args.muon_lr}, AdamW LR: {args.adam_lr}\n")
        f.write(f"Schedule: Plateau + Discrete Halving (last 3k steps)\n")
        f.write(f"Steps: {args.steps}, Batch: {args.batch_size}, Seq: {args.seq_len}\n\n")

    best_val_loss = float('inf')
    model.train()
    pbar = tqdm(range(args.steps), desc="Neon300-Discrete")

    for step in pbar:
        lr_mult = get_lr_multiplier(step, args.steps)
        for g in optimizer_muon.param_groups: g['lr'] = args.muon_lr * lr_mult
        for g in optimizer_adam.param_groups: g['lr'] = args.adam_lr * lr_mult

        x, y = sampler.get_batch('train')
        with autocast('cuda'):
            logits, loss = model(x, y)

        if torch.isnan(loss):
            print(f"\nCRITICAL: NaN loss at step {step}!")
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"nan_dump_step_{step}.pth"))
            break

        optimizer_muon.zero_grad(set_to_none=True)
        optimizer_adam.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_muon)
        scaler.unscale_(optimizer_adam)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer_muon)
        scaler.step(optimizer_adam)
        scaler.update()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr_mult:.4f}"})

        if (step + 1) % args.eval_interval == 0:
            val_loss = estimate_loss(model, sampler)
            log_msg = f"Step {step+1}: Train {loss.item():.4f}, Val {val_loss:.4f}, LR_mult {lr_mult:.4f}"
            tqdm.write(log_msg)

            with open(log_path, "a") as f:
                f.write(log_msg + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pth"))
                tqdm.write(f"--> New best val loss: {val_loss:.4f}")

            torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pth"))

    print("\nTRAINING DONE.")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "neon300_discrete_final.pth"))
    print(f"Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
