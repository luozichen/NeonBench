"""Progressive Trainer for Neon231 (The 8-Layer Gold Standard).
Strictly aligned with neon213_muon_long architecture and Muon V4 settings.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import math
import numpy as np
from tokenizers import Tokenizer

# FIX: Prevent torch.compile re-compilation loops on RTX 4090
torch._dynamo.config.cache_size_limit = 64

# Project Imports
sys.path.append(os.getcwd())
from models.neon231 import Neon231
from train import get_config

# ============================================================
# 1. Muon Optimizer Implementation (V4 - Reference Aligned)
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
    def __init__(self, params, lr=0.005, momentum=0.95, ns_steps=5):
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
# 2. Data Sampler (TurboSampler)
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

    def get_batch(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.seq_len, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

# ============================================================
# 3. Main Running Loop
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description="Neon231 Progressive Trainer (The Standard)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon231")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/neon231_training_log.txt"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    sampler = TurboSampler(args.data, batch_size=args.batch_size, seq_len=256, device=DEVICE)
    
    config = get_config("neon231")
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    print(f"Initializing Neon231 (8-Layer Gold Standard)...")
    model = Neon231(config).to(DEVICE)
    
    print("Compiling model (using 64-limit cache)...")
    model = torch.compile(model)
    
    # Optimizer Parameters
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "token_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    # STABILITY: Use 0.005 LR from neon213_muon_long for absolute safety
    optimizer_muon = Muon(muon_params, lr=0.005)
    optimizer_adam = torch.optim.AdamW(adam_params, lr=0.00015, weight_decay=0.1)
    scaler = GradScaler()

    # Growth Schedule
    growth_thresholds = {
        15000: 3, 20000: 5, 23000: 7, 25000: 9, 26000: 11,
        27000: 13, 28000: 15, 28500: 17, 29000: 19, 29500: 21
    }
    current_k = 1
    model.set_kernel_size(current_k)

    # 4. Training
    model.train()
    pbar = tqdm(range(args.steps), desc="Neon231")
    for step in pbar:
        if step in growth_thresholds:
            target_k = growth_thresholds[step]
            print(f"\n[GROWTH] Step {step}: k={current_k} -> k={target_k}")
            model.set_kernel_size(target_k)
            current_k = target_k
            torch.cuda.empty_cache() 
        
        # Cosine Decay (Standard)
        progress = step / args.steps
        lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        for g in optimizer_muon.param_groups: g['lr'] = 0.005 * lr_mult
        for g in optimizer_adam.param_groups: g['lr'] = 0.00015 * lr_mult
            
        x, y = sampler.get_batch('train')
        with autocast('cuda'):
            logits, loss = model(x, y)
        
        if torch.isnan(loss):
            print(f"\nCRITICAL: NaN Loss detected at step {step}!")
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
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "k": current_k})
        
        if (step+1) % 500 == 0:
            torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pth"))

    print("\nTRAINING COMPLETE.")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "neon231_final.pth"))

if __name__ == "__main__":
    main()
