"""Unified Trainer for Phase 5 Innovation Models (Neon231, Neon232).
Implements Dual-Stream Parity training with Strided Loss Masking.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
from tokenizers import Tokenizer

# Project Imports
sys.path.append(os.getcwd())
from train import get_config

# --- Simple Data Sampler ---
class TurboSampler:
    def __init__(self, data_path, batch_size, seq_len, device):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.n_total = len(self.data)
        self.train_data = self.data[:int(self.n_total * 0.99)]
        self.val_data = self.data[int(self.n_total * 0.99):]

    def get_batch(self, split='train', parity_shift=False):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        
        if parity_shift:
            # Shifted targets: input [Z, x0, x1...] means target is [x0, x1, x2...]
            # The model handles the Z-shift. Targets should be the original tokens.
            y = x.clone()
        else:
            # Standard: input [x0, x1...] means target is [x1, x2...]
            y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
            
        return x.to(self.device), y.to(self.device)

# --- Training Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["neon231", "neon232", "neon233"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    if args.model == "neon231":
        from models.neon231 import Neon231 as ModelClass
    elif args.model == "neon232":
        from models.neon232 import Neon232 as ModelClass
    else:
        from models.neon233 import Neon233 as ModelClass

    config = get_config(args.model)
    config['vocab_size'] = tokenizer.get_vocab_size()
    config['batch_size'] = args.batch_size
    
    print(f"Initializing {args.model} Fusion/Staircase (5M scale)...")
    model = ModelClass(config).to(DEVICE)
    model = torch.compile(model)
    
    # Simple AdamW for parity innovations (Muon can be used too, but AdamW is safer for weird archs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sampler = TurboSampler(args.data, batch_size=args.batch_size, seq_len=256, device=DEVICE)
    
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{args.model}_training_log.txt"

    pbar = tqdm(range(args.steps), desc=args.model)
    for step in pbar:
        # Alternate Parity
        parity_shift = (step % 2 == 1)
        
        # Linear Decay
        lr = args.lr * (1.0 - step / args.steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        x, y = sampler.get_batch('train', parity_shift=parity_shift)
        
        logits, loss = model(x, y, parity_shift=parity_shift)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", p=("Even" if not parity_shift else "Odd"))
        
        if (step + 1) % 500 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    for p_s in [False, True]:
                        vx, vy = sampler.get_batch('val', parity_shift=p_s)
                        _, vl = model(vx, vy, parity_shift=p_s)
                        val_losses.append(vl.item())
            val_loss = sum(val_losses) / len(val_losses)
            msg = f"Step {step+1}: Val Loss {val_loss:.4f}"
            tqdm.write(msg)
            with open(log_path, "a") as f: f.write(msg + "\n")
            model.train()

    print("TRAINING DONE.")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{args.model}_final.pth")

if __name__ == "__main__":
    main()
