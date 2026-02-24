"""Training Script for Neon231 (Fusion Model) with Strided Loss.
Alternates between Parity 0 and Parity 1 (Z-token shift) to ensure full coverage.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math
import numpy as np
from tokenizers import Tokenizer

# Project Imports
sys.path.append(os.getcwd())
from models.neon231 import Neon231
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
            # The model internally handles the Z-shift on x and cropping.
            # So targets provided here should be exactly the x tokens.
            y = x.clone()
        else:
            # Standard autoregressive targets: offset by 1
            y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
            
        return x.to(self.device), y.to(self.device)

# --- Training Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    config = get_config("neon231")
    config['vocab_size'] = vocab_size
    config['batch_size'] = args.batch_size
    
    print(f"Initializing Neon231 Fusion Model (5M scale)...")
    model = Neon231(config).to(DEVICE)
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sampler = TurboSampler(args.data, batch_size=args.batch_size, seq_len=256, device=DEVICE)
    
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/neon231_fusion_log.txt"

    pbar = tqdm(range(args.steps), desc="Neon231 Fusion")
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
            # Eval
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(10):
                    # Eval on both parities
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
    torch.save(model.state_dict(), "checkpoints/neon231_fusion_final.pth")

if __name__ == "__main__":
    main()
