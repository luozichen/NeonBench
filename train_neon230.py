"""Progressive Trainer for Neon230 (29M Momentum Model).
Implements the specific non-linear growth schedule and CUDA memory optimizations.
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

# Project Imports
sys.path.append(os.getcwd())
from models.neon230 import Neon230
from train import get_config, TextDataset

# Optimizers from project
from optimizer.muon import Muon

# ============================================================
# Optimized Configuration
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    parser = argparse.ArgumentParser(description="Neon230 Progressive Trainer")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32) # Standard for 20M
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon230")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Tokenizer & Data
    tokenizer = Tokenizer.from_file(args.tokenizer)
    dataset = TextDataset(args.data, tokenizer, block_size=256)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize Model
    config = get_config("neon230")
    config['vocab_size'] = tokenizer.get_vocab_size()
    config['batch_size'] = args.batch_size
    model = Neon230(config).to(DEVICE)
    
    # Compile
    print("Compiling neon230...")
    model = torch.compile(model)
    
    # 3. Setup Optimizers
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "token_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    optimizer_muon = Muon(muon_params, lr=0.02) # Higher Scale LR
    optimizer_adam = torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)
    scaler = GradScaler()

    # 4. Growth Schedule (User Requested)
    # k=3 at 15000, 5 at 20000, 7 at 23000, 9 at 25000, 11 at 26000, 13 at 27000, 
    # 15 at 28000, 17 at 28500, 19 at 29000, 21 at 29500
    growth_thresholds = {
        15000: 3, 20000: 5, 23000: 7, 25000: 9, 26000: 11,
        27000: 13, 28000: 15, 28500: 17, 29000: 19, 29500: 21
    }
    current_k = 1
    model.set_kernel_size(current_k)

    # 5. Training Loop
    model.train()
    step = 0
    pbar = tqdm(total=args.steps, desc="Training Neon230")
    
    while step < args.steps:
        for x, y in dataloader:
            if step >= args.steps: break
            
            # Growth Check
            if step in growth_thresholds:
                target_k = growth_thresholds[step]
                print(f"\n[GROWTH] Step {step}: k={current_k} -> k={target_k}")
                model.set_kernel_size(target_k)
                current_k = target_k
                torch.cuda.empty_cache() # Prevent OOM from re-compilation
            
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            with autocast('cuda'):
                logits, loss = model(x, y)
            
            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adam.zero_grad(set_to_none=True)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer_muon)
            scaler.step(optimizer_adam)
            scaler.update()
            
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "k": current_k})
            
            if step % 500 == 0:
                torch.save(model.state_dict(), os.path.join(args.out_dir, "latest.pth"))

    print("\nTRAINING COMPLETE!")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "neon230_final.pth"))

if __name__ == "__main__":
    main()
