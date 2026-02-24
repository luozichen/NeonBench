import os
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from tokenizers import Tokenizer
import numpy as np

# Unified Trainer for Parity-Aware Models
def get_config(model_name):
    # Relative import from train.py to keep configs synced
    from train import get_config as base_get_config
    return base_get_config(model_name)

class TurboSampler:
    def __init__(self, data_path, seq_len, batch_size, device):
        self.data = np.fromfile(data_path, dtype=np.uint16)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

    def get_batch(self, split, parity_shift=False):
        # We always keep y = x.clone() shifted by 1 relative to raw file for parity logic
        # data[i:i+L] vs data[i+1:i+1+L]
        data = self.data # simplified for demo, in real we split train/val
        ix = torch.randint(len(data) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        
        if parity_shift:
            # Physical shift for models that need it (Neon231)
            # Targets are the original tokens
            y = x.clone()
        else:
            # Standard shifted targets for models that use internal masking (Neon232/233)
            y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
            
        return x.to(self.device), y.to(self.device)

def run_eval(model, sampler, model_name, steps=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            for strm in [False, True]:
                # Neon231 needs sampler shift. 232/233 use mask shift.
                s_shift = strm if model_name == "neon231" else False
                vx, vy = sampler.get_batch('val', parity_shift=s_shift)
                if model_name == "neon231":
                    _, loss = model(vx, vy, parity_shift=strm)
                else:
                    _, loss = model(vx, vy, is_odd_stream=strm)
                losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_interval", type=int, default=500)
    args = parser.parse_args()

    config = get_config(args.model)
    device = config['device']
    tokenizer = Tokenizer.from_file(args.tokenizer)
    config['vocab_size'] = tokenizer.get_vocab_size()

    # Import Model
    module = __import__(f"models.{args.model}", fromlist=[args.model.capitalize()])
    ModelClass = getattr(module, args.model.capitalize())
    model = ModelClass(config).to(device)
    
    print(f"Initializing {args.model} (Parity Training)...")
    print(f"Non-Embedding Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) - model.token_emb.weight.numel():,}")

    sampler = TurboSampler(args.data, config['block_size'], config['batch_size'], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    log_path = f"logs/{args.model}_parity_log.txt"
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f: f.write(f"Training {args.model} - Parity Sync\n")

    pbar = tqdm(range(args.steps), desc=args.model)
    for step in pbar:
        # Strict Alternation
        is_odd_stream = (step % 2 != 0)
        
        # Logging check (mirror train.py: log at 0, 500... 9500)
        if step % args.eval_interval == 0:
            val_loss = run_eval(model, sampler, args.model)
            # Forward pass just for the train loss log
            s_shift = is_odd_stream if args.model == "neon231" else False
            x, y = sampler.get_batch('train', parity_shift=s_shift)
            with torch.no_grad():
                if args.model == "neon231": _, train_loss = model(x, y, parity_shift=is_odd_stream)
                else: _, train_loss = model(x, y, is_odd_stream=is_odd_stream)
            
            log_msg = f"Step {step}: Train Loss {train_loss.item():.4f}, Val Loss {val_loss:.4f}"
            tqdm.write(log_msg)
            with open(log_path, "a") as f: f.write(log_msg + "\n")

        # Training Step
        sampler_shift = is_odd_stream if args.model == "neon231" else False
        x, y = sampler.get_batch('train', parity_shift=sampler_shift)
        
        if args.model == "neon231":
            logits, loss = model(x, y, parity_shift=is_odd_stream)
        else:
            logits, loss = model(x, y, is_odd_stream=is_odd_stream)
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", p=("Odd" if is_odd_stream else "Even"))

    print("Training Complete.")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/{args.model}_parity_final.pth")

if __name__ == "__main__":
    main()
