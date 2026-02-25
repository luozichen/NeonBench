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
        
        # Simple split for eval consistency
        n = len(self.data)
        self.train_data = self.data[:int(n*0.9)]
        self.val_data = self.data[int(n*0.9):]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        # Standard shifted targets for all models
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)

def run_eval(model, sampler, model_name, steps=25):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            for strm in [False, True]:
                vx, vy = sampler.get_batch('val')
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
        
        # Step label E/O (Fixed length)
        p_label = "O" if is_odd_stream else "E"
        
        # Logging (Match train.py exactly: Step X: Train Loss Y, Val Loss Z)
        if step % args.eval_interval == 0:
            val_loss = run_eval(model, sampler, args.model)
            # Capture train loss for this specific step
            tx, ty = sampler.get_batch('train')
            with torch.no_grad():
                _, train_loss = model(tx, ty, is_odd_stream=is_odd_stream)
            
            log_msg = f"Step {step}: Train Loss {train_loss.item():.4f}, Val Loss {val_loss:.4f}"
            tqdm.write(log_msg)
            with open(log_path, "a") as f: f.write(log_msg + "\n")

        # Training
        x, y = sampler.get_batch('train')
        
        logits, loss = model(x, y, is_odd_stream=is_odd_stream)
            
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", p=p_label)

    print("Training Complete.")
    os.makedirs("checkpoints", exist_ok=True)
    final_ckpt_path = f"checkpoints/{args.model}_parity_final.pth"
    torch.save(model.state_dict(), final_ckpt_path)
    
    print("\n--- Running Final 500-Batch Massive Eval ---")
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(250):
            for strm in [False, True]:
                vx, vy = sampler.get_batch('val')
                _, loss = model(vx, vy, is_odd_stream=strm)
                losses.append(loss.item())
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/250] Running Avg: {sum(losses)/len(losses):.5f}")
                
    final_loss = sum(losses) / len(losses)
    final_msg = f"==> FINAL MASSIVE EVAL LOSS: {final_loss:.5f} (over {len(losses)} batches)"
    print(final_msg)
    with open(log_path, "a") as f:
        f.write("\n" + final_msg + "\n")

if __name__ == "__main__":
    main()
