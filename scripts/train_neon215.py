import sys
import os
# Fix import error if run from scripts/
sys.path.append(os.getcwd())

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from models.neon215 import Neon215

# --- Muon Optimizer Implementation ---
def muon_update(p, grad, lr, momentum, state):
    if momentum > 0:
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(grad)
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad)
        g = buf
    else:
        g = grad

    if g.ndim == 2: # Linear or Conv
        X = g.to(torch.float32)
        if X.shape[0] < X.shape[1]: X = X.T
        
        # Iterative Orthogonalization (Newton-Schulz)
        for _ in range(5):
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
            
        if g.shape[0] < g.shape[1]: X = X.T
        g = X.to(g.dtype)
        
    p.data.add_(g, alpha=-lr)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
        
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                muon_update(p, p.grad, lr, momentum, state)

# --- Data Sampler ---
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

# --- Training Logic ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_DIR = "checkpoints/neon215_balanced"
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_path = "logs/neon215_balanced_log.txt"

    # Base Configuration
    base_config = {
        'd_model': 256,
        'n_head': 8,
        'd_ff': 1024,
        'vocab_size': 16384,
        'block_size': 256,
        'conv_k': 21,
        'mlp_k': 21,
    }

    # 2-Stage Plan: 7L -> 14L
    STAGES = [
        {'n_layers': 7, 'steps': 10000, 'lr': 0.0002},
        {'n_layers': 14, 'steps': 20000, 'lr': 0.0001},
    ]

    print("Loading dataset into GPU memory...")
    sampler = TurboSampler(args.data, batch_size=32, seq_len=256, device=DEVICE)
    
    current_model = None
    global_step = 0

    for stage_idx, stage_info in enumerate(STAGES):
        n_layers = stage_info['n_layers']
        steps = stage_info['steps']
        lr = stage_info['lr']
        
        print(f"\nNEON 215 STAGE {stage_idx+1} | Layers={n_layers} | lr={lr}")
        
        config = base_config.copy()
        config['n_layers'] = n_layers
        
        new_model = Neon215(config).to(DEVICE)
        
        if current_model is not None:
            print(f"Growing model: {current_model.config['n_layers']}L -> {n_layers}L")
            new_sd = new_model.state_dict()
            old_sd = current_model.state_dict()
            # Copy all matching keys (bottom layers and head/embedding)
            for k, v in old_sd.items():
                if k in new_sd:
                    new_sd[k].copy_(v)
            new_model.load_state_dict(new_sd)
        
        current_model = new_model
        # Use Muon for all 2D parameters
        muon_params = [p for p in current_model.parameters() if p.ndim >= 2]
        adam_params = [p for p in current_model.parameters() if p.ndim < 2]
        
        optimizer = Muon(muon_params, lr=lr)
        adam = torch.optim.AdamW(adam_params, lr=lr/10) # Low LR for norms/etc
        
        scaler = GradScaler()
        current_model.train()
        
        pbar = tqdm(range(steps), desc=f"Stage {stage_idx+1} ({n_layers}L)")
        for _ in pbar:
            x, y = sampler.get_batch('train')
            
            with autocast():
                logits, loss = current_model(x, y)
                
            optimizer.zero_grad()
            adam.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(adam)
            scaler.update()
            
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if global_step % 250 == 0:
                current_model.eval()
                with torch.no_grad():
                    eval_iters = 50
                    v_losses = torch.zeros(eval_iters)
                    for i in range(eval_iters):
                        vX, vY = sampler.get_batch('val')
                        _, v_loss_batch = current_model(vX, vY)
                        v_losses[i] = v_loss_batch.item()
                    v_loss = v_losses.mean()
                    
                    msg = f"Neon215 S{stage_idx+1} | Step {global_step}: Train {loss.item():.4f}, Val {v_loss.item():.4f} ({n_layers}L)"
                    tqdm.write(msg)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
                current_model.train()

        # Save stage checkpoint
        ckpt_path = os.path.join(CKPT_DIR, f"stage{stage_idx+1}.pth")
        torch.save({
            'model': current_model.state_dict(),
            'config': current_model.config,
            'step': global_step
        }, ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")

if __name__ == "__main__":
    train()
