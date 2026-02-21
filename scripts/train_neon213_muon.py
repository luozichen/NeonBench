import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Fix import error if run from scripts/
sys.path.append(os.getcwd())
from models.neon213 import Neon213

# --- Muon Optimizer Implementation (V3 - Hardened) ---
def muon_update(p, grad, lr, momentum, state):
    if momentum > 0:
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(grad)
        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad)
        g = buf
    else:
        g = grad

    # Muon is only for 2D Dense weights.
    # Depthwise Convolutions are handled by AdamW.
    if g.ndim == 2:
        X = g.to(torch.float32)
        if X.shape[0] < X.shape[1]: X = X.T
        
        # Pre-normalize for Newton-Schulz convergence
        X /= (X.norm() + 1e-7)
        
        for _ in range(5):
            X = 1.5 * X - 0.5 * X @ (X.T @ X)
            
        if g.shape[0] < g.shape[1]: X = X.T
        
        # Matrix Scaling: matches AdamW update energy
        scale = 0.5 * max(g.shape[0], g.shape[1])**0.5
        g = (X * scale).to(g.dtype)
        
    p.data.add_(g, alpha=-lr)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.95):
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

# --- Main Run ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_DIR = "checkpoints/neon213_muon"
    os.makedirs(CKPT_DIR, exist_ok=True)
    log_path = "logs/neon213_muon_log.txt"

    config = {
        'd_model': 384, 'n_head': 6, 'n_layers': 8, 'd_ff': 1536, 
        'vocab_size': 16384, 'block_size': 256, 'conv_k': 21, 'mlp_k': 21
    }

    print(f"Initializing Neon213 (Muon Experiment)...")
    model = Neon213(config).to(DEVICE)
    
    # MUON RULES (Keller Jordan):
    # Apply only to 2D parameters.
    # Exclude Embeddings and Head.
    # Our Convolutions are Depthwise (Groups=D), so exclude them too.
    muon_params = []
    adam_params = []
    
    for name, p in model.named_parameters():
        if p.ndim == 2 and "token_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    
    print(f"Muon parameters: {len(muon_params)} | AdamW parameters: {len(adam_params)}")
    
    optimizer = Muon(muon_params, lr=0.01)
    adam = torch.optim.AdamW(adam_params, lr=0.0003)
    
    BATCH_SIZE = 64
    sampler = TurboSampler(args.data, batch_size=BATCH_SIZE, seq_len=256, device=DEVICE)
    
    def get_lr(it, max_it, base_lr):
        return base_lr * (1.0 - it / max_it)

    scaler = GradScaler()
    model.train()

    pbar = tqdm(range(args.steps), desc="Neon213 + Muon")
    for step in pbar:
        curr_muon_lr = get_lr(step, args.steps, 0.01)
        curr_adam_lr = get_lr(step, args.steps, 0.0003)
        for g in optimizer.param_groups: g['lr'] = curr_muon_lr
        for g in adam.param_groups: g['lr'] = curr_adam_lr

        x, y = sampler.get_batch('train')
        
        with autocast():
            logits, loss = model(x, y)
            
        optimizer.zero_grad()
        adam.zero_grad()
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        scaler.unscale_(adam)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.step(adam)
        scaler.update()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if (step+1) % 250 == 0:
            model.eval()
            with torch.no_grad():
                eval_iters = 50
                v_losses = torch.zeros(eval_iters)
                for i in range(eval_iters):
                    vX, vY = sampler.get_batch('val')
                    _, v_loss_batch = model(vX, vY)
                    v_losses[i] = v_loss_batch.item()
                v_loss = v_losses.mean()
                
                msg = f"Neon213 Muon | Step {step+1}: Train {loss.item():.4f}, Val {v_loss.item():.4f}"
                tqdm.write(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            model.train()

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "neon213_muon_final.pth"))

if __name__ == "__main__":
    train()
