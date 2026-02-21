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

# --- Muon Optimizer Implementation (V4 - Reference Aligned) ---

# Coeffs for Polar Express iterative orthogonalization
coeffs_list = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

def zeropower_polar_express(G: torch.Tensor, steps: int = 5):
    """Polar express as replacement for Newton-Schulz iteration"""
    X = G.to(torch.float32)
    transpose_needed = X.size(-2) > X.size(-1) 
    if transpose_needed: X = X.mT 
    
    # Safety normalization
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

                # Use lerp_ for stable momentum (Ref implementation)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) # Uses nesterov-like logic implicitly or just lerped buf
                
                # Use Polar Express for orthogonalization
                g = zeropower_polar_express(g, steps=group["ns_steps"])
                g = g.to(p.dtype)
                
                # Aspect-ratio based scaling (Key to avoiding NaNs)
                scale = max(1, p.size(-2) / p.size(-1))**0.5
                p.add_(g.view_as(p), alpha=-group["lr"] * scale)

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

    print(f"Initializing Neon213 (Muon V4 - Reference Aligned)...")
    model = Neon213(config).to(DEVICE)
    
    # Hybrid Optimizer setup
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "token_emb" not in name and "head" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    
    print(f"Muon params: {len(muon_params)} | AdamW params: {len(adam_params)}")
    
    optimizer = Muon(muon_params, lr=0.01) # Ref uses 0.02, we use 0.01 for stability on small BS
    adam = torch.optim.AdamW(adam_params, lr=0.0003)
    
    BATCH_SIZE = 64
    sampler = TurboSampler(args.data, batch_size=BATCH_SIZE, seq_len=256, device=DEVICE)
    
    def get_lr(it, max_it, base_lr):
        return base_lr * (1.0 - it / max_it)

    scaler = GradScaler()
    model.train()

    pbar = tqdm(range(args.steps), desc="Neon213 + Muon V4")
    for step in pbar:
        # Linear decay
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
                
                msg = f"Neon213 Muon V4 | Step {step+1}: Train {loss.item():.4f}, Val {v_loss.item():.4f}"
                tqdm.write(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            model.train()

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "neon213_muon_final.pth"))

if __name__ == "__main__":
    train()
