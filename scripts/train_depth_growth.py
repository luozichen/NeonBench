"""
Neon 214: Progressive Depth Growth (8L -> 16L -> 24L -> 32L)
Optimizer: Muon (Orthogonalization) for Linear/Conv layers.
Includes: Mixed Precision (AMP) and TurboSampler.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import warnings
from transformers import logging
from tokenizers import Tokenizer
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.getcwd())
from models.neon214 import Neon214, Block

# ============================================================
# Muon Optimizer (Newton-Schulz Orthogonalization)
# ============================================================
class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum Update with Orthogonalization.
    Specifically designed for Transformer weight matrices.
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None: continue
                
                g = p.grad
                state = self.state[p]

                # 1. Momentum Accumulation
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # 2. Spectral Orthogonalization (Newton-Schulz)
                # Only apply to 2D matrices (Linear/Conv)
                if p.dim() >= 2:
                    X = buf.view(buf.size(0), -1)
                    # Normalize spectral norm
                    X /= (X.norm() + 1e-7)
                    
                    # Newton-Schulz iteration
                    for _ in range(ns_steps):
                        X = 1.5 * X - 0.5 * X @ X.t() @ X
                    
                    # Apply update
                    p.data.add_(X.view_as(p), alpha=-lr)
                else:
                    # Fallback to standard SGD for biases/1D
                    p.data.add_(buf, alpha=-lr)

# ============================================================
# Training Configuration
# ============================================================
STAGES = [
    {'n_layers': 8, 'steps': 5000, 'lr': 1e-4},
    {'n_layers': 16, 'steps': 5000, 'lr': 1e-4},
    {'n_layers': 24, 'steps': 5000, 'lr': 1e-4},
    {'n_layers': 32, 'steps': 10000, 'lr': 5e-5},
]

BASE_CONFIG = {
    'd_model': 192,
    'n_head': 6,
    'd_ff': 768,
    'block_size': 256,
    'batch_size': 64,
    'conv_k': 21,
    'mlp_k': 21,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ============================================================
# GPU Sampler (Copied for isolation)
# ============================================================
class TurboSampler:
    def __init__(self, data_path, block_size, batch_size, device, train_frac=0.9):
        print(f"Loading dataset into GPU memory...")
        data_np = np.fromfile(data_path, dtype=np.uint16)
        n = len(data_np)
        self.split = int(train_frac * n)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.data_gpu = torch.from_numpy(data_np.astype(np.int64)).to(device)

    def get_batch(self, split='train'):
        hi = self.split if split == 'train' else len(self.data_gpu)
        lo = 0 if split == 'train' else self.split
        max_idx = hi - self.block_size - 1
        idxs = torch.randint(lo, max_idx, (self.batch_size,), device=self.device)
        x = torch.stack([self.data_gpu[i : i + self.block_size] for i in idxs])
        y = torch.stack([self.data_gpu[i+1 : i+1 + self.block_size] for i in idxs])
        return x, y

def save_checkpoint(model, optimizer, scaler, global_step, stage_idx, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'model': base_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'global_step': global_step,
        'stage_idx': stage_idx,
        'config': config,
    }, path)
    print(f"  Saved checkpoint -> {path}")

def main():
    parser = argparse.ArgumentParser(description="Neon 214 Depth Growth Trainer")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon214_depth")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    device = BASE_CONFIG['device']
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    sampler = TurboSampler(args.data, BASE_CONFIG['block_size'], BASE_CONFIG['batch_size'], device)

    log_path = os.path.join(args.log_dir, "neon214_depth_log.txt")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Initialize Model at Stage 1 (8 Layers)
    current_config = {**BASE_CONFIG, 'vocab_size': vocab_size, 'n_layers': STAGES[0]['n_layers']}
    model = Neon214(current_config).to(device)
    
    global_step = 0
    scaler = GradScaler()
    model = torch.compile(model)

    for stage_idx, stage in enumerate(STAGES):
        print(f"\nNEON 214 STAGE {stage_idx+1} | Layers={stage['n_layers']} | lr={stage['lr']}")
        
        # Growth Logic: If we need more layers, add them and inherit weights
        if stage['n_layers'] > current_config['n_layers']:
            print(f"Growing model: {current_config['n_layers']}L -> {stage['n_layers']}L")
            base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            # Create a new config and model
            new_config = {**current_config, 'n_layers': stage['n_layers']}
            new_model = Neon214(new_config).to(device)
            
            # Transfer weights: First N layers are copied, new layers are initialized fresh or from neighbors
            # A common trick: initialize new layer as an identity or copy the previous layer's weights
            with torch.no_grad():
                # Copy Embeddings, Final Norm, Header
                new_model.token_emb.weight.copy_(base_model.token_emb.weight)
                new_model.ln_f.weight.copy_(base_model.ln_f.weight)
                new_model.head.weight.copy_(base_model.head.weight)
                
                # Copy existing blocks
                for idx in range(current_config['n_layers']):
                    new_model.blocks[idx].load_state_dict(base_model.blocks[idx].state_dict())
                
                # Initialize new blocks (Simple method: Copy previous block to give a 'warm' identity start)
                for idx in range(current_config['n_layers'], stage['n_layers']):
                    prev_idx = idx - 1 if idx > 0 else 0
                    new_model.blocks[idx].load_state_dict(new_model.blocks[prev_idx].state_dict())
            
            model = torch.compile(new_model)
            current_config = new_config

        # Setup Muon Optimizer for the current model
        # Research suggests Muon works best with a lower LR than AdamW
        optimizer = Muon(model.parameters(), lr=stage['lr'])

        model.train()
        pbar = tqdm(range(stage['steps']), desc=f"Stage {stage_idx+1} ({current_config['n_layers']}L)")
        
        for step in pbar:
            X, Y = sampler.get_batch('train')
            
            with autocast():
                _, loss = model(X, Y)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1

            if global_step % 250 == 0:
                model.eval()
                with torch.no_grad():
                    vX, vY = sampler.get_batch('val')
                    _, v_loss = model(vX, vY)
                    msg = f"Neon214 S{stage_idx+1} | Step {global_step}: Train {loss.item():.4f}, Val {v_loss.item():.4f} ({current_config['n_layers']}L)"
                    tqdm.write(msg)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
                model.train()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save stage checkpoint
        save_checkpoint(model, optimizer, scaler, global_step, stage_idx, 
                        current_config, os.path.join(args.out_dir, f"stage{stage_idx+1}.pth"))

    # Final Save
    final_path = os.path.join(args.out_dir, "neon214_final.pth")
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(base_model.state_dict(), final_path)
    print(f"\nTRAINING COMPLETE! Weights saved to: {final_path}")

if __name__ == "__main__":
    main()
