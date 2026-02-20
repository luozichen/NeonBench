"""
Neon Turbo Trainer: Optimized Growth (k=9 to k=21)
Includes: Mixed Precision (AMP), torch.compile, and GPU-resident data.
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
from models.neon213 import Neon213, Block

# ============================================================
# Extended Stages
# ============================================================
STAGES = [
    {'n_layers': 8, 'conv_k': 11, 'mlp_k': 11, 'steps': 3000, 'lr': 1e-4},
    {'n_layers': 8, 'conv_k': 13, 'mlp_k': 13, 'steps': 3000, 'lr': 1e-4},
    {'n_layers': 8, 'conv_k': 15, 'mlp_k': 15, 'steps': 3000, 'lr': 1e-4},
    {'n_layers': 8, 'conv_k': 17, 'mlp_k': 17, 'steps': 3000, 'lr': 1e-4},
    {'n_layers': 8, 'conv_k': 19, 'mlp_k': 19, 'steps': 3000, 'lr': 1e-4},
    {'n_layers': 8, 'conv_k': 21, 'mlp_k': 21, 'steps': 5000, 'lr': 1e-4},
]

BASE_CONFIG = {
    'd_model': 384,
    'n_head': 6,
    'd_ff': 1536,
    'block_size': 256,
    'batch_size': 64,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Suppress annoying warning messages
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
torch._dynamo.config.suppress_errors = True
# Increase recompile limit for growth stages
torch._dynamo.config.recompile_limit = 32

# ============================================================
# GPU-Accelerated Sampler
# ============================================================
class TurboSampler:
    """Loads the entire dataset to GPU for near-instant sampling."""
    def __init__(self, data_path, block_size, batch_size, device, train_frac=0.9):
        print(f"Loading dataset into GPU memory...")
        # Load tokens from disk
        data_np = np.fromfile(data_path, dtype=np.uint16)
        n = len(data_np)
        self.split = int(train_frac * n)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # MUST use int64 (Long) for labels in CUDA cross_entropy
        self.data_gpu = torch.from_numpy(data_np.astype(np.int64)).to(device)
        print(f"Dataset Ready on {device} ({n:,} tokens).")

    def get_batch(self, split='train'):
        hi = self.split if split == 'train' else len(self.data_gpu)
        lo = 0 if split == 'train' else self.split
        max_idx = hi - self.block_size - 1
        
        # Fast GPU sampling
        idxs = torch.randint(lo, max_idx, (self.batch_size,), device=self.device)
        x = torch.stack([self.data_gpu[i : i + self.block_size] for i in idxs])
        y = torch.stack([self.data_gpu[i+1 : i+1 + self.block_size] for i in idxs])
        return x, y

# ============================================================
# Growth Functions
# ============================================================
def expand_kernels(model, old_k, new_k, config, device):
    d_model = config['d_model']
    pad_size = new_k - old_k
    # Note: We access the base model if compiled
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    for block in base_model.blocks:
        for conv_name in ['conv_q', 'conv_k', 'conv_v', 'conv_i']:
            old_conv = getattr(block.attn, conv_name)
            new_conv = nn.Conv1d(d_model, d_model, kernel_size=new_k,
                                 groups=d_model, bias=False).to(device)
            with torch.no_grad():
                new_conv.weight.zero_()
                new_conv.weight[:, :, pad_size:] = old_conv.weight
            setattr(block.attn, conv_name, new_conv)
            block.attn.k = new_k
        
        old_mlp_conv = block.mlp.conv_gate
        new_mlp_conv = nn.Conv1d(d_model, d_model, kernel_size=new_k,
                                  groups=d_model, bias=False).to(device)
        with torch.no_grad():
            new_mlp_conv.weight.zero_()
            new_mlp_conv.weight[:, :, pad_size:] = old_mlp_conv.weight
        block.mlp.conv_gate = new_mlp_conv
        block.mlp.k = new_k
    return model

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
    parser = argparse.ArgumentParser(description="Neon Turbo Trainer")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon213_turbo")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    device = BASE_CONFIG['device']
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    sampler = TurboSampler(args.data, BASE_CONFIG['block_size'], BASE_CONFIG['batch_size'], device)

    log_path = os.path.join(args.log_dir, "neon213_turbo_log.txt")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load Model (Detect size from checkpoint)
    print(f"Loading checkpoint from {args.resume}...")
    state = torch.load(args.resume, map_location=device)
    
    # Extract config from checkpoint or use default base (k=9)
    ckpt_config = state.get('config', {**BASE_CONFIG, 'vocab_size': vocab_size, 'n_layers': 8, 'conv_k': 9, 'mlp_k': 9})
    model = Neon213(ckpt_config).to(device)
    model.load_state_dict(state['model'] if 'model' in state else state)
    
    global_step = state.get('global_step', 31000)
    checkpoint_stage_idx = state.get('stage_idx', 10) # 10 means we just finished stage 9
    current_k = ckpt_config.get('conv_k', 9)

    print(f"Resuming at Global Step {global_step}, Stage {checkpoint_stage_idx}, Current k={current_k}")

    # 2. Setup AMP Scaling
    scaler = GradScaler()
    if 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    
    # 3. Initial Compile
    print("Compiling model for turbo speed...")
    model = torch.compile(model)

    # Calculate which index in STAGES to start from
    # If checkpoint_stage_idx is 10 (finished stage 10), we want index 1 (stage 11).
    # If it's 9 (finished stage 9), we want index 0 (stage 10).
    start_offset = 0 if checkpoint_stage_idx < 10 else (checkpoint_stage_idx - 10 + 1)

    for stage_idx in range(start_offset, len(STAGES)):
        stage = STAGES[stage_idx]
        actual_stage_num = stage_idx + 10
        print(f"\nTURBO STAGE {actual_stage_num} | k={stage['conv_k']} | lr={stage['lr']}")
        
        # Expand Kernels if necessary
        if stage['conv_k'] > current_k:
            model = expand_kernels(model, current_k, stage['conv_k'], {**BASE_CONFIG, **stage}, device)
            current_k = stage['conv_k']
            # After expansion, we need to ensure the optimizer sees the new params
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage['lr'])
        else:
            # First iteration or same size resume
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage['lr'])
            if 'optimizer' in state and stage_idx == start_offset:
                try:
                    optimizer.load_state_dict(state['optimizer'])
                    print("  Restored optimizer state.")
                except:
                    print("  Warning: Could not restore optimizer state. Starting fresh for this stage.")

        model.train()
        pbar = tqdm(range(stage['steps']), desc=f"Stage {actual_stage_num} (k={current_k})")
        
        for step in pbar:
            X, Y = sampler.get_batch('train')
            
            # --- Turbo Training Step (Mixed Precision) ---
            with autocast():
                _, loss = model(X, Y)
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # ----------------------------------------------
            
            global_step += 1

            if global_step % 250 == 0:
                # Direct evaluation (no autocast here for precision)
                # We average over 50 batches to match the previous logs exactly
                model.eval()
                with torch.no_grad():
                    eval_iters = 50
                    v_losses = torch.zeros(eval_iters)
                    for i in range(eval_iters):
                        vX, vY = sampler.get_batch('val')
                        _, v_loss_batch = model(vX, vY)
                        v_losses[i] = v_loss_batch.item()
                    v_loss = v_losses.mean()
                    
                    msg = (f"Turbo Stage {actual_stage_num} | Step {global_step}: "
                           f"Train {loss.item():.4f}, Val {v_loss:.4f} (k={current_k})")
                    tqdm.write(msg)
                    with open(log_path, "a") as f:
                        f.write(msg + "\n")
                model.train()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save stage
        save_checkpoint(model, optimizer, scaler, global_step, stage_idx + 10, 
                        {**BASE_CONFIG, **stage, 'vocab_size': vocab_size}, 
                        os.path.join(args.out_dir, f"stage{stage_idx+10}.pth"))

    print(f"\nTURBO TRAINING COMPLETE! Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
