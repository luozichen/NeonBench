"""
Extended Growth Trainer for neon213 (k=9 to k=21).
Resumes from a k=9 checkpoint and expands to k=21.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
from models.neon213 import Neon213, Block

# ============================================================
# Extended Stages
# ============================================================
# We resume from the end of the previous training (Stage 9, k=9)
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

# ============================================================
# Growth Functions
# ============================================================
def expand_kernels(model, old_k, new_k, config, device):
    """Expand all conv kernels from old_k to new_k with zero-padding."""
    d_model = config['d_model']
    pad_size = new_k - old_k
    for block in model.blocks:
        for conv_name in ['conv_q', 'conv_k', 'conv_v', 'conv_i']:
            old_conv = getattr(block.attn, conv_name)
            new_conv = nn.Conv1d(d_model, d_model, kernel_size=new_k,
                                 groups=d_model, bias=False).to(device)
            with torch.no_grad():
                new_conv.weight.zero_()
                # Right-side padding preserves causal history mapping
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
    print(f"  [GROWTH] Expanded all conv kernels: k={old_k} -> k={new_k}")
    return model

# ============================================================
# Data (Memory-Mapped)
# ============================================================
class DataSampler:
    def __init__(self, data_path, block_size, batch_size, train_frac=0.9):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        n = len(self.data)
        self.split = int(train_frac * n)
        self.block_size = block_size
        self.batch_size = batch_size
        print(f"Loaded {n:,} tokens (memmap). Train: {self.split:,}, Val: {n - self.split:,}")

    def get_batch(self, split='train'):
        hi = self.split if split == 'train' else len(self.data)
        lo = 0 if split == 'train' else self.split
        max_idx = hi - self.block_size - 1
        idxs = np.random.randint(lo, max_idx, size=self.batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int64) for i in idxs])
        y = np.stack([self.data[i+1 : i+1 + self.block_size].astype(np.int64) for i in idxs])
        return torch.from_numpy(x).to(BASE_CONFIG['device']), torch.from_numpy(y).to(BASE_CONFIG['device'])

def estimate_loss(model, sampler, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for i in range(eval_iters):
            X, Y = sampler.get_batch('val')
            _, loss = model(X, Y)
            losses[i] = loss.item()
    model.train()
    return losses.mean()

def save_checkpoint(model, optimizer, global_step, stage_idx, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
        'stage_idx': stage_idx,
        'config': config,
    }, path)
    print(f"  Saved checkpoint -> {path}")

def main():
    parser = argparse.ArgumentParser(description="Extended Growth Trainer")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True, help="Path to neon213_final.pth")
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon213_extended")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--eval_interval", type=int, default=500)
    args = parser.parse_args()

    device = BASE_CONFIG['device']
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    sampler = DataSampler(args.data, BASE_CONFIG['block_size'], BASE_CONFIG['batch_size'])

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "neon213_extended_log.txt")

    # Load Base Model (k=9)
    print(f"Loading base model from {args.resume}...")
    # We must manually specify the k=9 config to load the state dict correctly
    base_config = {**BASE_CONFIG, 'vocab_size': vocab_size, 'n_layers': 8, 'conv_k': 9, 'mlp_k': 9}
    model = Neon213(base_config).to(device)
    state = torch.load(args.resume, map_location=device)
    # Check if we are loading from a full trainer checkpoint or just the state_dict
    if 'model' in state:
        model.load_state_dict(state['model'])
        global_step = state['global_step']
    else:
        model.load_state_dict(state)
        global_step = 31000 # Assume we finished stage 9

    current_k = 9
    
    for stage_idx, stage in enumerate(STAGES):
        print(f"\nEXTENDED STAGE {stage_idx + 10} | k={stage['conv_k']} | steps={stage['steps']}")
        
        # Expand Kernels
        config = {**BASE_CONFIG, **stage, 'vocab_size': vocab_size}
        model = expand_kernels(model, current_k, stage['conv_k'], config, device)
        model.config = config
        current_k = stage['conv_k']
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=stage['lr'])
        
        model.train()
        pbar = tqdm(range(stage['steps']), desc=f"k={current_k}")
        for step in pbar:
            X, Y = sampler.get_batch('train')
            _, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_interval == 0:
                val_loss = estimate_loss(model, sampler, device)
                msg = (f"Extended Stage {stage_idx+10} | Step {global_step}: "
                       f"Train {loss.item():.4f}, Val {val_loss:.4f} (k={current_k})")
                tqdm.write(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save stage checkpoint
        save_checkpoint(model, optimizer, global_step, stage_idx + 10, config, 
                        os.path.join(args.out_dir, f"stage{stage_idx+10}.pth"))

    # Final Save
    final_path = os.path.join(args.out_dir, "neon213_k21_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nEXTENDED TRAINING COMPLETE! Final kernel size: {current_k}")
    print(f"Final model saved to: {final_path}")

if __name__ == "__main__":
    main()
