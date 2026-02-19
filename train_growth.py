"""Progressive Growth Trainer for neon213.
Trains a model through multiple stages, growing layers (4→8) and expanding
conv kernels (k=1→k=9). Saves full checkpoints between stages.

Usage:
  python3 train_growth.py --data data/fineweb/fineweb_tok6.bin \
                           --tokenizer tokenizers/fineweb_tok6.json

Resume from checkpoint:
  python3 train_growth.py --data data/fineweb/fineweb_tok6.bin \
                           --tokenizer tokenizers/fineweb_tok6.json \
                           --resume checkpoints/neon213_growth/latest.pth
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from models.neon213 import Neon213, Block

# ============================================================
# Growth Stages
# ============================================================
STAGES = [
    {'n_layers': 4, 'conv_k': 1, 'mlp_k': 1, 'steps': 5000,  'lr': 3e-4},
    {'n_layers': 5, 'conv_k': 1, 'mlp_k': 1, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 6, 'conv_k': 1, 'mlp_k': 1, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 7, 'conv_k': 1, 'mlp_k': 1, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 8, 'conv_k': 1, 'mlp_k': 1, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 8, 'conv_k': 3, 'mlp_k': 3, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 8, 'conv_k': 5, 'mlp_k': 5, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 8, 'conv_k': 7, 'mlp_k': 7, 'steps': 3000,  'lr': 3e-4},
    {'n_layers': 8, 'conv_k': 9, 'mlp_k': 9, 'steps': 5000,  'lr': 3e-4},
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
def add_layer(model, config, device):
    """Add a new Block with identity-init (output projections = 0)."""
    new_block = Block(config).to(device)
    nn.init.zeros_(new_block.attn.c_proj.weight)
    nn.init.zeros_(new_block.mlp.w2.weight)
    model.blocks.append(new_block)
    print(f"  [GROWTH] Added layer {len(model.blocks)}. c_proj and w2 zero-init'd.")
    return model

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
# Data (Memory-Mapped — near-zero RAM usage)
# ============================================================
class BinDataset(Dataset):
    def __init__(self, data_path, block_size, start=0, end=None):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.start = start
        self.end = end if end is not None else len(self.data)
        self.block_size = block_size

    def __len__(self):
        return (self.end - self.start) - self.block_size

    def __getitem__(self, idx):
        i = self.start + idx
        chunk = self.data[i : i + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())
        y = torch.from_numpy(chunk[1:].copy())
        return x, y

def make_loaders(data_path, block_size, batch_size):
    n = len(np.memmap(data_path, dtype=np.uint16, mode='r'))
    split = int(0.9 * n)
    print(f"Loaded {n:,} tokens (memmap). Train: {split:,}, Val: {n - split:,}")
    train_ds = BinDataset(data_path, block_size, start=0, end=split)
    val_ds = BinDataset(data_path, block_size, start=split, end=n)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ============================================================
# Eval / Checkpoint
# ============================================================
def estimate_loss(model, dataloader, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for i, (X, Y) in enumerate(dataloader):
            if i >= eval_iters: break
            X, Y = X.to(device), Y.to(device)
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

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Progressive Growth Trainer")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="checkpoints/neon213_growth")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--eval_interval", type=int, default=500)
    args = parser.parse_args()

    device = BASE_CONFIG['device']

    # Tokenizer (for vocab size only)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    # Data (memory-mapped)
    train_loader, val_loader = make_loaders(
        args.data, BASE_CONFIG['block_size'], BASE_CONFIG['batch_size'])

    # Logging
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "neon213_growth_log.txt")

    # Resume or fresh
    start_stage = 0
    global_step = 0
    ckpt = None

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_stage = ckpt['stage_idx']
        global_step = ckpt['global_step']
        print(f"Resuming from stage {start_stage}, step {global_step}")

    # ============================================================
    # Stage Loop
    # ============================================================
    model = None
    for stage_idx in range(start_stage, len(STAGES)):
        stage = STAGES[stage_idx]
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx + 1}/{len(STAGES)}: "
              f"layers={stage['n_layers']}, conv_k={stage['conv_k']}, "
              f"mlp_k={stage['mlp_k']}, steps={stage['steps']}")
        print(f"{'='*60}")

        config = {**BASE_CONFIG, **stage, 'vocab_size': vocab_size}

        if stage_idx == start_stage and ckpt is not None:
            # Resume from checkpoint
            model = Neon213(ckpt['config']).to(device)
            model.load_state_dict(ckpt['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage['lr'])
            optimizer.load_state_dict(ckpt['optimizer'])
            steps_done = global_step - sum(s['steps'] for s in STAGES[:stage_idx])
            remaining = stage['steps'] - steps_done
            print(f"  Resumed. {steps_done} steps done, {remaining} remaining.")
        else:
            if stage_idx == 0:
                model = Neon213(config).to(device)
            else:
                prev = STAGES[stage_idx - 1]
                if stage['n_layers'] > prev['n_layers']:
                    for _ in range(stage['n_layers'] - prev['n_layers']):
                        model = add_layer(model, config, device)
                    model.config = config
                if stage['conv_k'] > prev['conv_k']:
                    model = expand_kernels(model, prev['conv_k'],
                                           stage['conv_k'], config, device)
                    model.config = config
            remaining = stage['steps']
            optimizer = torch.optim.AdamW(model.parameters(), lr=stage['lr'])

        total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        emb_p = model.token_emb.weight.numel()
        print(f"  Params: {total_p:,} total, {total_p - emb_p:,} non-emb")

        # Train
        model.train()
        train_iter = iter(train_loader)
        pbar = tqdm(range(remaining), desc=f"Stage {stage_idx+1}")

        for _ in pbar:
            try:
                X, Y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                X, Y = next(train_iter)

            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_interval == 0:
                val_loss = estimate_loss(model, val_loader, device)
                msg = (f"Stage {stage_idx+1} | Step {global_step}: "
                       f"Train {loss.item():.4f}, Val {val_loss:.4f} "
                       f"(L={stage['n_layers']}, k={stage['conv_k']})")
                tqdm.write(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Checkpoint
        save_checkpoint(model, optimizer, global_step, stage_idx + 1,
                        config, os.path.join(args.out_dir, f"stage{stage_idx+1}.pth"))
        save_checkpoint(model, optimizer, global_step, stage_idx + 1,
                        config, os.path.join(args.out_dir, "latest.pth"))

    # Final
    torch.save(model.state_dict(), os.path.join(args.out_dir, "neon213_final.pth"))
    print(f"\nTRAINING COMPLETE! Total steps: {global_step}")

if __name__ == "__main__":
    main()
