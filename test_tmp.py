"""Verify neon213 parameter counts at each growth stage."""
import torch, sys, os
sys.path.append(os.getcwd())
from models.neon213 import Neon213, Block
import torch.nn as nn

base = {'vocab_size': 16384, 'd_model': 384, 'n_head': 6, 'd_ff': 1536, 'block_size': 256}

stages = [
    {'n_layers': 4, 'conv_k': 1, 'mlp_k': 1},
    {'n_layers': 5, 'conv_k': 1, 'mlp_k': 1},
    {'n_layers': 6, 'conv_k': 1, 'mlp_k': 1},
    {'n_layers': 7, 'conv_k': 1, 'mlp_k': 1},
    {'n_layers': 8, 'conv_k': 1, 'mlp_k': 1},
    {'n_layers': 8, 'conv_k': 3, 'mlp_k': 3},
    {'n_layers': 8, 'conv_k': 9, 'mlp_k': 9},
]

print(f"{'Stage':<8} {'Layers':<8} {'K':<4} {'Total':>12} {'Emb':>10} {'Non-Emb':>12}")
print("-" * 58)

for i, s in enumerate(stages):
    config = {**base, **s}
    model = Neon213(config)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.token_emb.weight.numel()
    non_emb = total - emb
    print(f"  {i+1:<6} {s['n_layers']:<8} {s['conv_k']:<4} {total:>12,} {emb:>10,} {non_emb:>12,}")

# Forward pass test at final stage
print("\nForward pass test (8 layers, k=9)...")
config = {**base, 'n_layers': 8, 'conv_k': 9, 'mlp_k': 9}
model = Neon213(config)
x = torch.randint(0, 16384, (2, 256))
_, loss = model(x, x)
print(f"OK! Loss={loss.item():.4f}")

# Test growth functions
print("\nTesting layer growth (4→5)...")
config4 = {**base, 'n_layers': 4, 'conv_k': 1, 'mlp_k': 1}
model = Neon213(config4)
x = torch.randint(0, 16384, (2, 256))
_, loss_before = model(x, x)
print(f"  Before: {len(model.blocks)} layers, loss={loss_before.item():.4f}")

# Add layer with identity init
new_block = Block(config4)
nn.init.zeros_(new_block.attn.c_proj.weight)
nn.init.zeros_(new_block.mlp.w2.weight)
model.blocks.append(new_block)

_, loss_after = model(x, x)
print(f"  After:  {len(model.blocks)} layers, loss={loss_after.item():.4f}")
print(f"  Loss change: {abs(loss_after.item() - loss_before.item()):.6f} (should be ~0)")
