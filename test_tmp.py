"""Verify neon212."""
import torch, sys, os
sys.path.append(os.getcwd())
config = {'vocab_size': 8192, 'd_model': 240, 'n_layers': 5, 'n_head': 4, 'd_ff': 981, 'block_size': 256}
from models.neon212 import Neon212
model = Neon212(config)
total = sum(p.numel() for p in model.parameters() if p.requires_grad)
emb = model.token_emb.weight.numel()
print(f"neon212: total={total:,} emb={emb:,} non_emb={total-emb:,} d={config['d_model']} ff={config['d_ff']} layers={config['n_layers']} ratio={config['d_ff']/config['d_model']:.2f}x")
x = torch.randint(0, 8192, (2, 256))
_, loss = model(x, x)
print(f"Forward pass OK, loss={loss.item():.4f}")
