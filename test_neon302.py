"""Smoke test for Neon302 Gram-Schmidt model."""
import torch, sys; sys.path.append('.')
from models.neon302 import Neon302

cfg = {'vocab_size': 1000, 'd_model': 512, 'n_layers': 4, 'n_head': 8, 'd_ff': 2048, 'block_size': 64}
m = Neon302(cfg)
x = torch.randint(0, 1000, (2, 64))
logits, loss = m(x, x)
loss.backward()
n = sum(p.numel() for p in m.parameters())
print(f"Neon302 (4-layer test) OK -- logits: {logits.shape}, loss: {loss.item():.4f}, params: {n:,}")

# Full config
cfg8 = {'vocab_size': 16384, 'd_model': 512, 'n_layers': 8, 'n_head': 8, 'd_ff': 2048, 'block_size': 512}
m8 = Neon302(cfg8)
n8 = sum(p.numel() for p in m8.parameters())
print(f"Neon302 (full 8-layer, vocab=16384) params: {n8:,}")
