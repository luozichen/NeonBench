"""Smoke test for Neon301 Gram-Schmidt model with torch.compile."""
import torch, sys; sys.path.append('.')
from models.neon301 import Neon301

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = {'vocab_size': 1000, 'd_model': 512, 'n_layers': 4, 'n_head': 8, 'd_ff': 2048, 'block_size': 64}
m = Neon301(cfg).to(device)
m = torch.compile(m)
x = torch.randint(0, 1000, (2, 64)).to(device)

# Forward and backward
logits, loss = m(x, x)
loss.backward()

print(f"Neon301 (compiled) OK -- logits: {logits.shape}, loss: {loss.item():.4f}")

print(f"Neon301 (compiled) OK -- logits: {logits.shape}, loss: {loss.item():.4f}")
