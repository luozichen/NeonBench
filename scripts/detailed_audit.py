import sys
import os
sys.path.append(os.getcwd())
import torch
from models.neon167 import Neon167
from train import get_config

config = get_config('neon167')
print(f"Config: {config}")

model = Neon167(config)
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,}")

total = sum(p.numel() for p in model.parameters() if p.requires_grad)
embed = model.token_emb.weight.numel()
non_embed = total - embed

print(f"\nNon-Embedding Total: {non_embed:,}")
