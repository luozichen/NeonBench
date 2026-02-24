import torch
import sys
import os
sys.path.append(os.getcwd())
from models.neon213 import Neon213
from models.neon230 import Neon230
from train import get_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

vocab_size = 50257
cfg213 = get_config("neon213")
cfg213.update({'vocab_size': vocab_size, 'n_layers': 8})
m213 = Neon213(cfg213)
p213 = count_parameters(m213)

print(f"Neon213 (8-layer Reference) Params: {p213:,}")

cfg230 = get_config("neon230")
cfg230.update({
    'vocab_size': vocab_size,
    'd_model': 384,
    'n_head': 6,
    'd_ff': 1536,
    'n_layers': 8
})
m230 = Neon230(cfg230)
p230 = count_parameters(m230)
print(f"Neon230 (Draft) Params: {p230:,}")

# Find d_ff for neon230 to match p213
target = p213

# Linear interp
cfg230['d_ff'] = 100
p1 = count_parameters(Neon230(cfg230))
cfg230['d_ff'] = 200
p2 = count_parameters(Neon230(cfg230))
dp_dff = (p2 - p1) / 100.0

best_d_ff = int(round(100 + (target - p1) / dp_dff))
print(f"Recommended d_ff for Neon230: {best_d_ff}")

cfg230['d_ff'] = best_d_ff
p_final = count_parameters(Neon230(cfg230))
print(f"Neon230 (Final) Params: {p_final:,} (Diff: {p_final - target})")
