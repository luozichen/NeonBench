import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())
from train import get_config
from models.neon185 import Neon185
from models.neon230 import Neon230
from models.neon232 import Neon232

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vocab_size = 50257
cfg185 = get_config("neon185")
cfg185['vocab_size'] = vocab_size
m185 = Neon185(cfg185)
p185 = count_parameters(m185)
print(f"Neon185 (Target) Params: {p185:,}")

def balance(name, ModelClass, d_ff_start=1072):
    best_d_ff = d_ff_start
    best_diff = float('inf')
    
    # Simple search for d_ff
    for d_ff in range(d_ff_start - 300, d_ff_start + 500):
        cfg = get_config(name)
        cfg.update({'vocab_size': vocab_size, 'd_ff': d_ff, 'n_layers': 4, 'd_model': 272})
        m = ModelClass(cfg)
        p = count_parameters(m)
        diff = abs(p - p185)
        if diff < best_diff:
            best_diff = diff
            best_d_ff = d_ff
            
    print(f"Recommended d_ff for {name}: {best_d_ff}")
    cfg = get_config(name)
    cfg.update({'vocab_size': vocab_size, 'd_ff': best_d_ff, 'n_layers': 4, 'd_model': 272})
    m = ModelClass(cfg)
    print(f"{name} Final Params: {count_parameters(m):,} (Diff: {count_parameters(m) - p185})")

print("\n--- Balancing ---")
balance("neon230", Neon230)
balance("neon231", Neon231)
balance("neon232", Neon232)
