import torch
import torch.nn as nn
import sys
import os
from tokenizers import Tokenizer

sys.path.append(os.getcwd())
from train import get_config
from models.neon185 import Neon185

from models.neon230 import Neon230
from models.neon231 import Neon231
from models.neon232 import Neon232
from models.neon233 import Neon233

def count_non_embed(model):
    shared_params = model.token_emb.weight.numel()
    return sum(p.numel() for p in model.parameters()) - shared_params

vocab_size = 8192 # wiki103_tok5
target = 5004528 # neon185 non-embed

def check(name, ModelClass, d_ff):
    cfg = get_config(name)
    cfg.update({
        'vocab_size': vocab_size, 
        'd_ff': d_ff, 
        'n_layers': 4, 
        'd_model': 272,
        'block_size': 256,
        'n_head': 4
    })
    m = ModelClass(cfg)
    p = count_non_embed(m)
    print(f"{name} (d_ff={d_ff}) Non-Embed: {p:,} (Diff: {p - target})")

print(f"Target Non-Embed (Neon185): {target:,}\n")
check("neon230", Neon230, 1169)
check("neon231", Neon231, 834)
check("neon232", Neon232, 1169)
check("neon233", Neon233, 1169)
