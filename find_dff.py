import sys
import torch
import torch.nn as nn

sys.path.append('.') # Add NeonBench to path
from check_params import count_non_embed
from models.neon238 import Neon238

def get_config(d_ff):
    return {
        'vocab_size': 32000,
        'd_model': 272,
        'n_layers': 4,
        'n_head': 4,
        'd_ff': d_ff,
        'block_size': 1024,
        'device': 'cpu'
    }

target = 5005616
for d_ff in range(700, 1000):
   config = get_config(d_ff)
   model = Neon238(config)
   params = count_non_embed(model)
   if params == target:
       print("FOUND EXACT PARITY d_ff =", d_ff)
       break
   if params > target:
       print(f"Exceeded target at d_ff={d_ff}: {params}")
       break
