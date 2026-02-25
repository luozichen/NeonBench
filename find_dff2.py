import sys
sys.path.append('.') # Add NeonBench to path
from check_params import count_non_embed
from models.neon238 import Neon238

def get_config(d_ff):
    return {'vocab_size': 32000, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': d_ff, 'block_size': 1024, 'device': 'cpu'}

target = 5005616
best_diff = 1e9
best_dff = -1
best_val = -1

for d_ff in range(500, 1500):
   config = get_config(d_ff)
   model = Neon238(config)
   params = count_non_embed(model)
   diff = abs(params - target)
   if diff < best_diff:
       best_diff = diff
       best_dff = d_ff
       best_val = params

print(f"BEST d_ff={best_dff} gives {best_val} (diff: {best_diff})")
