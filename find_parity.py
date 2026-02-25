import sys
sys.path.append('.')
from check_params import count_non_embed
from models.neon257 import Neon257

target = 5004528
cfg = {
    'vocab_size': 32000,
    'd_model': 272,
    'n_layers': 4,
    'n_head': 4,
    'block_size': 1024,
    'device': 'cpu'
}

for d_ff in range(1072, 800, -1):
    cfg['d_ff'] = d_ff
    count = count_non_embed(Neon257(cfg))
    if count <= target:
        print(f"FOUND PARITY: d_ff={d_ff} yields {count} params. Diff: {target-count}")
        break
