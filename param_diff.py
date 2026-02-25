import sys
sys.path.append('.')
from models.neon233 import Neon233
from models.neon238 import Neon238

cfg3 = {'vocab_size': 32000, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1170, 'block_size': 1024, 'device': 'cpu'}
cfg8 = {'vocab_size': 32000, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1072, 'block_size': 1024, 'device': 'cpu'}

m3 = Neon233(cfg3)
m8 = Neon238(cfg8)

def get_counts(m):
    return {n: p.numel() for n, p in m.named_parameters()}

c3 = get_counts(m3)
c8 = get_counts(m8)

print('Neon233 Total:', sum(c3.values()))
print('Neon238 Total:', sum(c8.values()))

all_keys = sorted(list(set(c3.keys()) | set(c8.keys())))
for k in all_keys:
    v3 = c3.get(k, 0)
    v8 = c8.get(k, 0)
    if v3 != v8:
        print(f'{k:30} 233:{v3:<8} 238:{v8:<8} Diff:{v8-v3}')
