import sys
sys.path.append('.')
from train import get_config
from check_params import count_non_embed
from models.neon238 import Neon238

c = get_config('neon238')
c['vocab_size'] = 32000
print(c)
m = Neon238(c)
print(count_non_embed(m))
print(sum(p.numel() for p in m.parameters() if p.requires_grad) - m.token_emb.weight.numel())
