import sys
sys.path.append('.')

from models.neon233 import Neon233
from models.neon238 import Neon238

def get_c(d_ff):
    return {'vocab_size': 8192, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': d_ff, 'block_size': 1024, 'device': 'cpu'}

m233 = Neon233(get_c(1170))
m238 = Neon238(get_c(1072))

def print_train_parity_logic(name, model):
    p_req = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb_size = model.token_emb.weight.numel()
    calc = p_req - emb_size
    print(f"{name} (Train Parity Logic): {calc:,}")

print_train_parity_logic("neon233", m233)
print_train_parity_logic("neon238", m238)
