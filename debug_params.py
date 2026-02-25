import sys
sys.path.append('.') # Add NeonBench to path
from models.neon238 import Neon238

def get_config(d_ff):
    return {'vocab_size': 32000, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': d_ff, 'block_size': 1024, 'device': 'cpu'}

config = get_config(1072)
model = Neon238(config)

shared_params = model.token_emb.weight.numel()
total_params = sum(p.numel() for p in model.parameters())
grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Shared (Embed = Head): {shared_params}")
print(f"Total Params: {total_params}")
print(f"Grad Params: {grad_params}")
print(f"Total Non-Embed: {total_params - shared_params}")
print(f"Grad Non-Embed: {grad_params - shared_params}")

for name, p in model.named_parameters():
    if not p.requires_grad:
        print(f"FROZEN: {name} | {p.numel()}")
