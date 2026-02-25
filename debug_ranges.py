import sys
sys.path.append('.')
from train import get_config

print("Testing Model Configs (Bug verify):")
for i in range(233, 243):
    name = f"neon{i}"
    c = get_config(name)
    print(f"{name}: d_model={c['d_model']}, n_layers={c['n_layers']}, n_head={c['n_head']}, d_ff={c['d_ff']}")
