import sys
import os
sys.path.append(os.getcwd())

from train import get_config
from verify_model_params import count_parameters
from models.neon185 import Neon185

cfg185 = get_config("neon185")
m185 = Neon185(cfg185)
target_total = count_parameters(m185)

print(f"Target total parameters (Neon185): {target_total:,}")

models_to_test = ["neon217", "neon218", "neon219", "neon220", "neon221"]

for m_name in models_to_test:
    module = __import__(f"models.{m_name}", fromlist=[m_name.capitalize()])
    ModelClass = getattr(module, m_name.capitalize())
    
    best_d_ff = -1
    min_diff = float('inf')
    
    for f in range(600, 1500, 1):
        cfg = get_config(m_name)
        cfg['d_ff'] = f
        m = ModelClass(cfg)
        p = count_parameters(m)
        diff = abs(p - target_total)
        if diff < min_diff:
            min_diff = diff
            best_d_ff = f
            
    # Final check
    cfg = get_config(m_name)
    cfg['d_ff'] = best_d_ff
    m = ModelClass(cfg)
    p = count_parameters(m)
    print(f"{m_name:<10} | d_ff: {best_d_ff:<5} | params: {p:>12,} | diff: {p - target_total}")
