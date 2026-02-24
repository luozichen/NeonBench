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
    
    cfg0 = get_config(m_name)
    cfg0['d_ff'] = 100
    m0 = ModelClass(cfg0)
    p0 = count_parameters(m0)
    
    cfg1 = get_config(m_name)
    cfg1['d_ff'] = 200
    m1 = ModelClass(cfg1)
    p1 = count_parameters(m1)
    
    dp_dff = (p1 - p0) / 100.0
    
    # We want: p0 + (d_ff - 100) * dp_dff = target_total
    # d_ff - 100 = (target_total - p0) / dp_dff
    best_d_ff = int(round(100 + (target_total - p0) / dp_dff))
    
    # Verify
    cfg_final = get_config(m_name)
    cfg_final['d_ff'] = best_d_ff
    m_final = ModelClass(cfg_final)
    p_final = count_parameters(m_final)
    
    print(f"{m_name:<10} | d_ff: {best_d_ff:<5} | params: {p_final:>12,} | diff: {p_final - target_total}")
