import sys
import os
import torch

# Add project root to path
sys.path.append(os.getcwd())
from train import get_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_params():
    models = ["neon167", "neon183", "neon184"]
    
    print(f"{'Model':<10} | {'Config d_ff':<12} | {'Parameters':>12}")
    print("-" * 40)
    
    for m_name in models:
        try:
            cfg = get_config(m_name)
            # Import dynamically
            module = __import__(f"models.{m_name}", fromlist=[m_name.capitalize()])
            ModelClass = getattr(module, m_name.capitalize())
            model = ModelClass(cfg)
            p = count_parameters(model)
            print(f"{m_name:<10} | {cfg['d_ff']:<12} | {p:>12,}")
        except Exception as e:
            print(f"{m_name:<10} | Error: {e}")

if __name__ == "__main__":
    verify_params()
