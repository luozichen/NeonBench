import torch
import sys
import os

# Add current directory to sys.path to import train and models
sys.path.append(os.getcwd())

from train import get_config
from models.neon081 import Neon081
from models.neon116 import Neon116
from models.neon129 import Neon129
from models.neon130 import Neon130
from models.neon131 import Neon131

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def audit():
    models = ["neon081", "neon116", "neon130", "neon132", "neon133", "neon134", "neon135", "neon136", "neon137", "neon138"]
    
    print(f"--- PARAMS AUDIT ---")
    print(f"{'Model':<10} | {'d_ff':<5} | {'Parameters':>12}")
    print("-" * 35)
    
    for m_name in models:
        cfg = get_config(m_name)
        # Import dynamically
        module = __import__(f"models.{m_name}", fromlist=[m_name.capitalize()])
        ModelClass = getattr(module, m_name.capitalize())
        m = ModelClass(cfg)
        p = count_parameters(m)
        print(f"{m_name:<10} | {cfg['d_ff']:<5} | {p:>12,}")
    print(f"--------------------")

if __name__ == "__main__":
    audit()
