import torch
import sys
import os

# Add current directory to sys.path to import train and models
sys.path.append(os.getcwd())

from train import get_config
from models.neon081 import Neon081
from models.neon116 import Neon116

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def audit():
    cfg81 = get_config("neon081")
    cfg116 = get_config("neon116")
    
    m81 = Neon081(cfg81)
    m116 = Neon116(cfg116)
    
    p81 = count_parameters(m81)
    p116 = count_parameters(m116)
    
    print(f"--- PARAMS AUDIT ---")
    print(f"Neon081 (d_ff={cfg81['d_ff']}): {p81:,}")
    print(f"Neon116 (d_ff={cfg116['d_ff']}): {p116:,}")
    print(f"Diff: {abs(p116 - p81):,} ({(p116 - p81) / p81 * 100:+.2f}%)")
    print(f"--------------------")

if __name__ == "__main__":
    audit()
