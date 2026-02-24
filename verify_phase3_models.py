import torch
import sys
import os

# Project Imports
sys.path.append(os.getcwd())
from train import get_config

models = ["neon226", "neon227", "neon228", "neon229", "neon230"]

print("--- Phase 3 & 4 Discovery Model Logic Verification ---")
print(f"{'Model':<10} | {'Status':<10} | {'Message'}")
print("-" * 50)

device = "cuda" if torch.cuda.is_available() else "cpu"

for m_name in models:
    try:
        module = __import__(f"models.{m_name}", fromlist=[m_name.capitalize()])
        ModelClass = getattr(module, m_name.capitalize())
        
        config = get_config(m_name)
        model = ModelClass(config).to(device)
        
        # Test input
        idx = torch.randint(0, config['vocab_size'], (2, config['block_size'])).to(device)
        targets = torch.randint(0, config['vocab_size'], (2, config['block_size'])).to(device)
        
        # Forward pass
        logits, loss = model(idx, targets)
        
        # Backward pass
        loss.backward()
        
        print(f"{m_name:<10} | PASS       | Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"{m_name:<10} | FAIL       | {str(e)}")

print("-" * 50)
