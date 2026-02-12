
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.getcwd())
from models.neon079 import Qwen3NextGatedDeltaNet, Qwen3NextAttention, Neon079

def test_delta_net():
    print("Testing Qwen3NextGatedDeltaNet...")
    config = {
        'd_model': 256,
        'n_head': 4,
        'rms_norm_eps': 1e-6,
        'hidden_act': 'silu'
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = Qwen3NextGatedDeltaNet(config).to(device)
    x = torch.randn(2, 64, 256).to(device) # B, T, D
    
    # Forward
    try:
        y = model(x, None, None)
        print(f"Forward Output shape: {y.shape}")
        if torch.isnan(y).any():
            print("ERROR: NaNs detected in forward pass!")
            # Inspect internals if possible (requires hook or stepping)
        else:
             print("Forward Pass: OK")
             
        # Backward
        loss = y.sum()
        loss.backward()
        print("Backward Pass: OK")
        
        # Check grads
        for n, p in model.named_parameters():
             if p.grad is not None:
                 if torch.isnan(p.grad).any():
                      print(f"ERROR: NaNs in grad of {n}")
             else:
                 print(f"Warning: No grad for {n}")

    except Exception as e:
        print(f"Crash: {e}")

def test_full_model():
    print("\nTesting Full Neon079...")
    config = {
        'vocab_size': 100,
        'd_model': 256,
        'n_layers': 4,
        'n_head': 4,
        'd_ff': 480,
        'block_size': 64,
        'rms_norm_eps': 1e-6,
        'hidden_act': 'silu'
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Neon079(config).to(device)
    idx = torch.randint(0, 100, (2, 64)).to(device)
    
    try:
        logits, loss = model(idx, idx)
        print(f"Logits shape: {logits.shape}, Loss: {loss.item()}")
        if torch.isnan(loss):
             print("ERROR: Loss is NaN")
             
        loss.backward()
        print("Backward Pass: OK")
        
    except Exception as e:
        print(f"Crash: {e}")

if __name__ == "__main__":
    test_delta_net()
    test_full_model()
