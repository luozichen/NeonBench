import torch
import os

# Load the finalized FP32 weights (easier to analyze than FP16)
checkpoint_path = "/home/luozichen/NeonBench/checkpoints/neon213_turbo/neon213_k21_final.pth"

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Analyzing Weights for {os.path.basename(checkpoint_path)}")
    print("-" * 50)
    
    for layer_idx in range(8):
        # Look for conv_i weights
        weight_key = f"blocks.{layer_idx}.attn.conv_i.weight"
        if weight_key in state_dict:
            w = state_dict[weight_key] # Shape [D, 1, K] or [D, K]
            w_mean = w.mean().item()
            w_std = w.std().item()
            w_pos_ratio = (w > 0).float().mean().item()
            
            # Look at a few specific dimensions to see the patterns
            # Dimensions are basically per-head/per-channel "experts"
            sample_w = w[:5].view(5, -1)
            
            print(f"Layer {layer_idx} Intent Kernel Stats:")
            print(f"  Mean: {w_mean:.4f} | Std: {w_std:.4f}")
            print(f"  Positivity Ratio: {w_pos_ratio:.2%}")
            # print(f"  Sample Kernel (dim 0): {sample_w[0].tolist()}")
            print("-" * 30)
else:
    print(f"File not found: {checkpoint_path}")
