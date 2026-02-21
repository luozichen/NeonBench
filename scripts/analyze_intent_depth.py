import torch
import os
import numpy as np

checkpoint_path = "/home/luozichen/NeonBench/checkpoints/neon213_turbo/neon213_k21_final.pth"

def analyze_intent_kernels():
    if not os.path.exists(checkpoint_path):
        print(f"File not found: {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(f"Deep Analysis of Intent Kernels (k=21)")
    print("=" * 60)

    for l in range(8):
        k_key = f"blocks.{l}.attn.conv_i.weight"
        if k_key not in state_dict: continue
        
        # w shape: [D, 1, K] (Depthwise Conv)
        w = state_dict[k_key].squeeze(1) # [D, K]
        D, K = w.shape
        
        # Per-channel sums
        c_sums = w.sum(dim=1)
        c_means = w.mean(dim=1)
        
        # How many channels have a positive sum?
        pos_sum_ratio = (c_sums > 0).float().mean().item()
        
        # Look for "Master Gaters" (channels with huge positive sums)
        top_sum, top_idx = torch.topk(c_sums, 3)
        bot_sum, bot_idx = torch.topk(c_sums, 3, largest=False)
        
        print(f"Layer {l}:")
        print(f"  Pos-Sum Channels: {pos_sum_ratio:.1%} (Ideal is 50% if random)")
        print(f"  Channel Sums: Mean={c_sums.mean():.4f}, Std={c_sums.std():.4f}, Max={c_sums.max():.4f}, Min={c_sums.min():.4f}")
        print(f"  Top 3 Gating Channels (Sums): {top_sum.tolist()}")
        print(f"  Top 3 Muting Channels (Sums): {bot_sum.tolist()}")
        
        # Sample a "vibrant blue" kernel if it exists
        master_kernel = w[top_idx[0]].tolist()
        print(f"  Impulse Response of Channel {top_idx[0].item()} (Master Gater):")
        # Format as tiny bar chart
        for val in master_kernel:
            bar = ("#" * int(abs(val)*40)) if abs(val) > 0.01 else "."
            sign = "+" if val >= 0 else "-"
            print(f"    {sign} {val:6.3f} {bar}")
        print("-" * 40)

if __name__ == "__main__":
    analyze_intent_kernels()
