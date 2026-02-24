import torch
import sys

def check_decay():
    ckpt_path = "checkpoints/neon220_tok5_wiki103_tok5_best.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Filter for alpha_raw parameters
    decay_params = {k: v for k, v in state_dict.items() if "alpha_raw" in k}
    
    if not decay_params:
        print("No 'alpha_raw' parameters found in state_dict.")
        return

    print("\nLearned Decay Analysis (alpha = sigmoid(alpha_raw))")
    print("-" * 50)
    print(f"{'Layer':<15} | {'Mean Alpha':<12} | {'Min Alpha':<12} | {'Max Alpha':<12}")
    print("-" * 50)
    
    for key in sorted(decay_params.keys()):
        alpha_raw = decay_params[key]
        alpha = torch.sigmoid(alpha_raw)
        
        mean_v = alpha.mean().item()
        min_v = alpha.min().item()
        max_v = alpha.max().item()
        
        # Extract layer index from key like 'blocks.0.attn.alpha_raw'
        layer = key.split(".")[1]
        
        print(f"Layer {layer:<8} | {mean_v:<12.4f} | {min_v:<12.4f} | {max_v:<12.4f}")

    print("-" * 50)
    print("Note: If Alpha = 0.0, the model is memoryless (identical to neon185).")
    print("      If Alpha > 0.0, the model is retaining intent history across layers.")

if __name__ == "__main__":
    check_decay()
