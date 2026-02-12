import matplotlib.pyplot as plt
import os
import re
import argparse

def parse_log(log_path):
    steps = []
    val_losses = []
    if not os.path.exists(log_path):
        return steps, val_losses

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r"Step (\d+):.*Val Loss ([\d.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
    return steps, val_losses

def plot_comparison(models, tok, data_name, title, out_file, baseline_model="neon016"):
    plt.figure(figsize=(10, 6))
    
    colors = {
        "neon016": "#7f8c8d", # Grey
        "neon091": "#3498db", # Blue
        "neon092": "#e67e22"  # Orange
    }

    for model in models:
        log_file = f"{model}_{tok}_{data_name}_log.txt"
        if data_name == "wiki103_tok4":
             log_file = f"{model}_{tok}_wiki103_tok4_log.txt"
             
        log_path = os.path.join("logs", log_file)
        steps, losses = parse_log(log_path)
        
        if not steps:
            continue
            
        label = f"{model} (min: {min(losses):.4f})"
        plt.plot(steps, losses, label=label, color=colors.get(model), linewidth=2, alpha=0.8)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Val Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Saved {out_file}")

def main():
    models = ["neon016", "neon091", "neon092"]
    
    # 1. HP0 Comparison
    plot_comparison(models, "tok4", "hp0", 
                    "Scaling Audit: HP0 (Tok4) - 3M vs 10M Hydra", 
                    "scaling_hp0_tok4.png")
    
    # 2. Wiki103 Comparison
    plot_comparison(models, "tok4", "wiki103_tok4", 
                    "Scaling Audit: Wiki103 (Tok4) - 3M vs 10M Hydra", 
                    "scaling_wiki_tok4.png")

if __name__ == "__main__":
    main()
