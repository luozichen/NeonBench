import matplotlib.pyplot as plt
import os
import re
import argparse

def parse_log(log_path):
    steps = []
    val_losses = []
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return steps, val_losses

    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r"Step (\d+):.*Val Loss ([\d.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
    return steps, val_losses

def main():
    log_dir = "logs"
    models = ["neon016", "neon081", "neon085"]
    tok = "tok4"
    data = "hp0"
    out_file = "sota_comparison.png"

    plt.figure(figsize=(12, 7))
    
    colors = {
        "neon016": "#7f8c8d", # Grey (Baseline)
        "neon081": "#e67e22", # Orange (Milestone)
        "neon085": "#27ae60"  # Green (SOTA)
    }
    
    styles = {
        "neon016": "--",
        "neon081": "-",
        "neon085": "-"
    }

    for model in models:
        log_path = os.path.join(log_dir, f"{model}_{tok}_{data}_log.txt")
        steps, losses = parse_log(log_path)
        
        if not steps:
            continue
            
        label = f"{model} (Val Loss: {min(losses):.4f})"
        if model == "neon016":
            label = f"{model} (Baseline: {min(losses):.4f})"
            
        plt.plot(steps, losses, label=label, color=colors.get(model, None), 
                 linestyle=styles.get(model, "-"), linewidth=2, marker='o', markersize=4, alpha=0.8)

    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.title("SOTA Comparison: neon085 (Dual-Scale Hydra) vs neon016 (Baseline)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Annotate the gap
    baseline_min = 0.9174 # From README/Log
    sota_min = 0.8670
    gap = baseline_min - sota_min
    plt.annotate(f"Improvement: {gap:.4f} ({gap/baseline_min*100:.1f}%)", 
                 xy=(9500, 0.87), xytext=(6000, 0.83),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
