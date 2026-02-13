import matplotlib.pyplot as plt
import os
import re
import numpy as np

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

def main():
    models = ["neon081", "neon085", "neon100", "neon116"]
    log_dir = "logs"
    out_file = "sota_comparison_multi.png"
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Premium Color Palette
    colors = ['#00e5ff', '#ff9100', '#27ae60', '#e74c3c']
    
    # Plot HP0
    for i, model in enumerate(models):
        path = os.path.join(log_dir, f"{model}_tok4_hp0_log.txt")
        steps, losses = parse_log(path)
        if steps:
            min_loss = min(losses)
            label = f"{model} ({min_loss:.4f})"
            ax1.plot(steps, losses, label=label, color=colors[i], linewidth=2.5, alpha=0.9)
    
    ax1.set_title("HP0 Dataset (3M Scale)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Steps", fontsize=11)
    ax1.set_ylabel("Validation Loss", fontsize=11)
    ax1.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.1)
    ax1.grid(True, linestyle="--", alpha=0.15)
    ax1.set_ylim(0, None) # Start at 0 for satisfaction
    
    # Plot Wiki103
    for i, model in enumerate(models):
        # Try both naming patterns for wiki
        path = os.path.join(log_dir, f"{model}_tok4_wiki103_tok4_log.txt")
        if not os.path.exists(path):
            path = os.path.join(log_dir, f"{model}_tok4_wiki103_log.txt")
            
        steps, losses = parse_log(path)
        if steps:
            min_loss = min(losses)
            label = f"{model} ({min_loss:.4f})"
            ax2.plot(steps, losses, label=label, color=colors[i], linewidth=2.5, alpha=0.9)
            
    ax2.set_title("Wiki103 Dataset (3M Scale)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Steps", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.1)
    ax2.grid(True, linestyle="--", alpha=0.15)
    ax2.set_ylim(0, None) # Start at 0 for satisfaction

    plt.suptitle("NeonBench: Generational SOTA Evolution", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"Success: Plot saved to {out_file}")

if __name__ == "__main__":
    main()
