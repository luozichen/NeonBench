import matplotlib.pyplot as plt
import os
import re
import argparse
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
    parser = argparse.ArgumentParser(description="NeonBench Unified Gauntlet Plotter")
    parser.add_argument("--models", type=str, default="neon081,neon085,neon100", help="Comma-separated model IDs")
    parser.add_argument("--data", type=str, default="hp0", help="Dataset name (hp0, wiki103, etc.)")
    parser.add_argument("--tok", type=str, default="tok4", help="Tokenizer name (tok1, tok4, etc.)")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing logs")
    parser.add_argument("--out", type=str, default=None, help="Output filename")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for Y-axis")
    parser.add_argument("--title", type=str, default=None, help="Custom chart title")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    out_file = args.out if args.out else f"plot_{args.data}_{args.tok}.png"
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # Premium Color Palette
    colors = ['#00e5ff', '#ff9100', '#27ae60', '#e74c3c', '#9b59b6', '#f1c40f', '#3498db', '#e67e22']
    
    found_any = False
    for i, model in enumerate(models):
        # Try both common naming conventions
        log_paths = [
            os.path.join(args.log_dir, f"{model}_{args.tok}_{args.data}_log.txt"),
            os.path.join(args.log_dir, f"{model}_{args.tok}_{args.data}_{args.tok}_log.txt"),
            os.path.join(args.log_dir, f"{model}_{args.tok}_{args.data.replace('0','')}_log.txt")
        ]
        
        steps, losses = [], []
        for p in log_paths:
            s, l = parse_log(p)
            if s:
                steps, losses = s, l
                break
        
        if steps:
            min_loss = min(losses)
            label = f"{model} (min: {min_loss:.4f})"
            ax.plot(steps, losses, label=label, color=colors[i % len(colors)], linewidth=2, alpha=0.9)
            found_any = True
            
    if not found_any:
        print("Error: No logs found for the specified models/data/tok combination.")
        return

    # Titles & Labels
    title = args.title if args.title else f"NeonBench SOTA Comparison: {args.data.upper()} ({args.tok})"
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    
    if args.log_scale:
        ax.set_yscale('log')
        ax.set_ylabel("Validation Loss (Log Scale)", fontsize=11)

    ax.legend(fontsize=10, loc='upper right', frameon=True, framealpha=0.1)
    ax.grid(True, linestyle="--", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Success: Plot saved to {out_file}")

if __name__ == "__main__":
    main()
