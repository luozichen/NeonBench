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
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="neon081,neon085,neon100")
    parser.add_argument("--data", type=str, default="hp0")
    parser.add_argument("--tok", type=str, default="tok4")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    out_file = args.out if args.out else f"sota_{args.data}_{args.tok}.png"

    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    
    colors = {
        "neon081": "#e67e22", # Orange
        "neon085": "#27ae60", # Green
        "neon100": "#00e5ff"  # Cyan (The New King)
    }

    for model in models:
        # Check for wiki103 specific naming if data is wiki103
        if "wiki" in args.data:
            log_path = os.path.join("logs", f"{model}_{args.tok}_{args.data}_log.txt")
            if not os.path.exists(log_path):
                # Fallback for the slightly longer name variant
                log_path = os.path.join("logs", f"{model}_{args.tok}_{args.data}_{args.tok}_log.txt")
        else:
            log_path = os.path.join("logs", f"{model}_{args.tok}_{args.data}_log.txt")
            
        steps, losses = parse_log(log_path)
        
        if not steps:
            continue
            
        plt.plot(steps, losses, label=f"{model} (min: {min(losses):.4f})", 
                 color=colors.get(model, None), linewidth=2, alpha=0.9)

    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.title(f"SOTA Comparison: {args.data.upper()} ({args.tok})", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
