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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--data_name", type=str, default="hp0")
    parser.add_argument("--tok", type=str, default="tok1")
    parser.add_argument("--out", type=str, default="custom_plot.png")
    args = parser.parse_args()

    # Define groups
    group_dashed = ["neon009", "neon010", "neon016"]
    group_dotted = [f"neon{i:03d}" for i in range(31, 41)]
    group_solid = [f"neon{i:03d}" for i in range(41, 51)]
    
    all_models = group_dashed + group_dotted + group_solid

    plt.figure(figsize=(12, 8))
    
    # Use a bigger color map
    cm = plt.get_cmap('tab20')
    
    for i, model in enumerate(all_models):
        log_path = os.path.join(args.log_dir, f"{model}_{args.tok}_{args.data_name}_log.txt")
        steps, losses = parse_log(log_path)
        
        if not steps:
            continue
            
        # Determine style
        style = '-'
        if model in group_dashed:
            style = '--'
        elif model in group_dotted:
            style = ':'
        
        plt.plot(steps, losses, label=model, linestyle=style, color=cm(i % 20), linewidth=1.5)

    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.title(f"Comparison: Reference (Dashed) vs Calc (Dotted) vs Gated-Calc (Solid)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved to {args.out}")

if __name__ == "__main__":
    main()
