import matplotlib.pyplot as plt
import os
import re
import argparse

def parse_log_blocks(filename):
    """
    Parses a training log file.
    Looking for lines like:
    Step 9500 | Train Loss: 1.2588 | Val Loss: 1.2551 | Time: 12.3s
    """
    steps = []
    val_losses = []
    
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return [], []
        
    pattern = re.compile(r"Step (\d+):.*Val Loss ([\d\.]+)")
    
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
                
    return steps, val_losses

def main():
    parser = argparse.ArgumentParser(description="Plot specific models with specific tokenizers.")
    parser.add_argument("--out", type=str, default="comparison_tok_scaling.png", help="Output filename")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing logs")
    
    args = parser.parse_args()
    
    # Define experiment parameters
    # Models: Determins Color
    models = ["neon016", "neon027", "neon055"]
    model_colors = {
        "neon016": "tab:blue",
        "neon027": "tab:orange",
        "neon055": "tab:green"
    }

    # Tokenizers: Determins Line Style
    # tok1 = solid, tok3 = dashed, tok4 = dotted
    tokenizers = ["tok1", "tok3", "tok4"]
    tok_styles = {
        "tok1": "-",
        "tok3": "--",
        "tok4": ":"
    }

    experiments = []
    for model in models:
        for tok in tokenizers:
            experiments.append((model, tok, f"{model} ({tok})"))
    
    plt.figure(figsize=(10, 6))
    
    print(f"{'Model':<15} | {'Min Val Loss':<12} | {'Final Val Loss':<12}")
    print("-" * 45)
    
    for model, tok, label in experiments:
        filename = os.path.join(args.log_dir, f"{model}_{tok}_hp0_log.txt")
        steps, losses = parse_log_blocks(filename)
        
        if not steps:
            print(f"Skipping {label} (No data)")
            continue
            
        color = model_colors.get(model, "black")
        style = tok_styles.get(tok, "-")
        
        # Plot
        plt.plot(steps, losses, label=label, linestyle=style, color=color, linewidth=1.5)
        
        min_loss = min(losses)
        final_loss = losses[-1]
        print(f"{label:<15} | {min_loss:.4f}       | {final_loss:.4f}")

    plt.xlabel("Steps")
    plt.ylabel("Validation Loss")
    plt.title("Architecture Robustness across Vocab Sizes\nSolid=tok1(1k), Dashed=tok3(2k), Dotted=tok4(4k)")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(args.out, dpi=150)
    print(f"\nPlot saved to {args.out}")

if __name__ == "__main__":
    main()
