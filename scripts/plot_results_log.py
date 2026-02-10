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
    parser.add_argument("--tokenizers", type=str, default="tok1", help="Comma-separated tokenizer names")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: neon001-neon022)")
    args = parser.parse_args()

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = [f"neon{i:03d}" for i in range(1, 25)]

    tokenizers = [t.strip() for t in args.tokenizers.split(",")]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
              '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
              '#dbdb8d', '#9edae5', '#393b79', '#637939',
              '#e7298a', '#66a61e']

    if len(tokenizers) > 1:
        fig, axes = plt.subplots(1, len(tokenizers), figsize=(7 * len(tokenizers), 6), sharey=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        axes = [ax]

    for tok_idx, tok in enumerate(tokenizers):
        ax = axes[tok_idx] if len(tokenizers) > 1 else axes[0]
        found_any = False

        for model_idx, model in enumerate(models):
            log_path = os.path.join(args.log_dir, f"{model}_{tok}_{args.data_name}_log.txt")
            steps, losses = parse_log(log_path)
            if steps:
                ax.plot(steps, losses, label=model,
                        color=colors[model_idx % len(colors)], linewidth=1.5)
                found_any = True

        if not found_any:
            ax.text(0.5, 0.5, "No log files found", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        ax.set_xlabel("Steps")
        ax.set_ylabel("Validation Loss (log scale)")
        ax.set_yscale('log')  # LOGARITHMIC Y-AXIS
        ax.set_title(f"{tok.upper()} - {args.data_name}")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3, which='both')  # Grid for major and minor ticks

    plt.suptitle(f"Neon Models Validation Loss Comparison (Log Scale)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = f"val_loss_comparison_{args.data_name}_log.png"
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
