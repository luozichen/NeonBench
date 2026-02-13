import matplotlib.pyplot as plt
import os
import re

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

def plot_ablation(data_name, tok_name="tok4"):
    # Model Mapping
    models = {
        "neon093": {"label": "Baseline (8-Layer Deep)", "color": "#7f8c8d"}, 
        "neon061": {"label": "Intent Only", "color": "#3498db"}, 
        "neon094": {"label": "Hydra Only", "color": "#9b59b6"}, 
        "neon092": {"label": "Full Hydra (SOTA)", "color": "#e67e22"}, 
    }

    plt.figure(figsize=(12, 7))
    
    for model_id, info in models.items():
        # Try both common naming patterns
        patterns = [
            f"{model_id}_{tok_name}_{data_name}_{tok_name}_log.txt", # Wiki style
            f"{model_id}_{tok_name}_{data_name}_log.txt"            # HP0 style
        ]
        
        steps, losses = [], []
        for p in patterns:
            log_path = os.path.join("logs", p)
            if os.path.exists(log_path):
                steps, losses = parse_log(log_path)
                break
        
        if steps:
            label = f"{info['label']} - min: {min(losses):.4f}"
            plt.plot(steps, losses, label=label, color=info['color'], linewidth=2.5, alpha=0.9)

    plt.title(f"The 10M 'Gauntlet': Architectural Ablation on {data_name.upper()}", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5)
    
    out_file = f"ablation_{data_name}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    plot_ablation("wiki103")
    plot_ablation("hp0")
