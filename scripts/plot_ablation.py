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

def plot_ablation():
    # Model Mapping
    # neon093: No QKVI, No Hydra (8-Layer Standard)
    # neon061: Yes QKVI, No Hydra (Standard Attention w/ Intent)
    # neon094: No QKVI, Yes Hydra (Hydra MLP w/ Standard Attention)
    # neon092: Yes QKVI, Yes Hydra (Full Dual-Scale Hydra)
    
    models = {
        "neon093": {"label": "Baseline (No QKVI, No Hydra)", "color": "#7f8c8d"}, # Gray
        "neon061": {"label": "Intent Only (Yes QKVI, No Hydra)", "color": "#3498db"}, # Blue
        "neon094": {"label": "Hydra Only (No QKVI, Yes Hydra)", "color": "#9b59b6"}, # Purple
        "neon092": {"label": "Full Hydra (Yes QKVI, Yes Hydra)", "color": "#e67e22"}, # Orange
    }

    plt.figure(figsize=(12, 7))
    
    for model_id, info in models.items():
        log_file = f"{model_id}_tok4_wiki103_tok4_log.txt"
        log_path = os.path.join("logs", log_file)
        steps, losses = parse_log(log_path)
        
        if steps:
            label = f"{info['label']} - min: {min(losses):.4f}"
            plt.plot(steps, losses, label=label, color=info['color'], linewidth=2.5, alpha=0.9)

    plt.title("The 10M 'Gauntlet': Architectural Ablation on Wiki103", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Zoom in on the final 2000 steps for clarity
    # plt.xlim(5000, 10000)
    # plt.ylim(3.0, 3.3)
    
    plt.tight_layout()
    plt.savefig("hydra_ablation_wiki103.png", dpi=300)
    print("Saved hydra_ablation_wiki103.png")

if __name__ == "__main__":
    plot_ablation()
