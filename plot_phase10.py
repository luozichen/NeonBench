import matplotlib.pyplot as plt
import re
import os

out_dir = 'assets'
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(14, 10))

for i in range(253, 257):
    log_file = f'logs/neon{i}_parity_log.txt'
    if not os.path.exists(log_file):
        continue
        
    steps = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if 'Val Loss' in line and 'FINAL' not in line:
                match = re.search(r'Step (\d+):.*Val Loss ([\d\.]+)', line)
                if match:
                    steps.append(int(match.group(1)))
                    val_losses.append(float(match.group(2)))
                    
    layer = i - 253
    plt.plot(steps, val_losses, label=f'neon{i} (Layer {layer} Lookahead)', linewidth=2)

# Also plot the best Phase 7 model (neon241 - 75% lookahead across all layers) for comparison baseline
baseline_log = 'logs/neon241_parity_log.txt'
if os.path.exists(baseline_log):
    b_steps, b_losses = [], []
    with open(baseline_log, 'r') as f:
        for line in f:
            if 'Val Loss' in line and 'FINAL' not in line:
                match = re.search(r'Step (\d+):.*Val Loss ([\d\.]+)', line)
                if match:
                    b_steps.append(int(match.group(1)))
                    b_losses.append(float(match.group(2)))
    plt.plot(b_steps, b_losses, label='neon241 (Phase 7 Best - 75% All Layers)', color='black', linestyle='--', linewidth=2)

plt.xlabel('Training Steps')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Steps (Phase 10: Layer-Specific Lookahead, 5M Params, Wiki103)')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

out_file = os.path.join(out_dir, 'phase10_layer_lookahead_val_loss.png')
plt.savefig(out_file, dpi=150)
print(f"Chart saved to {out_file}")
