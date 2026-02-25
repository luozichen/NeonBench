import matplotlib.pyplot as plt
import re
import os

out_dir = 'assets'
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(14, 10))

for i in range(257, 259):
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
                    
    name = 'neon257 (Wide Conv Kernel=9 Static)' if i == 257 else 'neon258 (Wide Conv Progressive Dropout)'
    plt.plot(steps, val_losses, label=name, linewidth=2)

# Also plot the best Phase 7 model (neon241) for comparison baseline
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
    plt.plot(b_steps, b_losses, label='neon241 (Phase 7 - Standard Conv Kernel=3)', color='black', linestyle='--', linewidth=2)

plt.xlabel('Training Steps')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Steps (Phase 11: Wide Conv & Progressive Dropout, 5M Params, Wiki103)')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)

# Stage lines for neon258
plt.axvline(x=3000, color='r', linestyle=':', label='Stage 2 Start (Progressive Fade-in)')
plt.axvline(x=7000, color='g', linestyle=':', label='Stage 3 Start (Full Conv)')

plt.tight_layout()

out_file = os.path.join(out_dir, 'phase11_wide_progressive_val_loss.png')
plt.savefig(out_file, dpi=150)
print(f"Chart saved to {out_file}")
