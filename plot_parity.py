import matplotlib.pyplot as plt
import re
import os

# Create assets directory to store the chart in the repo
out_dir = 'assets'
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(14, 10))

# Color coding by phase to make it readable
colors = {
    6: 'gray',
    7: 'blue',
    8: 'red',
    9: 'green'
}

for i in range(233, 253):
    log_file = f'logs/neon{i}_parity_log.txt'
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found")
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
                    
    phase = 6 if 233 <= i <= 237 else \
            7 if 238 <= i <= 242 else \
            8 if 243 <= i <= 247 else 9
            
    phase_name = {6: 'Baseline', 7: 'Conv+Int', 8: 'Intent Only', 9: 'Conv Only'}[phase]
    
    # Calculate offset within the phase (0 to 4)
    # Phase 6: 233-237, Phase 7: 238-242, Phase 8: 243-247, Phase 9: 248-252
    offset = (i - 233) % 5
    lookahead_pct = {0: '0%', 1: '25%', 2: '50%', 3: '75%', 4: '100%'}
    
    linestyles = {
        0: '-',      # Solid
        1: '--',     # Dashed
        2: '-.',     # Dash-dot
        3: ':',      # Dotted
        4: (0, (3, 1, 1, 1, 1, 1)) # Dash-dot-dot
    }
            
    plt.plot(steps, val_losses, label=f'neon{i} ({phase_name}, L={lookahead_pct[offset]})', 
             color=colors[phase], linestyle=linestyles[offset], alpha=0.8, linewidth=1.5)

plt.xlabel('Training Steps')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Steps (Phase 6-9, 5M Params, Wiki103)')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

out_file = os.path.join(out_dir, 'phase_6_to_9_conv_vs_intent_wiki103_val_loss.png')
plt.savefig(out_file, dpi=150)
print(f"Chart saved to {out_file}")
