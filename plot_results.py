import matplotlib.pyplot as plt
import re
import os

def parse_log(path):
    steps, val_losses = [], []
    if not os.path.exists(path):
        return None, None
    with open(path, 'r') as f:
        for line in f:
            match = re.search(r'Step (\d+):.*Val ([\d\.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))
    return steps, val_losses

logs = {
    'Neon 300 (Baseline)': 'logs/neon300_training_log.txt',
    'Neon 301 (Ortho + Detach)': 'logs/neon301_training_log.txt',
    'Neon 303 (Global GS + Live)': 'logs/neon303_training_log.txt',
    'Neon 304 (Local Ortho)': 'logs/neon304_training_log.txt'
}

plt.figure(figsize=(10, 6))
plt.style.use('dark_background')

colors = ['#888888', '#ff7f0e', '#2ca02c', '#1f77b4']
for (label, path), color in zip(logs.items(), colors):
    s, v = parse_log(path)
    if s:
        plt.plot(s, v, label=label, linewidth=1.5, color=color, alpha=0.9)

plt.title('Validation Loss Comparison', fontsize=14, pad=15)
plt.xlabel('Step')
plt.ylabel('Val Loss')
plt.grid(True, alpha=0.1)
plt.legend()
plt.ylim(3.2, 3.8) # Focus area
plt.tight_layout()

plt.savefig('results_chart_300.png', dpi=120)
print("Chart generated: results_chart_300.png")
