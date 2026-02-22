import re
import matplotlib.pyplot as plt
import os

def parse_log(filepath):
    steps = []
    losses = []
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return steps, losses
        
    with open(filepath, 'r') as f:
        for line in f:
            # Match "Step XXX" followed by anything until "Val YYY"
            match = re.search(r'Step\s+(\d+).*Val\s+([\d.]+)', line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
    return steps, losses

# Path setup for Linux (WSL)
log_dir = "/home/luozichen/NeonBench/logs/"

# AdamW Line
a_steps1, a_losses1 = parse_log(log_dir + "neon213_growth_log.txt")
a_steps2, a_losses2 = parse_log(log_dir + "neon213_extended_log.txt")
adamw_steps = a_steps1 + a_steps2
adamw_losses = a_losses1 + a_losses2

# Muon Line
m_steps_g, m_losses_g = parse_log(log_dir + "neon213_muon_growth_log.txt")
m_steps_l, m_losses_l = parse_log(log_dir + "neon213_muon_long_log.txt")

# Concatenate Muon Growth and Long tail (offset by 35000)
muon_steps = m_steps_g + [s + 35000 for s in m_steps_l]
muon_losses = m_losses_g + m_losses_l

plt.figure(figsize=(10, 6))
plt.plot(adamw_steps, adamw_losses, label='AdamW (Original)', color='#f44336', alpha=0.5, linewidth=1.5)
plt.plot(muon_steps, muon_losses, label='Muon SOTA (Growth + Tail)', color='#2196f3', linewidth=2.5)

plt.title('Neon213 Training: AdamW vs. Muon Performance', fontsize=14, fontweight='bold')
plt.xlabel('Total Training Steps', fontsize=12)
plt.ylabel('Validation Loss (FineWeb-Edu)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(3.4, 6.0) # Focus on the meaningful range

plt.tight_layout()
plt.savefig('/home/luozichen/NeonBench/neon213_muon_vs_adamw.png', dpi=300)
print("Chart generated successfully: neon213_muon_vs_adamw.png")
