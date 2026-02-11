import re
import os
import glob

# 1. Get 10k Loss Data
loss_map = {}
log_files = glob.glob("logs/*_tok1_hp0_log.txt")
for log_file in log_files:
    model_name = os.path.basename(log_file).split("_")[0]
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Find the last "Step X: ... Val Loss Y" line
            last_valid_line = None
            for line in reversed(lines):
                if "Val Loss" in line and "Step" in line:
                    last_valid_line = line
                    break
            
            if last_valid_line:
                m = re.search(r"Val Loss ([\d.]+)", last_valid_line)
                if m:
                    loss_map[model_name] = m.group(1)
    except:
        pass

# 2. Read ARCH_LOG.md
with open("ARCH_LOG.md", "r") as f:
    lines = f.readlines()

new_lines = []
in_table = False
header_processed = False

for line in lines:
    stripped = line.strip()
    
    # Detect Header (Robust)
    if "| Model | Params |" in line and "Key Feature" in line:
        # Replace header
        new_lines.append("| Model | Params | ~10k Val Loss | Key Feature |\n")
        in_table = True
        continue
        
    # Detect Separator
    if "|---" in line and in_table and not header_processed:
        new_lines.append("|---|---|---|---|\n")
        header_processed = True
        continue
        
    # Detect Table Row
    if stripped.startswith("| neon") and in_table:
        parts = [p.strip() for p in stripped.split("|")]
        # parts[0] is empty, parts[1] is model, parts[2] is params, parts[3] is description/key feature, parts[4] is empty
        if len(parts) >= 4:
            model = parts[1]
            params = parts[2]
            desc = parts[3]
            
            val_loss = loss_map.get(model, "—")
            
            # Reconstruct
            new_line = f"| {model} | {params} | {val_loss} | {desc} |\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

# 3. Write Back
with open("ARCH_LOG.md", "w") as f:
    f.writelines(new_lines)

print("Updated ARCH_LOG.md with val loss column.")
