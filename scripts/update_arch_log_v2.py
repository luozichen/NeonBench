import re
import os
import glob

# 1. Get Log Data (Robust)
loss_map = {}
log_files = glob.glob("logs/*_tok1_hp0_log.txt")
for log_file in log_files:
    model_name = os.path.basename(log_file).split("_")[0]
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            last_valid_line = None
            # Scan backwards for "Step X: ... Val Loss Y"
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

# 2. Process ARCH_LOG.md
with open("ARCH_LOG.md", "r") as f:
    lines = f.readlines()

new_lines = []
in_table = False
header_processed = False
col_indices = {} # map column name to index
has_loss_col = False
insert_pos = 2 # Default: Insert loss after Params (index 2 in parts list if split by |)

for line in lines:
    stripped = line.strip()
    
    # Header Detection
    if "| Model |" in line and not header_processed:
        parts = [p.strip() for p in stripped.split("|")]
        # Remove empty strings from split
        cols = [p for p in parts if p]
        
        # Build index map
        col_indices = {name: i for i, name in enumerate(cols)}
        
        # Check if loss column exists
        has_loss_col = any("Val Loss" in c for c in cols)
        
        if has_loss_col:
            new_lines.append(line)
        else:
            # Insert column
            # Header: | Model | Params | ~10k Val Loss | Key Feature |
            # Original might be: | Model | Params | Key Feature |
            # We want to insert at index 2 (After Params)
            
            # Find index of "Params"
            p_idx = -1
            for i, c in enumerate(cols):
                if "Params" in c:
                    p_idx = i
                    break
            
            if p_idx != -1:
                insert_pos = p_idx + 1
            
            # Construct new header
            new_cols = list(cols)
            new_cols.insert(insert_pos, "~10k Val Loss")
            new_header = "| " + " | ".join(new_cols) + " |\n"
            new_lines.append(new_header)
            
        in_table = True
        header_processed = True
        continue

    # Separator Detection
    if "|---" in line and in_table:
        if has_loss_col:
            new_lines.append(line)
        else:
            # Add separator column
            parts = [p.strip() for p in stripped.split("|")]
            valid_parts = [p for p in parts if p] # e.g. ['---', '---', '---']
            if len(valid_parts) == len(col_indices):
                 valid_parts.insert(insert_pos, "---")
                 new_sep = "| " + " | ".join(valid_parts) + " |\n"
                 new_lines.append(new_sep)
            else:
                # Fallback
                 new_lines.append("|---|---|---|---|\n")
        continue

    # Row Processing
    if stripped.startswith("| neon") and in_table:
        parts = [p.strip() for p in stripped.split("|")]
        # Filter empty strings (usually first and last if formatted correctly)
        # But split("|") on "| a | b |" gives ['', 'a', 'b', '']
        # We want to work with the content parts
        
        content_parts = parts[1:-1] # Assuming standard markdown table format
        
        if len(content_parts) < 2:
            new_lines.append(line)
            continue
            
        model = content_parts[0] # Model
        
        # Lookup Loss
        val_loss = loss_map.get(model, "—")
        
        if has_loss_col:
            # Update existing column
            # Find index of loss column
            # We need to know which index in content_parts corresponds to loss
            # col_indices maps 'Model' -> 0, 'Params' -> 1, 'Loss' -> 2
            
            # Identify the loss column index from header analysis
            loss_col_idx = -1
            for name, idx in col_indices.items():
                if "Val Loss" in name:
                    loss_col_idx = idx
                    break
            
            if loss_col_idx != -1 and loss_col_idx < len(content_parts):
                if val_loss != "—":
                    content_parts[loss_col_idx] = val_loss
            
            # Reconstruct
            new_line = "| " + " | ".join(content_parts) + " |\n"
            new_lines.append(new_line)
            
        else:
            # Insert new column
            content_parts.insert(insert_pos, val_loss)
            new_line = "| " + " | ".join(content_parts) + " |\n"
            new_lines.append(new_line)
            
    else:
        new_lines.append(line)

# 3. Write Back
with open("ARCH_LOG.md", "w") as f:
    f.writelines(new_lines)

print("Safely updated ARCH_LOG.md with val loss data.")
