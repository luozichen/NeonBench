import os
import re

def parse_log(filepath):
    filename = os.path.basename(filepath)
    # Pattern: neonXXX_tokY_DATASET_log.txt
    match = re.search(r'(neon\d+)_(tok\d+)_(.+?)_log\.txt', filename)
    if not match:
        return None
    
    model = match.group(1)
    tok = match.group(2)
    dataset = match.group(3)
    
    val_loss = None
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # Look for final val loss
            for line in reversed(lines):
                if 'step' in line.lower() and 'val loss' in line.lower():
                    # Step 9500: Train Loss 0.9779, Val Loss 0.9174
                    loss_match = re.search(r'Val Loss ([\d\.]+)', line, re.IGNORECASE)
                    if loss_match:
                        val_loss = loss_match.group(1)
                        if val_loss == '0.0000': continue # Skip failed runs if any
                        break
    except Exception:
        pass
        
    return {
        "model": model,
        "tok": tok,
        "dataset": dataset,
        "loss": val_loss
    }

def main():
    logs_dir = 'logs'
    results = []
    if not os.path.exists(logs_dir):
        print(f"Log directory {logs_dir} not found.")
        return
        
    for f in os.listdir(logs_dir):
        if f.endswith('_log.txt'):
            res = parse_log(os.path.join(logs_dir, f))
            if res:
                results.append(res)
    
    # Sort by model, then dataset, then tok
    results.sort(key=lambda x: (x['model'], x['dataset'], x['tok']))
    
    # Also get non-embedding params from final_params_u8.txt if available
    params_map = {}
    if os.path.exists('final_params_u8.txt'):
        with open('final_params_u8.txt', 'r') as f:
            for line in f:
                if '|' in line and 'neon' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) > 1:
                        model_name = parts[0]
                        non_emb = parts[1]
                        params_map[model_name] = non_emb

    print(f"{'Model':<10} | {'Dataset':<10} | {'Tok':<6} | {'Params':<10} | {'Loss'}")
    print("-" * 60)
    for r in results:
        params = params_map.get(r['model'], 'N/A')
        print(f"{r['model']:<10} | {r['dataset']:<10} | {r['tok']:<6} | {params:<10} | {r['loss']}")

if __name__ == "__main__":
    main()
