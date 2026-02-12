import os
import re
import glob

def summarize():
    log_files = glob.glob("logs/*.txt")
    results = []

    print(f"Found {len(log_files)} logs.")

    for log_file in sorted(log_files):
        model_name = os.path.basename(log_file).replace("_log.txt", "")
        
        last_val_loss = None
        params = "Unknown"
        vocab = "Unknown"
        block_size = "Unknown"
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # Parse Config
            for line in lines:
                if "Config:" in line:
                    # simplistic parse
                    if "'vocab_size': 1024" in line: vocab = "1k"
                    elif "'vocab_size': 4096" in line: vocab = "4k"
                    
                    if "'block_size': 64" in line: block_size = "64"
                    elif "'block_size': 128" in line: block_size = "128"
                    elif "'block_size': 256" in line: block_size = "256"
                
                if "Val Loss" in line:
                    parts = line.split("Val Loss")
                    if len(parts) > 1:
                        try:
                            val = float(parts[1].strip())
                            last_val_loss = val
                        except:
                            pass
        
        if last_val_loss is not None:
             results.append({
                 "model": model_name,
                 "vocab": vocab,
                 "block": block_size,
                 "loss": last_val_loss
             })

    # Print Table
    print(f"{'Model':<30} | {'Vocab':<6} | {'Ctx':<4} | {'Val Loss':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<30} | {r['vocab']:<6} | {r['block']:<4} | {r['loss']:<10.4f}")

if __name__ == "__main__":
    summarize()
