import os
import glob
import re

def parse_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    config = {}
    val_loss = None
    tokenizer = "Unknown"
    
    # Parse Config and Tokenizer from first line
    for line in lines:
        if "tokenizer" in line and "Training" in line:
            # Training neon001 on ... with tokenizer tok4
            parts = line.split("tokenizer")
            if len(parts) > 1:
                tokenizer = parts[1].strip()
        
        if "Config:" in line:
            # Config: {'vocab_size': ...}
            try:
                config_str = line.split("Config:")[1].strip().replace("'", '"')
                # simplistic parse
                if "'vocab_size': 1024" in line: config['vocab'] = "1k"
                elif "'vocab_size': 4096" in line: config['vocab'] = "4k"
                
                if "'d_ff':" in line:
                     # extract d_ff
                     match = re.search(r"'d_ff': (\d+)", line)
                     if match: config['d_ff'] = int(match.group(1))
            except:
                pass
        
        if "Val Loss" in line:
            parts = line.split("Val Loss")
            try:
                val_loss = float(parts[1].strip())
            except:
                pass
                
    return tokenizer, config, val_loss

def main():
    logs = glob.glob("logs/*.txt")
    results = []
    
    for log in logs:
        model = os.path.basename(log).split("_")[0]
        tok, conf, loss = parse_log(log)
        if loss is not None:
            results.append((model, tok, conf.get('d_ff', '?'), loss))
            
    # Sort by Model ID
    results.sort(key=lambda x: x[0])
    
    print(f"{'Model':<10} | {'Tok':<5} | {'d_ff':<6} | {'Loss':<10}")
    print("-" * 40)
    for model, tok, d_ff, loss in results:
        print(f"{model:<10} | {tok:<5} | {d_ff:<6} | {loss:.4f}")

if __name__ == "__main__":
    main()
