"""
Audit script v2: Extract ground truth from model source code and training logs.
Outputs structured data for README correction.
"""
import os, re, glob

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
MODELS_DIR = os.path.join(BASE, 'models')
LOGS_DIR = os.path.join(BASE, 'logs')

def get_model_files():
    files = glob.glob(os.path.join(MODELS_DIR, 'neon*.py'))
    def key(f):
        m = re.search(r'neon(\d+)', os.path.basename(f))
        return int(m.group(1)) if m else 0
    return sorted(files, key=key)

def extract_docstring(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    doc = re.search(r'^"""(.*?)"""', content, re.DOTALL | re.MULTILINE)
    if not doc:
        doc = re.search(r"^'''(.*?)'''", content, re.DOTALL | re.MULTILINE)
    if doc:
        return doc.group(1).strip()
    lines = content.split('\n')
    comments = []
    for line in lines:
        s = line.strip()
        if s.startswith('#'):
            comments.append(s)
        elif comments:
            break
    return '\n'.join(comments[:10]) if comments else 'NO DOCSTRING'

def extract_config(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    config = {}
    m = re.search(r'(?:config|CONFIG)\s*=\s*\{([^}]+)\}', content)
    if m:
        raw = m.group(1)
        for key in ['d_model', 'd_ff', 'n_head', 'n_layers', 'block_size', 'vocab_size']:
            km = re.search(rf"['\"]?{key}['\"]?\s*:\s*(\d+)", raw)
            if km:
                config[key] = int(km.group(1))
    return config

def get_log_scores():
    """Extract best (minimum) val loss from each log file."""
    scores = {}
    for lf in sorted(glob.glob(os.path.join(LOGS_DIR, '*.txt'))):
        basename = os.path.basename(lf)
        # neonXXX_tokY_dataset_log.txt or neonXXX_tokY_dataset_tokZ_log.txt
        m = re.match(r'(neon\d+)_(.+?)_log\.txt', basename)
        if not m:
            continue
        model_id = m.group(1)
        variant = m.group(2)
        
        try:
            with open(lf, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            continue
        
        # Format: "Step XXXX: Train Loss X.XXXX, Val Loss X.XXXX"
        val_losses = re.findall(r'Val Loss\s+([\d.]+)', content)
        if val_losses:
            # Take the minimum as the best
            best = min(float(v) for v in val_losses)
            # Take the last as the final
            final = float(val_losses[-1])
            key = f"{model_id}|{variant}"
            scores[key] = {'best': best, 'final': final, 'file': basename}
    
    return scores

def determine_dataset_tok(variant):
    """Parse variant string like 'tok1_hp0' or 'tok4_wiki103_tok4' into (dataset, tokenizer)."""
    if 'wiki103' in variant:
        dataset = 'wiki103'
    elif 'hp0' in variant:
        dataset = 'hp0'
    else:
        dataset = variant
    
    tok_m = re.search(r'tok(\d+)', variant)
    tok = f'tok{tok_m.group(1)}' if tok_m else '?'
    return dataset, tok

def main():
    model_files = get_model_files()
    scores = get_log_scores()
    
    # Group scores by model
    model_scores = {}
    for key, val in scores.items():
        model_id, variant = key.split('|')
        if model_id not in model_scores:
            model_scores[model_id] = {}
        dataset, tok = determine_dataset_tok(variant)
        run_key = f"{dataset}/{tok}"
        model_scores[model_id][run_key] = val
    
    out = []
    out.append("=" * 120)
    out.append("NEONBENCH FULL AUDIT REPORT")
    out.append(f"Total model files: {len(model_files)}")
    out.append(f"Total log entries: {len(scores)}")
    out.append("=" * 120)
    
    for mf in model_files:
        basename = os.path.basename(mf)
        model_id = re.search(r'(neon\d+)', basename).group(1)
        doc = extract_docstring(mf)
        config = extract_config(mf)
        
        out.append(f"\n{'─' * 80}")
        out.append(f"  {model_id}")
        out.append(f"{'─' * 80}")
        
        # Doc (first 3 lines)
        for line in doc.split('\n')[:4]:
            out.append(f"  DOC: {line.strip()}")
        
        # Config
        if config:
            cfg_str = ', '.join(f"{k}={v}" for k, v in config.items())
            out.append(f"  CFG: {cfg_str}")
        
        # Scores
        ms = model_scores.get(model_id, {})
        if ms:
            for run, sv in sorted(ms.items()):
                out.append(f"  SCORE [{run}]: best={sv['best']:.4f}, final={sv['final']:.4f} ({sv['file']})")
        else:
            out.append(f"  SCORE: NO LOGS FOUND")
    
    # Models in README not found as files (check gaps)
    model_nums = sorted(int(re.search(r'(\d+)', os.path.basename(f)).group(1)) for f in model_files)
    all_nums = set(range(1, max(model_nums) + 1))
    present = set(model_nums)
    missing = sorted(all_nums - present)
    if missing:
        out.append(f"\n\nGAP ANALYSIS: Model numbers with NO .py file: {missing}")
    
    report = '\n'.join(out)
    
    # Write to file
    outfile = os.path.join(BASE, 'scripts', 'audit_report.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n\nReport saved to: {outfile}")

if __name__ == '__main__':
    main()
