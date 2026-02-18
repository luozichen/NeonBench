"""
Generate comprehensive param counts for ALL models (neon001-187).
Outputs model_id, total_params, non_emb_params, d_model, d_ff, n_head, n_layers.
"""
import sys, os, importlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from train import get_config

def count_non_emb(model, cfg):
    """Count params excluding embedding and lm_head."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = 0
    # Commonly: model.token_embedding, model.lm_head
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'token_embedding' in name or 'tok_emb' in name or 'wte' in name:
            emb += p.numel()
        elif 'lm_head' in name or 'head' in name and 'head' == name.split('.')[-1]:
            # Only count if it's the final linear head
            if name.endswith('.weight') or name.endswith('.bias'):
                if 'lm_head' in name:
                    emb += p.numel()
    return total, emb, total - emb

def main():
    results = []
    for i in range(1, 188):
        model_name = f"neon{i:03d}"
        try:
            cfg = get_config(model_name)
            module = importlib.import_module(f"models.{model_name}")
            # Try different class name patterns
            cls_name = None
            for candidate in [model_name.capitalize(), f"Neon{i:03d}", f"Neon{i}"]:
                if hasattr(module, candidate):
                    cls_name = candidate
                    break
            if cls_name is None:
                # Find the first class that looks like a model
                for attr in dir(module):
                    obj = getattr(module, attr)
                    if isinstance(obj, type) and hasattr(obj, 'forward'):
                        cls_name = attr
                        break
            
            if cls_name is None:
                print(f"{model_name} | ERROR: No model class found")
                continue
            
            ModelClass = getattr(module, cls_name)
            model = ModelClass(cfg)
            total, emb, non_emb = count_non_emb(model, cfg)
            
            d_model = cfg.get('d_model', '?')
            d_ff = cfg.get('d_ff', '?')
            n_head = cfg.get('n_head', '?')
            n_layers = cfg.get('n_layers', '?')
            
            non_emb_m = non_emb / 1_000_000
            print(f"{model_name} | {non_emb_m:.2f}M | d={d_model} ff={d_ff} h={n_head} L={n_layers} | total={total:,} emb={emb:,} non_emb={non_emb:,}")
            results.append((model_name, non_emb_m, d_model, d_ff, n_head, n_layers))
            
            del model
            
        except Exception as e:
            print(f"{model_name} | ERROR: {str(e)[:100]}")
    
    # Write summary
    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"{r[0]}: {r[1]:.2f}M (d={r[2]}, ff={r[3]}, h={r[4]}, L={r[5]})")

if __name__ == '__main__':
    main()
