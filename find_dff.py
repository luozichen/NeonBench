import sys
sys.path.append('.')
from check_params import count_non_embed

def find_d_ff(model_name, start_d_ff):
    import importlib.util
    spec = importlib.util.spec_from_file_location(model_name, f"models/{model_name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[model_name] = mod
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, model_name.capitalize())
    
    target = 5004528
    
    d_ff = start_d_ff
    while True:
        config = {'vocab_size': 32000, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': d_ff, 'block_size': 1024, 'device': 'cpu'}
        m = ModelClass(config)
        p = sum(param.numel() for param in m.parameters() if param.requires_grad) - m.token_emb.weight.numel()
        
        diff = target - p
        if diff == 0:
            print(f"{model_name}: EXACT MATCH. d_ff={d_ff}")
            break
        elif abs(diff) > 2000:
            d_ff += int(diff / 2200) # conservative jump
        elif diff > 0:
            d_ff += 1
        elif diff < 0:
            # check if previous was closer
            config_prev = {'vocab_size': 32000, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': d_ff - 1, 'block_size': 1024, 'device': 'cpu'}
            m_prev = ModelClass(config_prev)
            p_prev = sum(param.numel() for param in m_prev.parameters() if param.requires_grad) - m_prev.token_emb.weight.numel()
            
            if abs(target - p_prev) < abs(target - p):
                print(f"{model_name}: BEST MATCH. d_ff={d_ff-1} (Diff {p_prev - target})")
                break
            else:
                print(f"{model_name}: BEST MATCH. d_ff={d_ff} (Diff {p - target})")
                break

find_d_ff("neon243", 1170)
find_d_ff("neon248", 1072)
