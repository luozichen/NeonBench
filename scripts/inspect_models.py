import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.getcwd())

from train import get_config

def inspect_model(model_name):
    # 1. Get Config
    try:
        config = get_config(model_name)
    except Exception:
        return None

    # 2. Load Model Class
    try:
        module = __import__(f"models.{model_name}", fromlist=[model_name.capitalize()])
        ModelClass = getattr(module, model_name.capitalize())
        model = ModelClass(config)
    except Exception as e:
        return f"Error: {e}"

    # 3. Count Parameters (Exclude Embeddings)
    # Heuristic: exclude nn.Embedding and the final head if it's tied
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Identify Embeddings and Head
    emb_params = 0
    head_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            emb_params += sum(p.numel() for p in module.parameters())
    
    # Head params - usually called 'head'
    head = getattr(model, 'head', None)
    if head and isinstance(head, nn.Linear):
         head_params = sum(p.numel() for p in head.parameters())
         
    # If tied (common in Neon), head.weight is same as token_emb.weight
    # Check if they share the same data pointer
    is_tied = False
    token_emb = getattr(model, 'token_emb', None)
    if token_emb and head:
        if token_emb.weight.data_ptr() == head.weight.data_ptr():
            is_tied = True

    # Param count excluding EMBEDDING and HEAD (the Transformers logic)
    # Often "non-embedding parameters" means excluding the token embedding but keeping everything else.
    # User said "param (excluding embedding)". I will exclude the token_emb.
    non_emb_params = total_params - emb_params
    
    # 4. Inspect Features (Heuristic)
    block = model.blocks[0] if hasattr(model, 'blocks') and len(model.blocks) > 0 else None
    
    norm_type = "Unknown"
    pos_emb = "None"
    act_type = "Unknown"
    bias = False
    
    if block:
        norm = getattr(block, 'ln1', getattr(block, 'norm1', None))
        if norm:
            if isinstance(norm, nn.LayerNorm): norm_type = "LN"
            elif "RMSNorm" in str(type(norm)): norm_type = "RMS"
        
        mlp = getattr(block, 'mlp', None)
        if mlp:
            if hasattr(mlp, 'gelu'): act_type = "GELU"
            elif "SwiGLU" in str(type(mlp)): act_type = "SwiGLU"
            elif hasattr(mlp, 'act'):
                act = mlp.act
                if isinstance(act, nn.SiLU): act_type = "SiLU"
                elif isinstance(act, nn.GELU): act_type = "GELU"
        
        if hasattr(block, 'attn'):
            attn = block.attn
            if hasattr(attn, 'c_attn') and getattr(attn.c_attn, 'bias', None) is not None: bias = True

    if hasattr(model, 'freqs_cos'): pos_emb = "RoPE"
    elif hasattr(model, 'pos_emb'): pos_emb = "Learned"

    return {
        "non_emb_params": non_emb_params,
        "total_params": total_params,
        "norm": norm_type,
        "pos": pos_emb,
        "act": act_type,
        "bias": bias,
        "d_ff": config.get('d_ff', '?'),
        "is_tied": is_tied
    }

def main():
    print(f"{'Model':<10} | {'Non-Emb Params':<15} | {'Total':<10} | {'d_ff':<6} | {'Norm':<5} | {'Pos':<7} | {'Act':<7} | {'Bias'}")
    print("-" * 90)
    
    for i in range(1, 80):
        name = f"neon{i:03d}"
        if not os.path.exists(f"models/{name}.py"):
             continue
        info = inspect_model(name)
        if isinstance(info, str):
            print(f"{name:<10} | {info}")
            continue
        if info:
            print(f"{name:<10} | {info['non_emb_params']:<15,} | {info['total_params']:<10,} | {info['d_ff']:<6} | {info['norm']:<5} | {info['pos']:<7} | {info['act']:<7} | {info['bias']}")

if __name__ == "__main__":
    main()
