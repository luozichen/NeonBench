"""Verify parameter counts and forward passes for neon207-211."""
import torch
import sys, os
sys.path.append(os.getcwd())

configs = {
    'neon207': {'vocab_size': 8192, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1072, 'block_size': 256},
    'neon208': {'vocab_size': 8192, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1072, 'block_size': 256},
    'neon209': {'vocab_size': 8192, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1072, 'block_size': 256},
    'neon210': {'vocab_size': 8192, 'd_model': 272, 'n_layers': 4, 'n_head': 4, 'd_ff': 1072, 'block_size': 256},
    'neon211': {'vocab_size': 8192, 'd_model': 280, 'n_layers': 4, 'n_head': 4, 'd_ff': 1106, 'block_size': 256},
}

for name, config in configs.items():
    module = __import__(f"models.{name}", fromlist=[name.capitalize()])
    ModelClass = getattr(module, name.capitalize())
    model = ModelClass(config)
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = model.token_emb.weight.numel()
    non_emb = total - emb
    
    x = torch.randint(0, config['vocab_size'], (2, config['block_size']))
    logits, loss = model(x, x)
    
    ratio = config['d_ff'] / config['d_model']
    print(f"{name}: total={total:,} emb={emb:,} non_emb={non_emb:,} d={config['d_model']} ff={config['d_ff']} ratio={ratio:.2f}x loss={loss.item():.4f} OK")
