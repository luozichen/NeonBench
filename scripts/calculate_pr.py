import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json

# Fix import error if run from scripts/
sys.path.append(os.getcwd())
from models.neon213 import Neon213

# CONFIGURATION (Neon214 Audit)
paths = {
    "model": "checkpoints/neon213_muon_long/neon213_muon_long_final.pth",
    "data": "data/fineweb/fineweb_tok6.bin",
    "tokenizer": "tokenizers/fineweb_tok6.json"
}

BATCH_SIZE = 4
SEQ_LEN = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_participation_ratio(x):
    """
    Calculates Participation Ratio (PR) of a matrix X [N, D].
    PR = (sum(eig))^2 / sum(eig^2)
    """
    # x: [N, D]
    if x.shape[0] < x.shape[1]:
        return 0.0 # Not enough samples
        
    # Standardize/Center (though some calculate PR on raw activations)
    x = x - x.mean(dim=0, keepdim=True)
    
    # SVD is more stable than eigenvalue decomposition of Cov
    try:
        _, s, _ = torch.svd(x)
        eigs = s ** 2
        pr = (eigs.sum() ** 2) / (eigs ** 2).sum()
        return pr.item()
    except:
        return 0.0

def probe_model():
    # 1. Load Model (Config for neon214)
    config = {
        'd_model': 384, 'n_head': 6, 'n_layers': 8, 'd_ff': 1536, 
        'vocab_size': 16384, 'block_size': 256, 'conv_k': 21, 'mlp_k': 21
    }
    
    print(f"Loading model: {paths['model']}")
    model = Neon213(config).to(DEVICE)
    checkpoint = torch.load(paths["model"], map_location=DEVICE)
    # Check if checkpoint is a dict with 'model' key or just weights
    sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(sd)
    model.eval()

    # 2. Prepare Data
    print(f"Loading data: {paths['data']}")
    data = np.memmap(paths["data"], dtype=np.uint16, mode='r')
    
    # Random sample for calibration
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix]).to(DEVICE)

    # 3. Setup hooks to capture Q, K, V, I AND Residual Hidden States
    pr_data = {l: {"q": [], "k": [], "v": [], "i": [], "h": []} for l in range(config['n_layers'])}
    
    def get_conv_hook(layer_idx, name):
        def hook(module, inp, out):
            B, D, T = out.shape
            x = out.transpose(1, 2).view(B, T, config['n_head'], config['d_model'] // config['n_head'])
            pr_data[layer_idx][name] = x.detach().cpu()
        return hook

    def get_residual_hook(layer_idx):
        def hook(module, inp, out):
            # out is [B, T, D]
            pr_data[layer_idx]["h"] = out.detach().cpu()
        return hook

    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.conv_q.register_forward_hook(get_conv_hook(i, "q")))
        hooks.append(block.attn.conv_k.register_forward_hook(get_conv_hook(i, "k")))
        hooks.append(block.attn.conv_v.register_forward_hook(get_conv_hook(i, "v")))
        hooks.append(block.attn.conv_i.register_forward_hook(get_conv_hook(i, "i")))
        hooks.append(block.register_forward_hook(get_residual_hook(i)))

    # 4. Forward
    print("Running Calibration Forward Pass (Capturing Q, K, V, I, Hidden)...")
    with torch.no_grad():
        model(x)

    # 5. Calculate PR and Activity Map
    print("\n" + "="*80)
    print(f"{'LAYER':<8} | {'H':<3} | {'Q-PR':<6} | {'K-PR':<6} | {'V-PR':<6} | {'I-PR':<6} | {'RES-PR':<7} | {'DIM-ACTIVITY (32-dim chunks)'}")
    print("-" * 80)
    
    head_dim = config['d_model'] // config['n_head']
    total_dim = config['d_model']
    all_q, all_k, all_v, all_i = [], [], [], []

    for l in range(config['n_layers']):
        q, k, v, i_ = pr_data[l]["q"], pr_data[l]["k"], pr_data[l]["v"], pr_data[l]["i"]
        res_h = pr_data[l]["h"] # [B, T, D]
        
        # Calculate Res PR
        res_flat = res_h.reshape(-1, total_dim)
        res_pr = get_participation_ratio(res_flat)
        
        # Calculate Activity Map (Variance per coordinate)
        variances = res_flat.var(dim=0)
        # Visualization: 64-dim chunks for Neon213
        chunk_size = 64
        chunks = variances.view(-1, chunk_size).mean(dim=1)
        norm_chunks = chunks / chunks.max() if chunks.max() > 0 else chunks
        activity_str = "".join(["█" if v > 0.8 else "▓" if v > 0.6 else "▒" if v > 0.4 else "░" if v > 0.2 else "." for v in norm_chunks])
        
        for h in range(config['n_head']):
            q_pr = get_participation_ratio(q[:, :, h, :].reshape(-1, head_dim))
            k_pr = get_participation_ratio(k[:, :, h, :].reshape(-1, head_dim))
            v_pr = get_participation_ratio(v[:, :, h, :].reshape(-1, head_dim))
            i_pr = get_participation_ratio(i_[:, :, h, :].reshape(-1, head_dim))
            
            # Print row for first head, then sub-rows
            if h == 0:
                print(f"L{l:<7} | {h:<3} | {q_pr:>6.1f} | {k_pr:>6.1f} | {v_pr:>6.1f} | {i_pr:>6.1f} | {res_pr:>7.1f} | {activity_str}")
            else:
                print(f"{'':<8} | {h:<3} | {q_pr:>6.1f} | {k_pr:>6.1f} | {v_pr:>6.1f} | {i_pr:>6.1f} | {'':<7} |")
            
            all_q.append(q_pr); all_k.append(k_pr); all_v.append(v_pr); all_i.append(i_pr)
        print("-" * 80)

    avg_q, avg_k, avg_v, avg_i = np.mean(all_q), np.mean(all_k), np.mean(all_v), np.mean(all_i)
    print(f"{'AVERAGE':<13} | {avg_q:>6.1f} | {avg_k:>6.1f} | {avg_v:>6.1f} | {avg_i:>6.1f} | {'':<7} |")
    print("="*80)

    # Cleanup
    for h in hooks: h.remove()

if __name__ == "__main__":
    probe_model()
