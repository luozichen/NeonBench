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

# CONFIGURATION (Neon213 is FineWeb-Edu tok6)
paths = {
    "model": "checkpoints/neon213_turbo/neon213_k21_final.pth",
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
    # Singular values s_i relate to eigenvalues of XX^T as lambda_i = s_i^2
    try:
        _, s, _ = torch.svd(x)
        eigs = s ** 2
        pr = (eigs.sum() ** 2) / (eigs ** 2).sum()
        return pr.item()
    except:
        return 0.0

def probe_model():
    # 1. Load Model (Config must match the neon213 checkpoint)
    config = {
        'd_model': 384, 'n_head': 6, 'n_layers': 8, 'd_ff': 1536, 
        'vocab_size': 16384, 'block_size': 256, 'conv_k': 21, 'mlp_k': 21
    }
    
    print(f"Loading model: {paths['model']}")
    model = Neon213(config).to(DEVICE)
    checkpoint = torch.load(paths["model"], map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # 2. Prepare Data
    print(f"Loading data: {paths['data']}")
    data = np.memmap(paths["data"], dtype=np.uint16, mode='r')
    
    # Random sample for calibration
    ix = torch.randint(len(data) - SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+SEQ_LEN]).astype(np.int64)) for i in ix]).to(DEVICE)

    # 3. Setup hooks to capture Q, K, V, I
    pr_data = {l: {"q": [], "k": [], "v": [], "i": []} for l in range(config['n_layers'])}
    
    # We will use a more direct method: registering hooks on the conv layers
    # to capture and reshape them exactly as they are used in attention.
    def get_conv_hook(layer_idx, name):
        def hook(module, inp, out):
            # out is [B, D, T] -> transpose to [B, T, D] -> view [B, T, h, d]
            B, D, T = out.shape
            x = out.transpose(1, 2).view(B, T, config['n_head'], config['d_model'] // config['n_head'])
            pr_data[layer_idx][name] = x.detach().cpu()
        return hook

    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.conv_q.register_forward_hook(get_conv_hook(i, "q")))
        hooks.append(block.attn.conv_k.register_forward_hook(get_conv_hook(i, "k")))
        hooks.append(block.attn.conv_v.register_forward_hook(get_conv_hook(i, "v")))
        hooks.append(block.attn.conv_i.register_forward_hook(get_conv_hook(i, "i")))

    # 4. Forward
    print("Running Calibration Forward Pass (Capturing Q, K, V, I)...")
    with torch.no_grad():
        model(x)

    # 5. Calculate PR
    print("\n" + "="*70)
    print(f"{'LAYER':<8} | {'H':<3} | {'Q-PR':<7} | {'K-PR':<7} | {'V-PR':<7} | {'I-PR':<7} | {'UTIL %':<8}")
    print("-" * 70)
    
    head_dim = config['d_model'] // config['n_head']
    all_q, all_k, all_v, all_i = [], [], [], []

    for l in range(config['n_layers']):
        q, k, v, i_ = pr_data[l]["q"], pr_data[l]["k"], pr_data[l]["v"], pr_data[l]["i"]
        
        for h in range(config['n_head']):
            q_pr = get_participation_ratio(q[:, :, h, :].reshape(-1, head_dim))
            k_pr = get_participation_ratio(k[:, :, h, :].reshape(-1, head_dim))
            v_pr = get_participation_ratio(v[:, :, h, :].reshape(-1, head_dim))
            i_pr = get_participation_ratio(i_[:, :, h, :].reshape(-1, head_dim))
            
            util = (q_pr / head_dim) * 100
            print(f"L{l:<7} | {h:<3} | {q_pr:>7.1f} | {k_pr:>7.1f} | {v_pr:>7.1f} | {i_pr:>7.1f} | {util:>7.1f}%")
            
            all_q.append(q_pr); all_k.append(k_pr); all_v.append(v_pr); all_i.append(i_pr)
        print("-" * 70)

    avg_q, avg_k, avg_v, avg_i = np.mean(all_q), np.mean(all_k), np.mean(all_v), np.mean(all_i)
    avg_util = (avg_q / head_dim) * 100
    print(f"{'AVERAGE':<13} | {avg_q:>7.1f} | {avg_k:>7.1f} | {avg_v:>7.1f} | {avg_i:>7.1f} | {avg_util:>7.1f}%")
    print("="*70)

    # Cleanup
    for h in hooks: h.remove()

if __name__ == "__main__":
    probe_model()
