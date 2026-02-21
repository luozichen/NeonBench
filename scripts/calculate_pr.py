import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from networks.neon213 import Neon213

# CONFIGURATION (Adjust these on server if needed)
paths = {
    "model": "checkpoints/neon213_turbo/neon213_k21_final.pth",
    "data": "data/wiki103/wiki103_tok5.bin",
    "tokenizer": "tokenizers/wiki103_tok5.json"
}

BATCH_SIZE = 4
SEQ_LEN = 512
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
    # 1. Load Model
    config = {
        'd_model': 384, 'n_head': 6, 'n_layers': 8, 'd_ff': 1536, 
        'vocab_size': 16384, 'block_size': 1024, 'conv_k': 21
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

    # 3. Setup Hooks to capture Q and K
    pr_data = {l: {"q": [], "k": []} for l in range(config['n_layers'])}
    
    def get_hook(layer_idx, name):
        def hook(module, inp, out):
            # We want to capture the vectors that are actually used in Softmax
            # In GrowableConvAttention, these are q, k after norm and rotary
            # This hook should be placed after RoPE
            # But theRoPE is applied in the forward, so we hook 'q_norm'/'k_norm' 
            # and simulate RoPE or just measure on normalized Q/K
            pr_data[layer_idx][name] = out.detach().cpu()
        return hook

    hooks = []
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.q_norm.register_forward_hook(get_hook(i, "q")))
        hooks.append(block.attn.k_norm.register_forward_hook(get_hook(i, "k")))

    # 4. Forward
    print("Running Calibration Forward Pass...")
    with torch.no_grad():
        model(x)

    # 5. Calculate PR
    print("\n" + "="*50)
    print(f"{'LAYER':<8} | {'HEAD':<6} | {'Q-PR':<8} | {'K-PR':<8} | {'UTIL %':<8}")
    print("-" * 50)
    
    head_dim = config['d_model'] // config['n_head']
    all_q_prs = []
    all_k_prs = []

    for l in range(config['n_layers']):
        # q, k shape: [B, T, n_head, head_dim]
        q = pr_data[l]["q"]
        k = pr_data[l]["k"]
        
        for h in range(config['n_head']):
            # Gather all tokens across batch/seq for this head
            q_head = q[:, :, h, :].reshape(-1, head_dim)
            k_head = k[:, :, h, :].reshape(-1, head_dim)
            
            q_pr = get_participation_ratio(q_head)
            k_pr = get_participation_ratio(k_head)
            
            util = (q_pr / head_dim) * 100
            print(f"L{l:<7} | H{h:<5} | {q_pr:>8.2f} | {k_pr:>8.2f} | {util:>7.1f}%")
            
            all_q_prs.append(q_pr)
            all_k_prs.append(k_pr)
        print("-" * 50)

    avg_q = sum(all_q_prs) / len(all_q_prs)
    avg_k = sum(all_k_prs) / len(all_k_prs)
    print(f"{'AVERAGE':<17} | {avg_q:>8.2f} | {avg_k:>8.2f} | {(avg_q/head_dim)*100:>7.1f}%")
    print("="*50)

    # Cleanup
    for h in hooks: h.remove()

if __name__ == "__main__":
    probe_model()
