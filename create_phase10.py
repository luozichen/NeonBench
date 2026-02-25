import os

template = """\"\"\"Neon{model_num}: Layer-Specific Conv-SplitBrain Attention (Layer {target_layer} Lookahead).
Phase 10: Layer-Targeted Intent + SplitBrain Fusion.
All heads in Layer {target_layer} use Staggered Lookahead masks.
All heads in other layers use strict causal masks.
Uses Conv1D on Q,K,V,Intent, and SiLU-Hydra MLP.
\"\"\"
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class ConvSplitBrainMHA_{model_num}(nn.Module):
    def __init__(self, config, is_lookahead_layer=False):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.block_size = config['block_size']
        self.is_lookahead_layer = is_lookahead_layer
        
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        
        # In-Projection convolutions (Full context blur)
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 1. Strict Causal Mask
        causal_mask = torch.tril(torch.ones(self.block_size, self.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, self.block_size, self.block_size).bool())
        
        # 2. Mask A (Even Block Boundaries)
        mask_a = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(0, self.block_size, 2):
            if i + 1 < self.block_size: mask_a[i, i+1] = 1.0
        self.register_buffer("mask_a", mask_a.view(1, 1, self.block_size, self.block_size).bool())
        
        # 3. Mask B (Odd Block Boundaries)
        mask_b = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(1, self.block_size - 1, 2):
            mask_b[i, i+1] = 1.0
        self.register_buffer("mask_b", mask_b.view(1, 1, self.block_size, self.block_size).bool())

    def forward(self, x, freqs_cos, freqs_sin, is_odd_stream=False):
        B, T, C = x.shape
        q_raw, k_raw, v_raw, intent_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        intent = self.conv_i(F.pad(intent_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        n_causal = 0 if self.is_lookahead_layer else self.n_head
        y_list = []
        
        if n_causal > 0:
            q_c, k_c, v_c = q[:, :n_causal], k[:, :n_causal], v[:, :n_causal]
            y_c = F.scaled_dot_product_attention(
                q_c, k_c, v_c,
                attn_mask=self.causal_mask[:, :, :T, :T]
            )
            y_list.append(y_c)
            
        if n_causal < self.n_head:
            q_l, k_l, v_l = q[:, n_causal:], k[:, n_causal:], v[:, n_causal:]
            look_mask = self.mask_b if is_odd_stream else self.mask_a
            y_l = F.scaled_dot_product_attention(
                q_l, k_l, v_l,
                attn_mask=look_mask[:, :, :T, :T]
            )
            y_list.append(y_l)
            
        attn_out = torch.cat(y_list, dim=1) if len(y_list) > 1 else y_list[0]
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

class PureHydraMLP_{model_num}(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        gate = F.silu(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block_{model_num}(nn.Module):
    def __init__(self, config, is_lookahead_layer=False):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = ConvSplitBrainMHA_{model_num}(config, is_lookahead_layer)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP_{model_num}(config)

    def forward(self, x, f_cos, f_sin, is_odd_stream=False):
        x = x + self.attn(self.ln1(x), f_cos, f_sin, is_odd_stream=is_odd_stream)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon{model_num}(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)

        self.blocks = nn.ModuleList([
            Block_{model_num}(config, is_lookahead_layer=(layer_idx == {target_layer}))
            for layer_idx in range(config['n_layers'])
        ])

        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight

        dim = config['d_model'] // config['n_head']
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size']).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, idx, targets=None, is_odd_stream=False):
        B, T = idx.shape
        x = self.token_emb(idx)
        
        f_cos, f_sin = self.freqs_cos[:T], self.freqs_sin[:T]
        for block in self.blocks:
            x = block(x, f_cos, f_sin, is_odd_stream=is_odd_stream)
        
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            flat_logits = logits.view(-1, self.config['vocab_size'])
            flat_targets = targets.view(-1)
            
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
            if is_odd_stream:
                mask[1::2] = False 
            else: 
                mask[0::2] = False 
            
            batch_mask = mask.repeat(B)
            loss_full = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = loss_full[batch_mask].mean()
            
        return logits, loss
"""

for model_num in range(253, 257):
    # mapping: neon253 -> target_layer=0, neon254 -> target_layer=1, ... neon256 -> target_layer=3
    target_layer = model_num - 253
    with open(f"models/neon{model_num}.py", "w") as f:
        f.write(template.format(model_num=model_num, target_layer=target_layer))

if __name__ == "__main__":
    print("Generated models neon253-256 for Phase 10 Layer-Specific Lookahead")
    
    # Test parameter count for neon253 (the structure parity is fixed)
    import sys
    sys.path.append('.') # Add NeonBench to path
    from check_params import count_non_embed
    from models.neon253 import Neon253

    def get_config():
        return {
            'vocab_size': 32000,
            'd_model': 272,
            'n_layers': 4,
            'n_head': 4,
            'd_ff': 1072, # From Phase 7 Intent+Conv
            'block_size': 1024,
            'device': 'cpu'
        }

    target = 5005616 # Expected parity for 5.0M non-embed boundary
    config = get_config()
    model = Neon253(config)
    params = count_non_embed(model)
    
    print(f"neon253 Parity Test (d_ff=1072): {params} non-embed parameters")
    if params == target:
        print("PERFECT PARITY MATCH!")
    else:
        print(f"WARNING: Parity mismatch! Expected {target}, got {params}. Diff: {params - target}")
