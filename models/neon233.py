"""Neon233: Staggered Block-Causal Attention Model (5M class).
Implements a 1-step quasi-encoder lookahead via alternating masks.
Natively handles Even ((0,1),(2,3)) and Odd ((0),(1,2),(3,4)) alignment.
No <Z> token or physical shift required.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class StaggeredMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.block_size = config['block_size']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute Mask A: Even Blocks (0,1), (2,3), ...
        mask_a = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(0, self.block_size, 2):
            if i + 1 < self.block_size:
                mask_a[i, i+1] = 1.0
        self.register_buffer("mask_a", mask_a.view(1, 1, self.block_size, self.block_size))
        
        # Precompute Mask B: Staggered Blocks (0), (1,2), (3,4), (5)
        mask_b = torch.tril(torch.ones(self.block_size, self.block_size))
        # Group (1,2), (3,4)... i starts at 1
        for i in range(1, self.block_size - 1, 2):
            mask_b[i, i+1] = 1.0 # Token i sees i+1
        self.register_buffer("mask_b", mask_b.view(1, 1, self.block_size, self.block_size))

    def forward(self, x, freqs_cos, freqs_sin, is_odd_stream=False):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = q_raw.view(B, T, self.n_head, self.head_dim)
        k = k_raw.view(B, T, self.n_head, self.head_dim)
        v = v_raw.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Select Mask
        mask = self.mask_b if is_odd_stream else self.mask_a
        
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask[:, :, :T, :T]
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = StaggeredMHA(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin, is_odd_stream=False):
        x = x + self.attn(self.ln1(x), f_cos, f_sin, is_odd_stream=is_odd_stream)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon233(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])

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
        
        f_cos = self.freqs_cos[:T]
        f_sin = self.freqs_sin[:T]
        
        for block in self.blocks:
            x = block(x, f_cos, f_sin, is_odd_stream=is_odd_stream)
            
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            flat_logits = logits.view(-1, self.config['vocab_size'])
            flat_targets = targets.view(-1)
            
            # Loss Masking
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
            if is_odd_stream:
                # Mask B (Staggered): Loss on 0, 2, 4... Mask 1, 3, 5...
                mask[1::2] = False
            else:
                # Mask A (Even): Loss on 1, 3, 5... Mask 0, 2, 4...
                mask[0::2] = False
            
            batch_mask = mask.repeat(B)
            loss_full = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = loss_full[batch_mask].mean()
            
        return logits, loss
