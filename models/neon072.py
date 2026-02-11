"""Neon072: Gated-Residual Hydra.
Combines standard Linear Gate (SiLU) with Hydra Attention Gate (Sigmoid).
Gate = SiLU(x W_g) + Sigmoid(Attn(x))
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb
# Reuse IntentAttention for the main block
from models.neon070 import IntentAttention

class ResHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff'] # Standard 512 likely
        self.head_dim = config['d_model'] // config['n_head']
        
        # Standard Linear Gate
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        
        # Hydra Gate Components
        self.c_gate_attn = nn.Linear(d_model, 3 * self.head_dim, bias=False)
        self.c_gate_proj = nn.Linear(self.head_dim, d_ff, bias=False)
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # Up
        self.w2 = nn.Linear(d_ff, d_model, bias=False) # Down

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        
        # 1. Compute Standard Gate
        linear_gate = F.silu(self.w_gate(x))
        
        # 2. Compute Hydra Attention Gate
        q, k, v = self.c_gate_attn(x).split(self.head_dim, dim=2)
        f_cos = freqs_cos[..., :self.head_dim//2]
        f_sin = freqs_sin[..., :self.head_dim//2]
        
        q = q.view(B, T, 1, self.head_dim)
        k = k.view(B, T, 1, self.head_dim)
        v = v.view(B, T, 1, self.head_dim)
        
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = context.transpose(1, 2).contiguous().view(B, T, self.head_dim)
        
        attn_gate = torch.sigmoid(self.c_gate_proj(context))
        
        # 3. Combine Gates (Add)
        gate = linear_gate + attn_gate
        
        # 4. Apply
        h = gate * self.w1(x)
        return self.w2(h)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = ResHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon072(nn.Module):
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

        # Freqs
        dim = config['d_model'] // config['n_head']
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size']).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x, self.freqs_cos, self.freqs_sin)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
