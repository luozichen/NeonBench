"""Neon140: Parallel Spectral Heads.
Each of the 4 attention heads specializes in a different kernel scale: k=3, 5, 7, 9.
The MLP (Hydra) uses a 4-path spectral gate merging these same scales.
Tests if diversity of receptive fields within a single layer is superior to a uniform scale.
Calibration: d_ff = 520.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class SpectralAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        # Projections
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.intent_proj = nn.Linear(d_model, d_model, bias=False) # Full Intent for head-specific gating
        
        # Parallel Spectral Kernels (Per Head)
        # Groups=d_model but each head only sees its own segment
        self.q_convs = nn.ModuleList([nn.Conv1d(self.head_dim, self.head_dim, kernel_size=k, groups=self.head_dim, bias=True) for k in [3, 5, 7, 9]])
        self.k_convs = nn.ModuleList([nn.Conv1d(self.head_dim, self.head_dim, kernel_size=k, groups=self.head_dim, bias=True) for k in [3, 5, 7, 9]])
        self.i_convs = nn.ModuleList([nn.Conv1d(self.head_dim, self.head_dim, kernel_size=k, groups=self.head_dim, bias=True) for k in [3, 5, 7, 9]])
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v = self.qkv_proj(x).split(C, dim=2)
        i_raw = self.intent_proj(x)
        
        # Split into heads and apply kernel-specific convolutions
        q_heads = q_raw.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # [B, H, T, D]
        k_heads = k_raw.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        i_heads = i_raw.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        q_out, k_out, i_out = [], [], []
        for h in range(self.n_head):
            k_size = 3 + (2 * h)
            q_h = self.q_convs[h](F.pad(q_heads[:, h].transpose(1, 2), (k_size - 1, 0))).transpose(1, 2)
            k_h = self.k_convs[h](F.pad(k_heads[:, h].transpose(1, 2), (k_size - 1, 0))).transpose(1, 2)
            i_h = self.i_convs[h](F.pad(i_heads[:, h].transpose(1, 2), (k_size - 1, 0))).transpose(1, 2)
            q_out.append(q_h)
            k_out.append(k_h)
            i_out.append(i_h)
            
        q = torch.stack(q_out, dim=1) # [B, H, T, D]
        k = torch.stack(k_out, dim=1)
        intent = torch.stack(i_out, dim=1)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MultiSpectralHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        
        # Parallel Spectral Gates
        self.convs = nn.ModuleList([nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, bias=False) for k in [3, 5, 7, 9]])
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        
        # Merge all spectral signals
        spectral_sum = 0
        for i, k in enumerate([3, 5, 7, 9]):
            spectral_sum += self.convs[i](F.pad(x_t, (k - 1, 0))).transpose(1, 2)
            
        gate = torch.sigmoid(self.c_gate_proj(spectral_sum))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = SpectralAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = MultiSpectralHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon140(nn.Module):
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

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x, self.freqs_cos, self.freqs_sin)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
