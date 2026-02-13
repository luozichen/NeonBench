"""Neon146: Multi-Head Global Hum.
Variation of neon142 (Global Hum) using Multi-Head Intent (MHI).
Each attention head has its own global bias signal for context selection.
Tests if 'Locality + Global Bias' is more effective when diversified across heads.
Calibration: d_ff = 507.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class MultiHeadGlobalHumAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        # Projections (Full Multi-Head Intent)
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        
        # Local Head-specific context
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        # Global Hum (Head-specific causal mean)
        self.hum_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Search
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw, i_raw = self.c_attn(x).split(C, dim=2)
        
        # 1. Local Head-Specific Gate
        intent_local = self.conv_i(F.pad(i_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        # 2. Global Head-Specific Hum
        hum_raw = self.hum_proj(x)
        hum_cumsum = torch.cumsum(hum_raw, dim=1)
        indices = torch.arange(1, T + 1, device=x.device).view(1, T, 1)
        intent_global = hum_cumsum / indices
        
        intent_all = intent_local + intent_global
        
        # Split into heads
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2).view(B, T, self.n_head, self.head_dim)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2).view(B, T, self.n_head, self.head_dim)
        v = v_raw.view(B, T, self.n_head, self.head_dim)
        intent = intent_all.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class GlobalHumMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.conv_local = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self.hum_proj = nn.Linear(d_model, d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        g_local = self.conv_local(F.pad(x_t, (8, 0))).transpose(1, 2)
        h_raw = self.hum_proj(x)
        h_cumsum = torch.cumsum(h_raw, dim=1)
        indices = torch.arange(1, T + 1, device=x.device).view(1, T, 1)
        g_global = h_cumsum / indices
        gate = torch.sigmoid(self.c_gate_proj(g_local + g_global))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = MultiHeadGlobalHumAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = GlobalHumMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon146(nn.Module):
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
