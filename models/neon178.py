"""Neon178: Spectral MHI Giant (Spectral Synergy).
Variation of neon167.
- Multi-Head Intent (MHI): Each attention head has its own unique Intent gate.
- Spectral Pyramid Convolutions: Replaces single-scale k=3/9 convolutions with a bank of k=3, 9, and 27.
- Applied to both Attention (Q, K, V, I) and MLP Gate.
Calibration: 5M Class (d_model=272, d_ff=670).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class CausalSpectralPyramid(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv3  = nn.Conv1d(channels, channels, kernel_size=3,  groups=channels, bias=False)
        self.conv9  = nn.Conv1d(channels, channels, kernel_size=9,  groups=channels, bias=False)
        self.conv27 = nn.Conv1d(channels, channels, kernel_size=27, groups=channels, bias=False)
    
    def forward(self, x):
        # x: [B, T, C]
        x_t = x.transpose(1, 2)
        c3  = self.conv3(F.pad(x_t, (2, 0))).transpose(1, 2)
        c9  = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        c27 = self.conv27(F.pad(x_t, (26, 0))).transpose(1, 2)
        return c3 + c9 + c27

class SpectralMHIAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        # Projections: MHI (Q, K, V, I)
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        
        # Spectral Bank for each signal
        self.spec_q = CausalSpectralPyramid(d_model)
        self.spec_k = CausalSpectralPyramid(d_model)
        self.spec_v = CausalSpectralPyramid(d_model)
        self.spec_i = CausalSpectralPyramid(d_model)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw, i_raw = self.c_attn(x).split(C, dim=2)
        
        # Multi-scale spectral filtering
        q = self.spec_q(q_raw)
        k = self.spec_k(k_raw)
        v = self.spec_v(v_raw)
        intent = self.spec_i(i_raw)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class SpectralHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.spec = CausalSpectralPyramid(d_model)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        spectral_mix = self.spec(x)
        gate = torch.sigmoid(self.c_gate_proj(spectral_mix))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = SpectralMHIAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SpectralHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon178(nn.Module):
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
