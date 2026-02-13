"""Neon135: Holographic Projection.
Implements Complex-Valued Attention projections (Magnitude + Phase).
The Attention signal is computed in complex-space, allowing for constructive/destructive interference.
Tests if complex-domain representations capture higher-order sequence relationships.
Calibration: d_ff = 310.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.re = nn.Linear(in_features, out_features, bias=bias)
        self.im = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # x is real [B, T, D]
        return torch.complex(self.re(x), self.im(x))

class HolographicAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        # Complex projections: [B, T, D] -> [B, T, D] (complex)
        self.q_proj = ComplexLinear(d_model, d_model, bias=False)
        self.k_proj = ComplexLinear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False) # Content stays real for simplicity
        self.i_proj = nn.Linear(d_model, self.head_dim, bias=False) # Intent stays real
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        
        # 1. Complex Projections
        q_z = self.q_proj(x) # [B, T, C] complex
        k_z = self.k_proj(x) # [B, T, C] complex
        v = self.v_proj(x)
        intent = self.i_proj(x).view(B, T, 1, self.head_dim)
        
        # 2. Extract Phase/Magnitude for Rotary-style complex rotation
        # but here we just take the magnitude-normalized dot product effectively
        q = q_z.view(B, T, self.n_head, self.head_dim)
        k = k_z.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        # We can't use SDPA directly on complex tensors easily in all torch versions
        # So we'll compute Heron scores: Re(q * conj(k))
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2) # [B, H, T, D]
        v = v.transpose(1, 2) # [B, H, T, D]
        
        # Compute Complex Dot Product: (Q_re + iQ_im) * (K_re - iK_im)
        # Re = Q_re*K_re + Q_im*K_im
        # Im = Q_im*K_re - Q_re*K_im
        attn_scores = (torch.matmul(q.real, k.real.transpose(-1, -2)) + 
                       torch.matmul(q.imag, k.imag.transpose(-1, -2)))
        
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        # Gating
        y = torch.sigmoid(intent.transpose(1, 2)) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

class PureHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        gate = torch.sigmoid(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = HolographicAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon135(nn.Module):
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
