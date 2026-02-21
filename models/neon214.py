"""Neon214: Narrow & Deep Architecture (192d, 36 layers, 6 heads, head_dim=32).
Designed based on PR Audit showing 80% dimensional collapse in neon213.
Reinvests width into extreme depth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .neon015 import RMSNorm, apply_rotary_emb

class DeepConvAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.k = config.get('conv_k', 21) # k=21 is the new standard

        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)
        pad = self.k - 1
        
        # Communicative Convolutions
        q = self.conv_q(F.pad(q.transpose(1,2), (pad, 0))).transpose(1,2)
        k = self.conv_k(F.pad(k.transpose(1,2), (pad, 0))).transpose(1,2)
        v = self.conv_v(F.pad(v.transpose(1,2), (pad, 0))).transpose(1,2)
        intent = self.conv_i(F.pad(intent.transpose(1,2), (pad, 0))).transpose(1,2)
        
        # QK-Norm (Crucial for PR Health)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        intent = intent.transpose(1,2)
        
        # Attention with Sigmoid Intent Gating
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.c_proj(y)

class DeepHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.k = config.get('mlp_k', 21)
        self.conv_gate = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        pad = self.k - 1
        c = self.conv_gate(F.pad(x_t, (pad, 0))).transpose(1, 2)
        gate = F.silu(self.c_gate_proj(c))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = DeepConvAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = DeepHydraMLP(config)
        
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon214(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        
        # Weight Tying
        self.token_emb.weight = self.head.weight
        
        # RoPE Frequencies
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
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
        return logits, loss
