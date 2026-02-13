"""Neon163: Alternating Ghost (8 Layers).
Iterative Hybrid: Silent -> Attention -> Silent -> Attention.
Tests if frequent global context updates stabilize the deep silent backbone.
Calibration: d_ff = 185.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class SilentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_g = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        v = self.conv_v(F.pad(self.v_proj(x).transpose(1, 2), (2, 0))).transpose(1, 2)
        g = self.conv_g(F.pad(self.g_proj(x).transpose(1, 2), (8, 0))).transpose(1, 2)
        y = torch.sigmoid(g) * v
        return self.c_proj(y)

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.intent_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.conv_i = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=3, groups=self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, f_cos, f_sin):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).split(C, dim=2)
        i_raw = self.intent_proj(x)
        intent = self.conv_i(F.pad(i_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        q, k, v = [t.view(B, T, self.n_head, self.head_dim) for t in (q, k, v)]
        intent = intent.view(B, T, 1, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb(q, f_cos, f_sin), apply_rotary_emb(k, f_cos, f_sin)
        q, k, v, intent = [t.transpose(1, 2) for t in (q, k, v, intent)]
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class PureHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model, d_ff = config['d_model'], config['d_ff']
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        c9 = self.conv9(F.pad(x.transpose(1, 2), (8, 0))).transpose(1, 2)
        gate = torch.sigmoid(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config, is_attention=False):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.is_attention = is_attention
        if is_attention: self.attn = AttentionBlock(config)
        else: self.gated = SilentBlock(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)
    def forward(self, x, f_cos=None, f_sin=None):
        if self.is_attention: x = x + self.attn(self.ln1(x), f_cos, f_sin)
        else: x = x + self.gated(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Neon163(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None: self.token_emb.weight.data.copy_(warm_embeddings)
        self.blocks = nn.ModuleList([Block(config, is_attention=(i%2==1)) for i in range(8)])
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
        for i, block in enumerate(self.blocks):
            if block.is_attention: x = block(x, self.freqs_cos, self.freqs_sin)
            else: x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
