"""Neon210: Differential Attention with Intent Gating.
True Differential Transformer: splits Q and K into halves, computes two
softmax attention maps, and subtracts them to cancel common-mode noise.
Combined with our Intent sigmoid gating and Hydra conv MLP.
Attn = [softmax(Q1·K1) - λ·softmax(Q2·K2)] · V, then gated by σ(I).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class DifferentialConvAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        self.half_dim = self.head_dim // 2
        d_model = config['d_model']

        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable lambda per head, initialized at 0.5
        self.lambda_param = nn.Parameter(torch.full((self.n_head,), 0.5))

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)

        q = self.conv_q(F.pad(q.transpose(1,2), (2,0))).transpose(1,2)
        k = self.conv_k(F.pad(k.transpose(1,2), (2,0))).transpose(1,2)
        v = self.conv_v(F.pad(v.transpose(1,2), (2,0))).transpose(1,2)
        intent = self.conv_i(F.pad(intent.transpose(1,2), (2,0))).transpose(1,2)

        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)

        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        intent = intent.transpose(1, 2)

        # Split Q and K into positive/negative halves
        q1, q2 = q[..., :self.half_dim], q[..., self.half_dim:]
        k1, k2 = k[..., :self.half_dim], k[..., self.half_dim:]

        # Manual attention (can't use SDPA for differential subtraction)
        scale = self.half_dim ** -0.5
        scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores1 = scores1.masked_fill(causal_mask, float('-inf'))
        scores2 = scores2.masked_fill(causal_mask, float('-inf'))

        w1 = F.softmax(scores1, dim=-1)
        w2 = F.softmax(scores2, dim=-1)

        # Differential: subtract noise map
        lam = self.lambda_param.view(1, self.n_head, 1, 1)
        diff_weights = w1 - lam * w2

        # Apply to full V
        attn_out = torch.matmul(diff_weights, v)

        # Apply intent gate
        y = torch.sigmoid(intent) * attn_out
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

    def forward(self, x):
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        gate = F.silu(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = DifferentialConvAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon210(nn.Module):
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
