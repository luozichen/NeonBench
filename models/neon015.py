"""Neon015: QKVI per-head Intent Attention — Result gating, raw I, raw V.
Formula: Output_i = I_i ⊙ Σ_j(A_ij V_j)
Intent is projected per-head (same as Q, K, V)."""
import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def apply_rotary_emb(x, freqs_cos, freqs_sin):
    d = x.shape[-1]
    x_even, x_odd = x[..., :d:2], x[..., 1:d:2]
    cos = freqs_cos[:x.shape[1]].view(1, x.shape[1], 1, -1)
    sin = freqs_sin[:x.shape[1]].view(1, x.shape[1], 1, -1)
    return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)

class IntentAttention(nn.Module):
    """QKVI Attention: Q, K, V, I all projected per-head.
    Result gating: Output = I ⊙ Attn(Q, K, V)"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']

        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)  # Q, K, V, I
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)

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
        y = intent * attn_out  # Result gating: raw I, raw V

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w_gate = nn.Linear(config['d_model'], config['d_ff'], bias=False)
        self.w_up   = nn.Linear(config['d_model'], config['d_ff'], bias=False)
        self.w_down = nn.Linear(config['d_ff'], config['d_model'], bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon015(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
            assert warm_embeddings.shape == (config['vocab_size'], config['d_model'])
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
