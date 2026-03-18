"""Neon306: Intent-Space Orthogonality Transformer.
Applies Gram-Schmidt to the Gated 'Intent' vectors in Attention blocks.
Goal: Force layers to 'attend' to unique feature subspaces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    cos = freqs_cos[:x.shape[-3]].view(1, x.shape[-3], 1, -1)
    sin = freqs_sin[:x.shape[-3]].view(1, x.shape[-3], 1, -1)
    return torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)

def gram_schmidt_project(residual, basis_list):
    """Sequential MGS Projection in FP32."""
    if not basis_list:
        return residual
    orig_dtype = residual.dtype
    r = residual.float()
    for b in basis_list:
        b_f = b.float()
        dot = (r * b_f).sum(dim=-1, keepdim=True)
        norm_sq = (b_f * b_f).sum(dim=-1, keepdim=True).clamp(min=1e-9)
        r = r - (dot / norm_sq) * b_f
    return r.to(orig_dtype)

class GatedSDPA_OrthoIntent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']; self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim); self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, intent_basis):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)
        
        # Orthogonalize the Intent vector
        intent_orth = gram_schmidt_project(intent, intent_basis)
        
        q = q.view(B, T, self.n_head, self.head_dim); k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim); intent_orth = intent_orth.view(B, T, self.n_head, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin); k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2); intent_orth = intent_orth.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Store the orthogonalized intent in the basis (must be same shape as projection)
        # Note: intent was (B, T, C) before viewing. We store the (B, T, C) version.
        res = self.c_proj((torch.sigmoid(intent_orth) * attn_out).transpose(1, 2).contiguous().view(B, T, C))
        return res, intent_orth.transpose(1, 2).contiguous().view(B, T, C)

class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']; d_ff = config['d_ff']
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False); self.w_down = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model']); self.attn = GatedSDPA_OrthoIntent(config)
        self.ln2 = RMSNorm(config['d_model']); self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin, intent_basis):
        attn_res, intent_orth = self.attn(self.ln1(x), f_cos, f_sin, intent_basis)
        x = x + attn_res
        mlp_res = self.mlp(self.ln2(x))
        x = x + mlp_res
        return x, intent_orth

class Neon306(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight
        dim = config['d_model'] // config['n_head']
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size']).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs)); self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, idx, targets=None):
        B, T = idx.shape; x = self.token_emb(idx)
        f_cos, f_sin = self.freqs_cos[:T], self.freqs_sin[:T]
        
        intent_basis = []
        for block in self.blocks:
            x, intent_orth = block(x, f_cos, f_sin, intent_basis)
            intent_basis.append(intent_orth)
            
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
