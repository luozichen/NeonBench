"""Neon302: Gram-Schmidt Orthogonal Residuals Transformer.
Exactly replicates the version from git history:
1. Classical (simultaneous) Gram-Schmidt.
2. List-based basis (no normalization).
3. No detach (live gradients).
4. @torch.compiler.disable on projection.
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

class GatedSDPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']; self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim); self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin); k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2); intent = intent.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj((torch.sigmoid(intent) * attn_out).transpose(1, 2).contiguous().view(B, T, C))

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
        self.ln1 = RMSNorm(config['d_model']); self.attn = GatedSDPA(config)
        self.ln2 = RMSNorm(config['d_model']); self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin):
        attn_res = self.attn(self.ln1(x), f_cos, f_sin)
        x = x + attn_res # ML saw raw residual
        mlp_res = self.mlp(self.ln2(x))
        return attn_res, mlp_res

@torch.compiler.disable
def gram_schmidt_project(residual, basis_list):
    """Project residual to be orthogonal to all vectors in basis_list.
    Calculates projections simultaneously.
    """
    if not basis_list:
        return residual
    basis = torch.stack(basis_list, dim=2)
    r = residual.unsqueeze(2)
    dots = (r * basis).sum(dim=-1)
    norm_sqs = (basis * basis).sum(dim=-1).clamp(min=1e-8)
    coeffs = dots / norm_sqs
    proj = (coeffs.unsqueeze(-1) * basis).sum(dim=2)
    return residual - proj

class Neon302(nn.Module):
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
        basis = []
        for block in self.blocks:
            attn_res, mlp_res = block(x, f_cos, f_sin)
            attn_res_orth = gram_schmidt_project(attn_res, basis)
            basis.append(attn_res_orth); x = x + attn_res_orth
            mlp_res_orth = gram_schmidt_project(mlp_res, basis)
            basis.append(mlp_res_orth); x = x + mlp_res_orth
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
