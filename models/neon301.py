"""Neon301: Gram-Schmidt Orthogonal Residuals Transformer.
Same architecture as Neon300 (Gated SDPA + SwiGLU) but each sub-layer's
residual is orthogonalized against all previous residuals per-token.

This forces every layer to contribute genuinely new information to the
residual stream, preventing redundant feature directions.
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
    """QKVI Attention with Sigmoid Intent Gate."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']

        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
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
        y = torch.sigmoid(intent) * attn_out

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = GatedSDPA(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)

    def forward(self, x, f_cos, f_sin):
        """Returns (output, attn_residual, mlp_residual) for orthogonalization."""
        attn_res = self.attn(self.ln1(x), f_cos, f_sin)
        x = x + attn_res  # Temporary; Neon301 will replace with orthogonalized version
        mlp_res = self.mlp(self.ln2(x))
        return attn_res, mlp_res


def gram_schmidt_project(residual, basis):
    """Project residual to be orthogonal to all vectors in basis.
    residual: [B, T, D]
    basis: [B, T, K, D] (where K is current number of basis vectors)
    """
    if basis is None:
        return residual
    
    # Force float32 for numerical stability in AMP (mixed precision)
    orig_dtype = residual.dtype
    r = residual.float()
    
    # Detach basis to stabilize gradients - we want to find 
    # the orthogonal component of the *current* residual relative
    # to the fixed state of previous residuals.
    b = basis.detach().float()
        
    # dots = residual . basis -> [B, T, 1, K]
    dots = torch.matmul(r.unsqueeze(-2), b.transpose(-1, -2))
    
    # norm_sqs = basis . basis -> [B, T, 1, K]
    norm_sqs = (b * b).sum(dim=-1, keepdim=True).transpose(-1, -2).clamp(min=1e-5)
    
    # coeffs = dots / norm_sqs -> [B, T, 1, K]
    coeffs = dots / norm_sqs
    
    # proj = coeffs . basis -> [B, T, 1, D]
    proj = torch.matmul(coeffs, b).squeeze(-2)
    
    return (r - proj).to(orig_dtype)


class Neon301(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
            self.token_emb.weight.data.copy_(warm_embeddings)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight  # Tied embeddings

        # RoPE frequencies
        dim = config['d_model'] // config['n_head']
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size']).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        f_cos, f_sin = self.freqs_cos[:T], self.freqs_sin[:T]

        # basis: [B, T, K, D] (starts as None, then grows)
        basis = None

        for block in self.blocks:
            attn_res, mlp_res = block(x, f_cos, f_sin)

            # 1. Attn Residual
            attn_res_orth = gram_schmidt_project(attn_res, basis)
            if basis is None:
                basis = attn_res_orth.unsqueeze(2)
            else:
                basis = torch.cat([basis, attn_res_orth.unsqueeze(2)], dim=2)
            x = x + attn_res_orth

            # 2. MLP Residual
            mlp_res_orth = gram_schmidt_project(mlp_res, basis)
            basis = torch.cat([basis, mlp_res_orth.unsqueeze(2)], dim=2)
            x = x + mlp_res_orth

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
