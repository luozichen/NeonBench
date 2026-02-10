"""Neon007: neon005 + DeltaNet (Linear Attention with Delta Rule).
Replaces softmax attention with a sequential recurrence using the delta rule."""
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

class DeltaNetAttention(nn.Module):
    """DeltaNet: linear attention with delta rule memory updates.
    S_t = S_{t-1} + beta_t * (v_t - S_{t-1}^T k_t) outer k_t
    o_t = S_t^T q_t
    No RoPE needed — recurrence provides inherent ordering."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, self.n_head, bias=False)  # per-head gate
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)

        # L2 normalize keys for recurrence stability
        k = F.normalize(k, dim=-1)

        # Beta gate: controls update strength
        beta = torch.sigmoid(self.beta_proj(x))  # (B, T, n_head)

        # Transpose to (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2).unsqueeze(-1)  # (B, n_head, T, 1)

        # Delta rule recurrence
        # S: (B, n_head, head_dim, head_dim) — associative memory
        S = torch.zeros(B, self.n_head, self.head_dim, self.head_dim,
                        device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            k_t = k[:, :, t, :]     # (B, n_head, head_dim)
            v_t = v[:, :, t, :]
            q_t = q[:, :, t, :]
            beta_t = beta[:, :, t]   # (B, n_head, 1)

            # Retrieve current prediction: S^T k_t
            retrieved = torch.einsum('bhde,bhe->bhd', S, k_t)

            # Delta update: S += beta * (v - retrieved) outer k
            delta = v_t - retrieved
            S = S + beta_t.unsqueeze(-1) * torch.einsum('bhd,bhe->bhde', delta, k_t)

            # Output: S^T q_t
            o_t = torch.einsum('bhde,bhe->bhd', S, q_t)
            outputs.append(o_t)

        y = torch.stack(outputs, dim=2)  # (B, n_head, T, head_dim)
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
        self.attn = DeltaNetAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon007(nn.Module):
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
        # Still register RoPE buffers for interface compatibility (unused by DeltaNet)
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
