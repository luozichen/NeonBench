import os

# Base Template (Derived from neon220 but with more flexibility for Block and Intent logic)
TEMPLATE = """\"\"\"{docstring}
\"\"\"
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

{attn_class}

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
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        gate = F.silu(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = {attn_class_name}(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)

    def forward(self, x, f_cos, f_sin, z_i):
        a_out, z_i_new = self.attn(self.ln1(x), z_i, f_cos, f_sin)
        x = x + a_out
        x = x + self.mlp(self.ln2(x))
        return x, z_i_new

class {model_class}(nn.Module):
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
        B, T, D = x.shape
        z_i = torch.zeros(B, T, D, device=x.device)
        for block in self.blocks:
            x, z_i = block(x, self.freqs_cos, self.freqs_sin, z_i)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
"""

# neon226: Multi-Scale Spectral Decay
ATTN_226 = """
class MultiScaleSpectralDecayAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        
        # Fixed Geometric Bank: alphas from 0.01 to 1.0 log-spaced
        alphas = torch.exp(torch.linspace(math.log(0.01), math.log(1.0), d_model))
        self.register_buffer("alphas", alphas)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        # Spectral Decay: Multi-speed persistence
        delta = self.w_delta(x)
        z_i_new = self.alphas * z_i + (1.0 - self.alphas) * delta
        
        intent = self.conv_i(F.pad(z_i_new.transpose(1, 2), (2, 0))).transpose(1, 2)
        
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
        return self.c_proj(y), z_i_new

import math
"""

# neon227: Residual State Flow
ATTN_227 = """
class ResidualStateFlowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_x = nn.Linear(d_model, d_model, bias=False)
        self.w_z = nn.Linear(d_model, d_model, bias=False)
        self.z_norm = RMSNorm(d_model)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        # Deep Residual State Update
        z_gate = torch.sigmoid(self.w_x(x) + self.w_z(z_i))
        z_i_new = z_i + self.z_norm(z_gate * z_i)
        
        intent = self.conv_i(F.pad(z_i_new.transpose(1, 2), (2, 0))).transpose(1, 2)
        
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
        return self.c_proj(y), z_i_new
"""

# neon228: Cross-Head Intent Shuffling
ATTN_228 = """
class CrossHeadIntentShufflingAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(d_model))
        
        # Inter-Head Mixing: [n_head, n_head]
        self.head_mix = nn.Parameter(torch.eye(self.n_head))
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        alpha = torch.sigmoid(self.alpha_raw)
        z_i_new = alpha * z_i + (1.0 - alpha) * self.w_delta(x)
        
        # Cross-Head Mixing
        z_i_heads = z_i_new.view(B, T, self.n_head, self.head_dim)
        z_i_mixed = torch.einsum('bthd,hk->btkd', z_i_heads, self.head_mix)
        z_i_new = z_i_mixed.reshape(B, T, C)
        
        intent = self.conv_i(F.pad(z_i_new.transpose(1, 2), (2, 0))).transpose(1, 2)
        
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
        return self.c_proj(y), z_i_new
"""

# neon229: Hadamard Gain Modulation
ATTN_229 = """
class HadamardGainModulationAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(d_model))
        
        # Map intent stream to per-head query/key scale (Gain)
        self.w_gain = nn.Linear(d_model, self.n_head, bias=False)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        alpha = torch.sigmoid(self.alpha_raw)
        z_i_new = alpha * z_i + (1.0 - alpha) * self.w_delta(x)
        intent = self.conv_i(F.pad(z_i_new.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        # Calculate Gain per token per head
        gain = torch.exp(self.w_gain(intent)).view(B, T, self.n_head, 1)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        # Dynamically scale Q by the Intent-derived Gain
        q = q * gain
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Traditional attention path (the gain is already inside q)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # We don't sigmoid gate here; we influenced the distribution instead!
        y = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), z_i_new
"""

MODELS = {
    "neon226": {
        "docstring": "Neon226: Multi-Scale Spectral Decay",
        "attn_class": ATTN_226,
        "attn_class_name": "MultiScaleSpectralDecayAttention",
        "model_class": "Neon226"
    },
    "neon227": {
        "docstring": "Neon227: Residual State Flow",
        "attn_class": ATTN_227,
        "attn_class_name": "ResidualStateFlowAttention",
        "model_class": "Neon227"
    },
    "neon228": {
        "docstring": "Neon228: Cross-Head Intent Shuffling",
        "attn_class": ATTN_228,
        "attn_class_name": "CrossHeadIntentShufflingAttention",
        "model_class": "Neon228"
    },
    "neon229": {
        "docstring": "Neon229: Hadamard Gain Modulation",
        "attn_class": ATTN_229,
        "attn_class_name": "HadamardGainModulationAttention",
        "model_class": "Neon229"
    }
}

for name, cfg in MODELS.items():
    content = TEMPLATE.format(**cfg)
    with open(f"models/{name}.py", "w") as f:
        f.write(content)
    print(f"Generated models/{name}.py")
