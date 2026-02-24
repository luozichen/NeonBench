import os

# Base Template (Derived from neon220 but with placeholders)
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
        {mlp_init_extra}
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x{mlp_fwd_args}):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        {mlp_fwd_logic}
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = {attn_class_name}(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)

    def forward(self, x, f_cos, f_sin, z_i{block_fwd_args}):
        a_out, z_i_new = self.attn(self.ln1(x), z_i{attn_call_args})
        x = x + a_out
        x = x + self.mlp(self.ln2(x){mlp_call_args})
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
        inv_freq = 1.0 / ({rope_base} ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size']).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))
        {model_init_extra}

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        B, T, D = x.shape
        z_i = torch.zeros(B, T, D, device=x.device)
        {model_fwd_pre}
        for block in self.blocks:
            x, z_i = block(x, self.freqs_cos, self.freqs_sin, z_i{model_block_call_args})
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
"""

ATTN_222 = """
class FullMultiHeadConvAttentionGatedDecay(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.w_alpha = nn.Linear(d_model, d_model, bias=False)
        
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
        
        # Gated Decay: Alpha depends on Semantic State
        delta = self.w_delta(x)
        alpha = torch.sigmoid(self.w_alpha(x))
        z_i_new = alpha * z_i + (1.0 - alpha) * delta
        
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

ATTN_223 = """
class FullMultiHeadConvAttentionDivergentRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(d_model))
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, f_cos, f_sin, i_cos, i_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        alpha = torch.sigmoid(self.alpha_raw)
        z_i_new = alpha * z_i + (1.0 - alpha) * self.w_delta(x)
        intent = self.conv_i(F.pad(z_i_new.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        # Apply NORMAL rope to Q, K
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # intent uses SLOW rope? Wait, the plan says Intent heads use slow rope.
        # But intent isn't an attention projection in this specific series, it's a gating factor.
        # However, for consistency with the spirit, we'll apply slow rope to intent heads if we had them.
        # In this gating architecture, intent IS the gating factor. 
        # Actually, let's apply RoPE to the Intent projection to stabilize its temporal gating.
        intent = apply_rotary_emb(intent, i_cos, i_sin)
        
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), z_i_new
"""

ATTN_224 = """
class FullMultiHeadConvAttentionWideIntent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(d_model))
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        # MASSIVE KERNEL for Intent
        self.conv_i = nn.Conv1d(d_model, d_model, kernel_size=33, groups=d_model, bias=False)
        
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
        # Pad 32 for k=33
        intent = self.conv_i(F.pad(z_i_new.transpose(1, 2), (32, 0))).transpose(1, 2)
        
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

ATTN_225 = """
class FullMultiHeadConvAttentionIntentDecay(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(d_model))
        
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

MODELS = {
    "neon222": {
        "docstring": "Neon222: State-Dependent Gated Decay (GRU-Intent)",
        "attn_class": ATTN_222,
        "attn_class_name": "FullMultiHeadConvAttentionGatedDecay",
        "mlp_init_extra": "",
        "mlp_fwd_args": "",
        "mlp_fwd_logic": "gate = F.silu(self.c_gate_proj(c9))",
        "block_fwd_args": "",
        "attn_call_args": ", f_cos, f_sin",
        "mlp_call_args": "",
        "rope_base": "10000.0",
        "model_init_extra": "",
        "model_fwd_pre": "",
        "model_block_call_args": ""
    },
    "neon223": {
        "docstring": "Neon223: Frequency-Divergent RoPE (Multi-Speed Routing)",
        "attn_class": ATTN_223,
        "attn_class_name": "FullMultiHeadConvAttentionDivergentRoPE",
        "mlp_init_extra": "",
        "mlp_fwd_args": "",
        "mlp_fwd_logic": "gate = F.silu(self.c_gate_proj(c9))",
        "block_fwd_args": ", i_cos, i_sin",
        "attn_call_args": ", f_cos, f_sin, i_cos, i_sin",
        "mlp_call_args": "",
        "rope_base": "10000.0",
        "model_init_extra": """
        inv_freq_slow = 1.0 / (1000000.0 ** (torch.arange(0, dim, 2).float() / dim))
        freqs_slow = torch.outer(t, inv_freq_slow)
        self.register_buffer("slow_cos", torch.cos(freqs_slow))
        self.register_buffer("slow_sin", torch.sin(freqs_slow))
""",
        "model_fwd_pre": "",
        "model_block_call_args": ", self.slow_cos, self.slow_sin"
    },
    "neon224": {
        "docstring": "Neon224: Divergent Kernel Scales (Wide Context Router)",
        "attn_class": ATTN_224,
        "attn_class_name": "FullMultiHeadConvAttentionWideIntent",
        "mlp_init_extra": "",
        "mlp_fwd_args": "",
        "mlp_fwd_logic": "gate = F.silu(self.c_gate_proj(c9))",
        "block_fwd_args": "",
        "attn_call_args": ", f_cos, f_sin",
        "mlp_call_args": "",
        "rope_base": "10000.0",
        "model_init_extra": "",
        "model_fwd_pre": "",
        "model_block_call_args": ""
    },
    "neon225": {
        "docstring": "Neon225: Intent-Steered MLP",
        "attn_class": ATTN_225,
        "attn_class_name": "FullMultiHeadConvAttentionIntentDecay",
        "mlp_init_extra": "self.w_iz = nn.Linear(d_model, d_ff, bias=False)",
        "mlp_fwd_args": ", z_i",
        "mlp_fwd_logic": "gate = F.silu(self.c_gate_proj(c9) + self.w_iz(z_i))",
        "block_fwd_args": "",
        "attn_call_args": ", f_cos, f_sin",
        "mlp_call_args": ", z_i_new",
        "rope_base": "10000.0",
        "model_init_extra": "",
        "model_fwd_pre": "",
        "model_block_call_args": ""
    }
}

for name, config in MODELS.items():
    cfg = config.copy()
    cfg['model_class'] = name.capitalize()
    content = TEMPLATE.format(**cfg)
    with open(f"models/{name}.py", "w") as f:
        f.write(content)
    print(f"Generated models/{name}.py")
