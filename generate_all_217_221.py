import os

template = """\"\"\"{model_name}: {desc}
2. Variation of neon185 (5M class).
3. {details}
4. Architecture: Full Multi-Head Conv-Attention + Swish Hydra MLP + Intent Stream.
\"\"\"
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb
{extra_imports}

{classes}

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = {attn_class}(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = {mlp_class}(config)
{init_extras}
    def forward(self, x, f_cos, f_sin{fwd_args}):
        {fwd_logic}
        return x{fwd_returns}

class {model_class}(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
{network_init_extras}
        self.blocks = nn.ModuleList([Block(config) for _ in range(config['n_layers'])])
{network_post_init}
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
{stream_init}
        for block in self.blocks:
            {block_call}
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
"""

c217 = """
class FullMultiHeadConvAttentionIntentStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        
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
        
        # Intent Stream Update
        delta = self.w_delta(x)
        z_i_new = z_i + delta
        # Apply intent blur
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
"""

c218 = """
class FullMultiHeadConvAttentionBottleneckStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        d_route = 32
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_route, bias=False)
        self.w_expand = nn.Linear(d_route, d_model, bias=False)
        
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
        
        # Intent Stream Update
        delta = self.w_delta(x)
        z_i_new = z_i + delta
        intent_expanded = self.w_expand(z_i_new)
        # Apply intent blur
        intent = self.conv_i(F.pad(intent_expanded.transpose(1, 2), (2, 0))).transpose(1, 2)
        
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
"""

c219 = """
class FullMultiHeadConvAttentionUniversalStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, w_delta_shared, conv_i_shared, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        # Intent Stream Update with Shared Parameters
        delta = F.linear(x, w_delta_shared)
        z_i_new = z_i + delta
        # Apply intent blur
        intent = conv_i_shared(F.pad(z_i_new.transpose(1, 2), (2, 0))).transpose(1, 2)
        
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
"""

c220 = """
class FullMultiHeadConvAttentionDecayStream(nn.Module):
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
        
        # Intent Stream Update with Decay
        delta = self.w_delta(x)
        alpha = torch.sigmoid(self.alpha_raw)
        z_i_new = alpha * z_i + (1.0 - alpha) * delta
        # Apply intent blur
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
"""

c221 = """
class FullMultiHeadConvAttentionIntentStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        
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
        
        # Intent Stream Update
        delta = self.w_delta(x)
        z_i_new = z_i + delta
        # Apply intent blur
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
"""

cfgs = [
    {
        'model_name': 'Neon217',
        'desc': 'Full Rank Independent Stream',
        'details': 'Intent Stream (Z_I) is fully disjoint and updated residually.',
        'classes': c217,
        'attn_class': 'FullMultiHeadConvAttentionIntentStream',
        'mlp_class': 'PureHydraMLP',
        'fwd_args': ', z_i',
        'fwd_logic': 'a_out, z_i_new = self.attn(self.ln1(x), z_i, f_cos, f_sin)\n        x = x + a_out\n        x = x + self.mlp(self.ln2(x))',
        'fwd_returns': ', z_i_new',
        'stream_init': '        z_i = torch.zeros(B, T, D, device=x.device)',
        'block_call': 'x, z_i = block(x, f_cos, f_sin, z_i)',
        'init_extras': '',
        'network_init_extras': '',
        'network_post_init': '',
        'extra_imports': ''
    },
    {
        'model_name': 'Neon218',
        'desc': 'Bottleneck Stream',
        'details': 'Intent Stream (Z_I) runs at reduced rank (d_route=32).',
        'classes': c218,
        'attn_class': 'FullMultiHeadConvAttentionBottleneckStream',
        'mlp_class': 'PureHydraMLP',
        'fwd_args': ', z_i',
        'fwd_logic': 'a_out, z_i_new = self.attn(self.ln1(x), z_i, f_cos, f_sin)\n        x = x + a_out\n        x = x + self.mlp(self.ln2(x))',
        'fwd_returns': ', z_i_new',
        'stream_init': '        z_i = torch.zeros(B, T, 32, device=x.device)',
        'block_call': 'x, z_i = block(x, f_cos, f_sin, z_i)',
        'init_extras': '',
        'network_init_extras': '',
        'network_post_init': '',
        'extra_imports': ''
    },
    {
        'model_name': 'Neon219',
        'desc': 'Strict Layer-Tying',
        'details': 'Shares exactly one delta mapping and conv mapping matrix across all layers.',
        'classes': c219,
        'attn_class': 'FullMultiHeadConvAttentionUniversalStream',
        'mlp_class': 'PureHydraMLP',
        'fwd_args': ', z_i, w_delta, conv_i',
        'fwd_logic': 'a_out, z_i_new = self.attn(self.ln1(x), z_i, w_delta, conv_i, f_cos, f_sin)\n        x = x + a_out\n        x = x + self.mlp(self.ln2(x))',
        'fwd_returns': ', z_i_new',
        'stream_init': '        z_i = torch.zeros(B, T, D, device=x.device)',
        'block_call': 'x, z_i = block(x, f_cos, f_sin, z_i, self.shared_w_delta.weight, self.shared_conv_i)',
        'init_extras': '',
        'network_init_extras': '        self.shared_w_delta = nn.Linear(config[\'d_model\'], config[\'d_model\'], bias=False)\n        self.shared_conv_i = nn.Conv1d(config[\'d_model\'], config[\'d_model\'], kernel_size=3, groups=config[\'d_model\'], bias=False)',
        'network_post_init': '',
        'extra_imports': ''
    },
    {
        'model_name': 'Neon220',
        'desc': 'Momentum Decay Stream',
        'details': 'Uses a sequence mixing alpha decay for history.',
        'classes': c220,
        'attn_class': 'FullMultiHeadConvAttentionDecayStream',
        'mlp_class': 'PureHydraMLP',
        'fwd_args': ', z_i',
        'fwd_logic': 'a_out, z_i_new = self.attn(self.ln1(x), z_i, f_cos, f_sin)\n        x = x + a_out\n        x = x + self.mlp(self.ln2(x))',
        'fwd_returns': ', z_i_new',
        'stream_init': '        z_i = torch.zeros(B, T, D, device=x.device)',
        'block_call': 'x, z_i = block(x, f_cos, f_sin, z_i)',
        'init_extras': '',
        'network_init_extras': '',
        'network_post_init': '',
        'extra_imports': ''
    },
    {
        'model_name': 'Neon221',
        'desc': 'Cross-Linked Stream',
        'details': 'Intent Stream directly modifies the Semantic stream via feedback.',
        'classes': c221,
        'attn_class': 'FullMultiHeadConvAttentionIntentStream',
        'mlp_class': 'PureHydraMLP',
        'fwd_args': ', z_i',
        'fwd_logic': 'x_eff = x + self.feedback(z_i)\n        a_out, z_i_new = self.attn(self.ln1(x_eff), z_i, f_cos, f_sin)\n        x = x + a_out\n        x = x + self.mlp(self.ln2(x))',
        'fwd_returns': ', z_i_new',
        'stream_init': '        z_i = torch.zeros(B, T, D, device=x.device)',
        'block_call': 'x, z_i = block(x, f_cos, f_sin, z_i)',
        'init_extras': '        self.feedback = nn.Linear(config[\'d_model\'], config[\'d_model\'], bias=False)',
        'network_init_extras': '',
        'network_post_init': '',
        'extra_imports': ''
    }
]

for cfg in cfgs:
    cfg['model_class'] = cfg['model_name']
    fname = f"models/{cfg['model_name'].lower()}.py"
    with open(fname, 'w') as f:
        f.write(template.format(**cfg))
        
print("Generated all files")
