"""Neon230: Progressive 20M Momentum Model
Combines neon220 (Momentum logic) with neon213 (20M Scale) and Progressive Masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class ProgressiveConv1d(nn.Module):
    def __init__(self, d_model, max_k=21):
        super().__init__()
        self.max_k = max_k
        self.current_k = 1
        # ALWAYS allocate full size for torch.compile stability
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=max_k, groups=d_model, bias=False)
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[:, :, -1] = 1.0 # Identity init (last weight = 1.0)
        
    def set_k(self, k):
        self.current_k = min(k, self.max_k)

    def forward(self, x_t): 
        # x_t: [B, C, T]
        # Create a dynamic mask based on current_k
        mask = torch.zeros_like(self.conv.weight)
        mask[:, :, -self.current_k:] = 1.0
        effective_weight = self.conv.weight * mask
        
        # Causal padding based on max_k to keep graph stable
        pad = self.max_k - 1
        return F.conv1d(F.pad(x_t, (pad, 0)), effective_weight, groups=x_t.size(1))

class FullMultiHeadConvAttentionProgressiveMomentum(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_delta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(d_model))
        
        self.conv_q = ProgressiveConv1d(d_model, max_k=21)
        self.conv_k = ProgressiveConv1d(d_model, max_k=21)
        self.conv_v = ProgressiveConv1d(d_model, max_k=21)
        self.conv_i = ProgressiveConv1d(d_model, max_k=21)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, z_i, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        x_t = x.transpose(1, 2)
        q = self.conv_q(q_raw.transpose(1, 2)).transpose(1, 2)
        k = self.conv_k(k_raw.transpose(1, 2)).transpose(1, 2)
        v = self.conv_v(v_raw.transpose(1, 2)).transpose(1, 2)
        
        # Intent Stream Update with Momentum Decay
        delta = self.w_delta(x)
        alpha = torch.sigmoid(self.alpha_raw)
        z_i_new = alpha * z_i + (1.0 - alpha) * delta
        
        intent = self.conv_i(z_i_new.transpose(1, 2)).transpose(1, 2)
        
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

class ProgressiveHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.conv_gate = ProgressiveConv1d(d_model, max_k=21)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        c_gate = self.conv_gate(x.transpose(1, 2)).transpose(1, 2)
        gate = F.silu(self.c_gate_proj(c_gate))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = FullMultiHeadConvAttentionProgressiveMomentum(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = ProgressiveHydraMLP(config)

    def forward(self, x, f_cos, f_sin, z_i):
        a_out, z_i_new = self.attn(self.ln1(x), z_i, f_cos, f_sin)
        x = x + a_out
        x = x + self.mlp(self.ln2(x))
        return x, z_i_new

class Neon230(nn.Module):
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

    def set_kernel_size(self, k):
        for block in self.blocks:
            block.attn.conv_q.set_k(k)
            block.attn.conv_k.set_k(k)
            block.attn.conv_v.set_k(k)
            block.attn.conv_i.set_k(k)
            block.mlp.conv_gate.set_k(k)
