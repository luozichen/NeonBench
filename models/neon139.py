"""Neon139: Sequential Kernel Expansion (Progressive Context).
Layers have progressively larger kernel sizes: k=3, 5, 7, 9.
Applies to Q, K, and Intent in Attention, and the Hydra gate in MLP.
Tests if hierarchical expansion of local context through the stack is superior.
Calibration: d_ff = 550.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class ProgressiveAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.k = 3 + (2 * layer_idx)
        
        # Projections (Multi-Query Intent Style)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.intent_proj = nn.Linear(d_model, self.head_dim, bias=False)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=True)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=True)
        self.conv_i = nn.Conv1d(self.head_dim, self.head_dim, kernel_size=self.k, groups=self.head_dim, bias=True)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v = self.qkv_proj(x).split(C, dim=2)
        i_raw = self.intent_proj(x)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        intent = self.conv_i(F.pad(i_raw.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, 1, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class ProgressiveHydraMLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.k = 3 + (2 * layer_idx)
        
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=self.k, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c = self.conv(F.pad(x_t, (self.k - 1, 0))).transpose(1, 2)
        gate = torch.sigmoid(self.c_gate_proj(c))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = ProgressiveAttention(config, layer_idx)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = ProgressiveHydraMLP(config, layer_idx)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon139(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config['n_layers'])])
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
