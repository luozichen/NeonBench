"""Neon070: The Hydra MLP (Context-Aware Gating).
Base: neon016 (Learned Intent).
Change: MLP Gate is replaced by a lightweight Attention Mechanism.
      - Standard SwiGLU: y = SiLU(x W_g) * (x W_1) * W_2
      - Hydra MLP: y = Sigmoid(Attn(x)) * (x W_1) * W_2
      - This allows the MLP to activate based on context, not just token identity.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

# Standard IntentAttention from neon016
class IntentAttention(nn.Module):
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

class HydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.head_dim = 64 # Small dedicated head for gating
        
        # Hydra Gate Components
        # We need Q, K, V for the gate context
        self.c_gate_attn = nn.Linear(d_model, 3 * self.head_dim, bias=False)
        # Project context to d_ff gate width
        self.c_gate_proj = nn.Linear(self.head_dim, d_ff, bias=False)
        
        # Standard MLP Components
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # Up
        self.w2 = nn.Linear(d_ff, d_model, bias=False) # Down

    def forward(self, x, freqs_cos, freqs_sin):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # 1. Compute Gate Context via Attention
        q, k, v = self.c_gate_attn(x).split(self.head_dim, dim=2)
        
        # Hydra needs RoPE too? Yes, for context awareness.
        # Use partial freqs for dim=64
        f_cos = freqs_cos[..., :self.head_dim//2]
        f_sin = freqs_sin[..., :self.head_dim//2]
        
        q = q.view(B, T, 1, self.head_dim)
        k = k.view(B, T, 1, self.head_dim)
        v = v.view(B, T, 1, self.head_dim)
        
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # context: (B, 1, T, 64)
        context = context.transpose(1, 2).contiguous().view(B, T, self.head_dim)
        
        # 2. Compute Gate Activation
        gate = torch.sigmoid(self.c_gate_proj(context)) # (B, T, d_ff)
        
        # 3. Apply Gate
        # y = Gate * (x W_1)
        h = gate * self.w1(x)
        
        # 4. Project Down
        return self.w2(h)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = HydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        # Hydra MLP needs freqs for its internal attention gate
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon070(nn.Module):
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

        # Standard Freqs
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
