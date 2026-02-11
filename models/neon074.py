"""Neon074: Swish-Gated Hydra.
Gate = x * Sigmoid(Attn(x)).
Context-Aware Swish activation.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb
from models.neon070 import IntentAttention

class SwishHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.head_dim = 64
        
        self.c_gate_attn = nn.Linear(d_model, 3 * self.head_dim, bias=False)
        self.c_gate_proj = nn.Linear(self.head_dim, d_ff, bias=False) # Projects context to gate width
        
        # Note: We still project x to d_ff via w1.
        # But instead of xW_g gating, we use Context Gating.
        # Wait, Swish is x * Sigmoid(x).
        # Here we do (x W1) * Sigmoid(Context).
        # This is basically neon070.
        # Oh, if we want TRUE Swish-Gated Hydra:
        # y = (x W1) * Sigmoid(Attn(x W1?)). No Attn is on x.
        # Let's keep it consistent: Gate = Sigmoid(Attn(x)).
        # But multiply by xW1? Yes.
        # Ah, maybe I meant: Gate = Context * Sigmoid(Context)? No.
        # Let's stick to neon070 logic but maybe remove the PROJECTION?
        # No, context is 64 dim. d_ff is 576. We need projection.
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        q, k, v = self.c_gate_attn(x).split(self.head_dim, dim=2)
        f_cos = freqs_cos[..., :self.head_dim//2]
        f_sin = freqs_sin[..., :self.head_dim//2]
        q = q.view(B, T, 1, self.head_dim)
        k = k.view(B, T, 1, self.head_dim)
        v = v.view(B, T, 1, self.head_dim)
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = context.transpose(1, 2).contiguous().view(B, T, self.head_dim)
        
        # Difference from neon070:
        # neon070: Gate = Sigmoid(Proj(Context))
        # neon074: let's try Swish logic on the Context itself before projection?
        # Or: Gate = Proj(Context) * Sigmoid(Proj(Context)) ? -> Swish Gate.
        
        gate_raw = self.c_gate_proj(context)
        gate = gate_raw * torch.sigmoid(gate_raw) # Swish activation on the gate signal
        
        h = gate * self.w1(x)
        return self.w2(h)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwishHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon074(nn.Module):
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
        for block in self.blocks:
            x = block(x, self.freqs_cos, self.freqs_sin)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
