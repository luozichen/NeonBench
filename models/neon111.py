"""Neon111: Space-Aware Pure Hydra (k=5 Matrix Projector).
Instead of summing context, we flatten the window to preserve spatial position.
Formula: gate = Sigmoid(Linear_wide(Flatten(x_{t-4:t})))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm
from models.neon070 import IntentAttention

class SpaceAwarePureHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.k = 5
        
        # Wide Projector: 5 * d_model -> d_ff
        # This is the "Space-Aware" component.
        self.c_gate_proj = nn.Linear(self.k * d_model, d_ff, bias=False)
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        
        # Construct spatial context matrix by shifting
        # tok_{t-4}, tok_{t-3}, tok_{t-2}, tok_{t-1}, tok_t
        contexts = []
        for i in range(self.k):
            # i=0: current, i=1: past1, etc.
            # We pad at the beginning for causality
            shift = (self.k - 1) - i
            if shift > 0:
                c = F.pad(x, (0, 0, shift, 0))[:, :-shift, :]
            else:
                c = x
            contexts.append(c)
        
        # Flattened context: [B, T, 5 * D]
        # order: old -> new
        flat_context = torch.cat(contexts, dim=-1)
        
        # Space-aware gate
        gate = torch.sigmoid(self.c_gate_proj(flat_context))
        
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SpaceAwarePureHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon111(nn.Module):
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
