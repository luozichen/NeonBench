"""Neon173: Dual Ascending Hierarchy (Attention & MLP).
Variation of neon167.
- Attention Kernels grow by layer: L0:k=3, L1:k=5, L2:k=7, L3:k=9.
- MLP Gate Kernels grow by layer: L0:k=3, L1:k=5, L2:k=7, L3:k=9.
Calibration: 5M Class (d_model=272, d_ff=1072).
"""
import torch
import torch.nn as nn
from models.neon015 import RMSNorm
from models.neon169 import ParametricConvAttention
from models.neon171 import ParametricHydraMLP

class Block(nn.Module):
    def __init__(self, config, k):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = ParametricConvAttention(config, k)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = ParametricHydraMLP(config, k)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon173(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
             
        # Dual Hierarchical Kernels: Ascending
        ks = [3, 5, 7, 9]
        self.blocks = nn.ModuleList([Block(config, ks[i]) for i in range(config['n_layers'])])
        
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
        loss = F = torch.nn.functional.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
