"""Neon164: Pyramidal Silent Hydra (8 Layers).
Expands the receptive field per layer: k = 3, 5, 7, 9, 11, 13, 15, 17.
Tests if hierarchical window expansion builds better abstraction than uniform kernels.
Calibration: d_ff = 205.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm

class PyramidalBlock(nn.Module):
    def __init__(self, config, k):
        super().__init__()
        d_model = config['d_model']
        self.k = k
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, bias=False)
        self.conv_g = nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        v = self.conv_v(F.pad(self.v_proj(x).transpose(1, 2), (self.k-1, 0))).transpose(1, 2)
        g = self.conv_g(F.pad(self.g_proj(x).transpose(1, 2), (self.k-1, 0))).transpose(1, 2)
        y = torch.sigmoid(g) * v
        return self.c_proj(y)

class PureHydraMLP(nn.Module):
    def __init__(self, config, k):
        super().__init__()
        d_model, d_ff = config['d_model'], config['d_ff']
        self.k = k
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        c = self.conv(F.pad(x.transpose(1, 2), (self.k-1, 0))).transpose(1, 2)
        gate = torch.sigmoid(self.c_gate_proj(c))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config, k):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.gated = PyramidalBlock(config, k)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config, k)
    def forward(self, x):
        x = x + self.gated(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Neon164(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None: self.token_emb.weight.data.copy_(warm_embeddings)
        
        kernels = [3, 5, 7, 9, 11, 13, 15, 17]
        self.blocks = nn.ModuleList([Block(config, kernels[i]) for i in range(8)])
        
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks: x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
