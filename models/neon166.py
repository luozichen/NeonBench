"""Neon166: Deep Spectral Hydra (8 Layers).
Applies a Multi-Scale Spectral Bank (k=3, 5, 9) in every layer block.
The most computationally dense Attention-Free model in the project.
Goal: Maximum texture sensing and structural stabilization.
Calibration: d_ff = 180.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm

class SpectralBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Spectral Bank
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, groups=d_model, bias=False)
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        v = self.v_proj(x)
        g_raw = self.g_proj(x).transpose(1, 2)
        
        g3 = self.conv3(F.pad(g_raw, (2, 0)))
        g5 = self.conv5(F.pad(g_raw, (4, 0)))
        g9 = self.conv9(F.pad(g_raw, (8, 0)))
        
        g = torch.sigmoid(g3 + g5 + g9).transpose(1, 2)
        y = g * v
        return self.c_proj(y)

class PureHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model, d_ff = config['d_model'], config['d_ff']
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        c9 = self.conv9(F.pad(x.transpose(1, 2), (8, 0))).transpose(1, 2)
        gate = torch.sigmoid(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.gated = SpectralBlock(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)
    def forward(self, x):
        x = x + self.gated(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Neon166(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None: self.token_emb.weight.data.copy_(warm_embeddings)
        self.blocks = nn.ModuleList([Block(config) for _ in range(8)])
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks: x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
