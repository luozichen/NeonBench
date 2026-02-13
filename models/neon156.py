"""Neon156: The Spectral Silent Hydra.
Extends the Attention-Free Silent Hydra (neon143) with a Multi-Scale Spectral Gate.
Replaces the single k=9 gate with a parallel bank of k=3, 5, 9 convolutions.
This allows the attention-free model to sense multiple temporal hierarchies simultaneously.
Calibration: d_ff = 650.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm

class SpectralGatedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Parallel Spectral Bank for the Gate
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, groups=d_model, bias=False)
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        x_t = x.transpose(1, 2)
        
        v = self.v_proj(x)
        g_raw = self.g_proj(x).transpose(1, 2)
        
        # Merge spectral signals
        s3 = self.conv3(F.pad(g_raw, (2, 0)))
        s5 = self.conv5(F.pad(g_raw, (4, 0)))
        s9 = self.conv9(F.pad(g_raw, (8, 0)))
        
        gate = torch.sigmoid(s3 + s5 + s9).transpose(1, 2)
        y = gate * v
        return self.c_proj(y)

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
        gate = torch.sigmoid(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.gated = SpectralGatedBlock(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)
    def forward(self, x):
        x = x + self.gated(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Neon156(nn.Module):
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

    def forward(self, idx, targets=None):
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1)) if targets is not None else None
        return logits, loss
