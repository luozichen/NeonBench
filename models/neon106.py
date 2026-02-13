"""Neon106: Dual-Decision Pure Hydra (Independent c3/c9 projections).
Removes the summed convolution projection for independent decision paths.
Formula: gate = Sigmoid(Proj3(c3)) + Sigmoid(Proj9(c9))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm
from models.neon070 import IntentAttention

class DualDecisionPureHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        
        # Parallel Convs
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        
        # Separate Decision Projections
        self.c3_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.c9_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        
        # Neighborhood sensing
        c3 = self.conv3(F.pad(x_t, (2, 0))).transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        
        # Independent Gating Decisions
        gate3 = torch.sigmoid(self.c3_gate_proj(c3))
        gate9 = torch.sigmoid(self.c9_gate_proj(c9))
        
        # Context combination (Additive)
        gate = gate3 + gate9
        
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = DualDecisionPureHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon106(nn.Module):
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
