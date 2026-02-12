"""Neon089: Dense Pyramidal Hydra (k=3, 5, 7, 9).
Expands on the success of neon085 by adding intermediate scales.
Aims to capture the transition between local syntax and global structure.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm
from models.neon070 import IntentAttention

class DensePyramidalHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        
        # Quadrant Scales
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, groups=d_model, bias=False)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size=7, groups=d_model, bias=False)
        self.conv9 = nn.Conv1d(d_model, d_model, kernel_size=9, groups=d_model, bias=False)
        
        # Feature merger (d_model -> d_ff)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        
        self.w_gate = nn.Linear(d_model, d_ff, bias=False) 
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        linear_gate = F.silu(self.w_gate(x))
        
        x_t = x.transpose(1, 2)
        
        # Parallel Paths
        c3 = self.conv3(F.pad(x_t, (2, 0)))
        c5 = self.conv5(F.pad(x_t, (4, 0)))
        c7 = self.conv7(F.pad(x_t, (6, 0)))
        c9 = self.conv9(F.pad(x_t, (8, 0)))
        
        # Dense Summation
        conv_out = (c3 + c5 + c7 + c9).transpose(1, 2)
        
        conv_gate = torch.sigmoid(self.c_gate_proj(conv_out))
        gate = linear_gate + conv_gate
        
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = DensePyramidalHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon089(nn.Module):
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
