"""Neon159: The Clean-Room Silent Hydra.
Adds a Denoising SiLU Bottleneck to the gate of the Silent Hydra.
Gate Formula: k=3 -> SiLU -> k=3 -> Sigmoid.
Allows the attention-free model to non-linearly filter its local context.
Calibration: d_ff = 660.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm

class DenoisingGatedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Denoising Bottleneck for the Gate
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        v = self.v_proj(x)
        g_raw_t = self.g_proj(x).transpose(1, 2)
        
        # Two-step local filtering
        mid = F.silu(self.conv1(F.pad(g_raw_t, (2, 0))))
        g = self.conv2(F.pad(mid, (2, 0))).transpose(1, 2)
        
        y = torch.sigmoid(g) * v
        return self.c_proj(y)

class DenoisingHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        # Also using bottleneck in MLP for consistency
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        mid = F.silu(self.conv1(F.pad(x_t, (2, 0))))
        c = self.conv2(F.pad(mid, (2, 0))).transpose(1, 2)
        
        gate = torch.sigmoid(self.c_gate_proj(c))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.gated = DenoisingGatedBlock(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = DenoisingHydraMLP(config)
    def forward(self, x):
        x = x + self.gated(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Neon159(nn.Module):
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
