"""Neon095: Progressive Receptive Fields.
The kernel size of the second convolution in the Hydra MLP increases with depth.
L0: k=3+3, L1: k=3+5, L2: k=3+9, L3: k=3+17.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm
from models.neon070 import IntentAttention

class ProgressiveHydraMLP(nn.Module):
    def __init__(self, config, k2):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        
        # Parallel Convs
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=False)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=k2, groups=d_model, bias=False)
        self.k2 = k2
        
        # Feature merger (d_model -> d_ff)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        
        self.w_gate = nn.Linear(d_model, d_ff, bias=False) 
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        linear_gate = F.silu(self.w_gate(x))
        
        x_t = x.transpose(1, 2)
        
        # Path 1 (k=3)
        c1 = self.conv1(F.pad(x_t, (2, 0)))
        
        # Path 2 (Progressive k)
        c2 = self.conv2(F.pad(x_t, (self.k2 - 1, 0)))
        
        # Merge (Sum features)
        conv_out = (c1 + c2).transpose(1, 2)
        
        conv_gate = torch.sigmoid(self.c_gate_proj(conv_out))
        gate = linear_gate + conv_gate
        
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        
        kernels = [3, 5, 9, 17]
        k2 = kernels[layer_idx]
        self.mlp = ProgressiveHydraMLP(config, k2=k2)
        
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon095(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config['n_layers'])])
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
