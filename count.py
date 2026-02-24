import torch
import torch.nn as nn
import sys
import os

# Mock RMSNorm and rotary for counting
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x): return x

def apply_rotary_emb(x, c, s): return x

# Minimal Block for counting
class StandardBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(d_model // n_head)
        self.k_norm = RMSNorm(d_model // n_head)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

class FusionBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(d_model // n_head)
        self.k_norm = RMSNorm(d_model // n_head)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        # 2N -> 4N -> 2N
        self.w_gate = nn.Linear(2 * d_model, 4 * d_model, bias=False)
        self.w1 = nn.Linear(2 * d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, 2 * d_model, bias=False)

class ConvBlock185(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.conv_q = nn.Conv1d(d_model, d_model, 3, groups=d_model, bias=False)
        self.conv_k = nn.Conv1d(d_model, d_model, 3, groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, 3, groups=d_model, bias=False)
        self.conv_i = nn.Conv1d(d_model, d_model, 3, groups=d_model, bias=False)
        self.q_norm = RMSNorm(d_model // n_head)
        self.k_norm = RMSNorm(d_model // n_head)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.conv9 = nn.Conv1d(d_model, d_model, 9, groups=d_model, bias=False)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

vocab_size = 50257
d_model = 272
n_head = 4
n_layers = 4

def count(blocks, name):
    embed = vocab_size * d_model
    head = d_model * vocab_size
    p = embed + head
    for b in blocks:
        p += sum(x.numel() for x in b.parameters())
    print(f"{name}: {p:,}")

# 185 baseline
b185 = [ConvBlock185(d_model, n_head, 1072) for _ in range(4)]
count(b185, "Neon185 (Target)")

# 230/232 baseline
b230 = [StandardBlock(d_model, n_head, 1166) for _ in range(4)]
count(b230, "Neon230/232 (d_ff=1166)")

# 231 fusion
b231 = [StandardBlock(d_model, n_head, 787) for _ in range(3)]
b231.append(FusionBlock(d_model, n_head, 787))
count(b231, "Neon231 (d_ff=787)")
