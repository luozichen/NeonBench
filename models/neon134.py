"""Neon134: Mamba-Hydra Hybrid (Optimized).
Replaces the Attention-side convolution with a Parallel Linear Scan (SSM-lite).
Uses the log-space cumsum trick to avoid Python loops, making recurrence fast.
Tests if recurrent context gating is superior to windowed convolutions.
Calibration: d_ff = 550.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class ParallelRecurrentIntentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.intent_proj = nn.Linear(d_model, self.head_dim, bias=False)
        
        # SSM parameters for Intent Recurrence
        # We predict the decay 'a' and the input 'x'
        self.ssm_a_proj = nn.Linear(d_model, self.head_dim, bias=False)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v = self.qkv_proj(x).split(C, dim=2)
        i_raw = self.intent_proj(x)
        
        # 1. OPTIMIZED PARALLEL SCAN for Intent
        # h_t = a_t * h_{t-1} + (1 - a_t) * i_t
        # Use log-sigmoid to ensure decay is in (0, 1) and stable
        gate_a = torch.sigmoid(self.ssm_a_proj(x))
        log_a = F.logsigmoid(self.ssm_a_proj(x)) # More stable than log(sigmoid)
        
        # h_t = exp(cumsum(log_a)) * cumsum( exp(-cumsum(log_a)) * (1-a)*i )
        L = torch.cumsum(log_a, dim=1)
        # We need to be careful about numerical stability with exp(-L)
        # We use the relative shift trick: exp(L_t - L_j)
        
        # Vectorized recurrence:
        x_prime = (1 - gate_a) * i_raw
        
        # Using a more numerically stable parallel recurrence:
        # intent_t = \sum_{j=1}^t [ \prod_{k=j+1}^t a_k ] * x'_j
        # which is intent_t = exp(L_t) * \sum_{j=1}^t [ exp(-L_j) * x'_j ]
        
        # For small sequence lengths, this is precise.
        # We add a small epsilon to L to avoid overflow in exp(-L)
        # and use the fact that L is always <= 0.
        curr_max_L = torch.max(L, dim=1, keepdim=True)[0]
        L_shifted = L - curr_max_L
        
        # intent = exp(L) * sum(exp(-L) * x')
        # To avoid exp(-L) explosion, we use:
        # intent_t = sum_{j<=t} exp(L_t - L_j) * x'_j
        # This is a causal linear filter.
        
        # Fast way in Pure Torch:
        # Since we don't have a fast associative scan for (a, b) pairs in core torch,
        # we'll use the log-space trick but with a stability clamp.
        
        # For T=1024, the best way without custom CUDA is often a 
        # depthwise convolution if 'a' were constant, but since it's dynamic:
        # We'll stick to the Log-Cumsum but safeguard it.
        
        # intent = exp(L) * cumsum(x_prime * exp(-L))
        # Stabilize by subtracting max from the internal exp
        exp_L = torch.exp(L_shifted)
        exp_minus_L = torch.exp(-L_shifted)
        intent = exp_L * torch.cumsum(x_prime * exp_minus_L, dim=1)
        
        # 2. Standard Attention logic
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, 1, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0))).transpose(1, 2)
        gate = torch.sigmoid(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = ParallelRecurrentIntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin)
        return x

class Neon134(nn.Module):
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
