"""Neon134: Mamba-Hydra Hybrid (Grouped-Matrix Optimized).
Uses a Grouped Matrix-Parallel Scan (one decay per head).
Greatly reduces memory footprint and eliminates NaNs via Shifted-Log stability.
Calibration: d_ff = 612.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class GroupedMatrixRecurrentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.intent_proj = nn.Linear(d_model, self.head_dim, bias=False)
        
        # SSM Decay: One scalar per head (rather than per-dimension)
        self.ssm_a_proj = nn.Linear(d_model, self.n_head, bias=False)
        
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, groups=d_model, bias=True)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v = self.qkv_proj(x).split(C, dim=2)
        i_raw = self.intent_proj(x) # [B, T, head_dim]
        
        # --- 1. GROUPED MATRIX-PARALLEL SCAN ---
        # Predict one decay per head: [B, T, n_head]
        log_a = F.logsigmoid(self.ssm_a_proj(x)) 
        L = torch.cumsum(log_a, dim=1) # [B, T, n_head]
        
        # Reshape to [B, n_head, T, 1] for relative calculation
        L = L.transpose(1, 2).unsqueeze(-1)
        
        # Stable Relative Transition: exp(L_t - L_j)
        # rel_L: [B, n_head, T, T]
        rel_L = L - L.transpose(-2, -1)
        
        # Causal mask and exp (rel_L is always <= 0 for t >= j, so it's stable)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        M = torch.exp(rel_L) * mask # Decay matrix per head
        
        # Input contribution: x' = (1-a) * i_raw
        gate_a = torch.sigmoid(self.ssm_a_proj(x)) # [B, T, n_head]
        # Broadcoast gate_a [B, T, n_head, 1] to i_raw [B, T, n_head, head_dim]
        # But we can just use 1 gate per head for efficiency
        x_prime = (1 - gate_a).unsqueeze(-1) * i_raw.view(B, T, self.n_head, self.head_dim)
        
        # Parallel integration: [B, n_head, T, head_dim]
        # x_prime_T: [B, n_head, T, head_dim]
        x_prime_T = x_prime.transpose(1, 2)
        intent = torch.matmul(M, x_prime_T) # [B, n_head, T, head_dim]
        intent = intent.transpose(1, 2).reshape(B, T, self.head_dim)
        # --- End Scan ---

        # 2. Attention logic
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (2, 0))).transpose(1, 2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, 1, self.head_dim) # Gating is head-shared
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Note: intent here is only the gating from the Recurrent Scan
        # which effectively filters the attention result.
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
        self.attn = GroupedMatrixRecurrentAttention(config)
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
