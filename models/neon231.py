"""Neon231: Quasi-Encoder Fusion innovation model (5M class).
Base matches neon230 (5M Pure Transformer).
Block 1 (Layer 2) is a 'Fusion Block' that mixes adjacent tokens in 2N space.
Uses learned Z-token for parity shifted training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class StandardMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, mask=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # SDPA handles mask if provided, else is_causal
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class SwiGLU_MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))

class FusionSplitMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        # 2N space (544 if N=272). Calibrated for 5.0M non-embed parity.
        self.w_gate = nn.Linear(2 * d_model, 4 * d_model, bias=False)
        self.w1 = nn.Linear(2 * d_model, 4 * d_model, bias=False)
        self.w2 = nn.Linear(4 * d_model, 2 * d_model, bias=False)
    def forward(self, x):
        B, T, N = x.shape
        # Sequence length must be even
        x_fused = x.view(B, T // 2, 2 * N)
        y_fused = self.w2(F.silu(self.w_gate(x_fused)) * self.w1(x_fused))
        return y_fused.view(B, T, N)

class Block(nn.Module):
    def __init__(self, config, is_fusion=False):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = StandardMHA(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = FusionSplitMLP(config) if is_fusion else SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon231(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
        self.blocks = nn.ModuleList([Block(config, (i == 1)) for i in range(config['n_layers'])])
        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight
        self.z_token = nn.Parameter(torch.zeros(1, 1, config['d_model']))
        dim = config['d_model'] // config['n_head']
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size'] + 1).float() # pad for shift
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, idx, targets=None, is_odd_stream=False):
        B, T = idx.shape
        x = self.token_emb(idx)
        
        # Parity Switching for Fusion
        if is_odd_stream:
            # Shift the stream by prepending <Z> to create new pairs
            # This makes pairs (Z, X0), (X1, X2)... instead of (X0, X1), (X2, X3)...
            x = torch.cat([self.z_token.expand(B, -1, -1), x[:, :-1, :]], dim=1)
            if targets is not None:
                targets = torch.cat([targets.new_zeros(B, 1), targets[:, :-1]], dim=1)
        
        f_cos, f_sin = self.freqs_cos[:T], self.freqs_sin[:T]
        
        for block in self.blocks:
            x = block(x, f_cos, f_sin)
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            flat_logits = logits.view(-1, self.config['vocab_size'])
            flat_targets = targets.view(-1)
            
            # Loss Masking for Fusion-Split MLP (Pair-Causality)
            # Logit i has seen (Input i, Input i+1) via Fusion.
            # Normal task: Logit i predicts Input i+1. -> CHEAT!
            # We must mask every token that has seen its own target.
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
            
            if is_odd_stream:
                # Sequence is: Z(0), X0(1), X1(2), X2(3)...
                # Target is:  X0(0), X1(1), X2(2), X3(3)... (standard shifted targets)
                # Pair (Z, X0) at index 0,1. Logit 0 predicts X0? Cheat!
                # Pair (X1, X2) at index 2,3. Logit 2 predicts X2? Cheat!
                # Keep odd indices (1, 3, 5...)
                mask[0::2] = False
            else:
                # Sequence is: X0(0), X1(1), X2(2), X3(3)...
                # Target is:  X1(0), X2(1), X3(2), X4(3)...
                # Pair (X0, X1) at index 0,1. Logit 0 predicts X1? Cheat!
                # Pair (X2, X3) at index 2,3. Logit 2 predicts X3? Cheat!
                # Keep odd indices (1, 3, 5...)
                mask[0::2] = False
            
            batch_mask = mask.repeat(B)
            loss_full = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = loss_full[batch_mask].mean()
        return logits, loss
