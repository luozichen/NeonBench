"""Neon233: Phantom Shift Mask Model (5M class).
Implements a 1-step quasi-encoder lookahead using a specialized mask.
The mask interleaves a 'phantom shift' on alternating rows by masking out index 0.
This allows bidirectional attention within 2x2 blocks while simulating dual-parity streams.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class PhantomShiftMHA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.block_size = config['block_size']
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Precompute Phantom Shift Mask
        # Base: Block-Causal (2x2)
        mask = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(0, self.block_size, 2):
            if i + 1 < self.block_size:
                mask[i, i+1] = 1.0 # T_n sees T_n+1 (bidirectional in block)
        
        # Phantom Shift: Mask out index 0 for all even rows
        for i in range(0, self.block_size, 2):
            mask[i, 0] = 0.0
            
        self.register_buffer("phantom_mask", mask.view(1, 1, self.block_size, self.block_size))

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        
        q = q_raw.view(B, T, self.n_head, self.head_dim)
        k = k_raw.view(B, T, self.n_head, self.head_dim)
        v = v_raw.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Apply Phantom Shift Mask
        # attn_mask in SDPA: 1.0 for keep, 0.0 for mask (or bool)
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=self.phantom_mask[:, :, :T, :T]
        )
        
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

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = PhantomShiftMHA(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon233(nn.Module):
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
        B, T = idx.shape
        x = self.token_emb(idx)
        
        for block in self.blocks:
            x = block(x, self.freqs_cos, self.freqs_sin)
            
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            # Drop-Half Loss Mask: Only calculate on the second token of every block
            flat_logits = logits.view(-1, self.config['vocab_size'])
            flat_targets = targets.view(-1)
            
            # Mask: True for tokens at indices 1, 3, 5... (second in 2x2 block)
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
            mask[0::2] = False
            
            batch_mask = mask.repeat(B)
            loss_full = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = loss_full[batch_mask].mean()
            
        return logits, loss
