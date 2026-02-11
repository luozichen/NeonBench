"""Neon063: Attention-in-MLP.
Base: neon016 (Learned Intent).
Change: MLP is replaced by a Multi-Head Attention mechanism.
      The 'Feed-Forward' step becomes a second Attention step.
      - 2 Heads.
      - Each head expands 2x.
      - Uses IntentAttention logic.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class IntentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class HeadedMLP(nn.Module):
    """The 'MLP' that is actually an Attention mechanism."""
    def __init__(self, config):
        super().__init__()
        self.n_head = 2 # Fixed 2 heads as per request
        d_model = config['d_model']
        # Each head expands to 2x size
        # So effective head_dim = d_model * 2? No, 'blasts to twice the size'.
        # Assuming head_dim of THIS layer is d_model (so 2 heads = 2*d_model total internal width?)
        # Or head_dim = d_model * 2 (so total width = 4 * d_model)?
        # Let's assume total internal width is 2x d_model per head.
        self.head_dim = d_model # Each head is size of model
        
        self.internal_dim = self.n_head * self.head_dim # 2 * d_model
        
        # Q, K, V, I
        self.c_attn = nn.Linear(d_model, 4 * self.internal_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Project back down
        self.c_proj = nn.Linear(self.internal_dim, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        # Split into Q, K, V, I (each size internal_dim)
        q, k, v, intent = self.c_attn(x).split(self.internal_dim, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Apply RoPE? Does MLP have position info? 
        # User said "Q dot product K... then hadamard sigmoid intent". 
        # Standard attention implies RoPE usually. Let's include it for consistency with "Attention".
        # We need to slice freqs to match head_dim if head_dim != model_head_dim
        # freqs are (BlockSize, dim/2). 
        # Here head_dim = d_model. We need freqs for d_model.
        # But global freqs are for d_model // n_head (64).
        # We can't easily reuse global freqs if dim is different.
        # Let's skip RoPE for this "MLP" block to play it safe, or use global freqs if dimensions match?
        # Dimensions do NOT match. Global head_dim=64. This head_dim=256.
        # SKIPPING RoPE for MLP-Attention unless specifically requested. It's "internal" processing.
        # Actually, self-attention without RoPE is just content addressing. That fits "MLP" vibe.
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = torch.sigmoid(intent) * attn_out
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.internal_dim)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = HeadedMLP(config) # The Frankenstein part
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x), f_cos, f_sin) # Pass freqs, even if unused
        return x

class Neon063(nn.Module):
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
