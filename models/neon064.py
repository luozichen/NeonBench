"""Neon064: Hadamard Head Merge.
Base: neon016 (Learned Intent).
Change: 8 Heads. Merged pairwise via Hadamard product.
      H1 * H2 -> Merged1
      H3 * H4 -> Merged2 ...
      Concat(Merged1..4) -> Output
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb, SwiGLU_MLP

class HadamardHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = 8 # Fixed 8 heads as requested
        self.head_dim = config['d_model'] // config['n_head'] # This will be smaller if d_model is same
        # Wait, usually we want to keep head_dim consistent?
        # If we use 8 heads with standard d_model (256), head_dim becomes 32 (vs 64).
        # This seems fine.
        d_model = config['d_model']
        
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False) # Q, K, V, I
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # Projection: We have 4 merged heads outputting. 
        # Each merged head is (head_dim). So total = 4 * head_dim.
        # But 4 * head_dim = 4 * (d_model/8) = d_model / 2.
        # We need to project back to d_model.
        self.c_proj = nn.Linear(d_model // 2, d_model, bias=False)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        q, k, v, intent = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        
        # RoPE needs to match head_dim. If using global freqs (dim=64) and we are dim=32, this fails.
        # We need to slice global freqs.
        f_cos = freqs_cos[..., :self.head_dim//2]
        f_sin = freqs_sin[..., :self.head_dim//2]
        
        q = apply_rotary_emb(q, f_cos, f_sin)
        k = apply_rotary_emb(k, f_cos, f_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Result gating
        heads_out = torch.sigmoid(intent) * attn_out # (B, 8, T, D)
        
        # Hadamard Merge
        # Split into pairs: 0&1, 2&3, 4&5, 6&7
        h1 = heads_out[:, 0::2] # Evens (0, 2, 4, 6) -> (B, 4, T, D)
        h2 = heads_out[:, 1::2] # Odds (1, 3, 5, 7) -> (B, 4, T, D)
        
        merged = h1 * h2 # Hadamard product -> (B, 4, T, D)
        
        # Concat
        y = merged.transpose(1, 2).contiguous().view(B, T, -1) # (B, T, 4*D)
        
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = HadamardHeadAttention(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon064(nn.Module):
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

        # Global freqs
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
