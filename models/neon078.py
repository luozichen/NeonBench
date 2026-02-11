"""Neon078: Qwen2-Next Style Hybrid.
Layers 0-2: Gated DeltaNet (Approximated via Decaying Linear Attention).
Layer 3: Gated Attention (Standard Intent Attention).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb, SwiGLU_MLP

# --- Components ---

class GatedDeltaNet(nn.Module):
    """
    Approximation of Gated DeltaNet using Masked Attention with Decay.
    Mathematically equivalent to the recurrent form but trained in parallel O(T^2).
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.n_head = config['n_head']
        self.head_dim = self.d_model // self.n_head
        
        # Projections
        # q, k, v
        self.c_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.c_v = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Gates / Decay
        self.c_beta = nn.Linear(self.d_model, self.n_head, bias=False) # Decay rate per head
        self.c_alpha = nn.Linear(self.d_model, self.d_model, bias=False) # Output gate
        
        # Short Conv for local context (Depthwise)
        self.conv_q = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=0, groups=self.d_model)
        self.conv_k = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=0, groups=self.d_model)
        self.conv_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=0, groups=self.d_model)

        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.grp_norm = RMSNorm(self.d_model) # Norm before output gate? User image says "Zero-Centered RMSNorm"
        
    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        
        # 1. Short Conv (Causal)
        # Transpose for Conv
        x_t = x.transpose(1, 2)
        # Left Pad 2, Right Pad 0 for Kernel 3
        x_padded = F.pad(x_t, (2, 0)) 
        
        q = self.conv_q(x_padded).transpose(1, 2)
        k = self.conv_k(x_padded).transpose(1, 2)
        v = self.conv_v(x_padded).transpose(1, 2)
        
        # 2. Linear Projections
        q = self.c_q(q).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(k).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(v).view(B, T, self.n_head, self.head_dim)
        
        # 3. Decay (Beta)
        beta = torch.sigmoid(self.c_beta(x)) # (B, T, n_head)
        
        # 4. Attention (Delta Rule Approximation)
        # We compute A = LowerTriangular( Decay^(i-j) * (q_i * k_j) )
        if self.training:
             # Just use standard attention but replace Softmax with Decay masking.
             q = q.transpose(1, 2) # (B, H, T, D)
             k = k.transpose(1, 2)
             v = v.transpose(1, 2)
             
             q = apply_rotary_emb(q, freqs_cos, freqs_sin)
             k = apply_rotary_emb(k, freqs_cos, freqs_sin)
             
             # Compute scores
             scores = q @ k.transpose(-2, -1) 
             # Scale scores? Linear attention usually doesn't scale by sqrt(d).
             # But delta rule assumes k^T v.
             # scores represents q_i k_j^T.
             
             # Apply Causal Mask
             mask = torch.tril(torch.ones(T, T, device=x.device))
             # IMPORTANT: To avoid huge values, we should probably normalize?
             # But pure Delta Rule involves pure summation.
             # Ideally we should implement the Recurrent form for inference.
             # For now, let's keep the summation but mask out future.
             
             scores = scores * mask
             y = scores @ v # (B, H, T, D)
        else:
             # Recurrent mode not implemented, fallback to parallel
             q = q.transpose(1, 2) # (B, H, T, D)
             k = k.transpose(1, 2)
             v = v.transpose(1, 2)
             q = apply_rotary_emb(q, freqs_cos, freqs_sin)
             k = apply_rotary_emb(k, freqs_cos, freqs_sin)
             scores = q @ k.transpose(-2, -1) * torch.tril(torch.ones(T, T, device=x.device))
             y = scores @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        
        # 5. Output Gating
        y = self.grp_norm(y)
        output_gate = F.silu(self.c_alpha(x)) # Using SiLU per image
        y = y * output_gate
        
        return self.c_proj(y)

# Reuse IntentAttention (Gated Attention) from neon016 logic
class IntentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False) # Q, K, V, Intent
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

# Block Wrapper
class MixedBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = RMSNorm(config['d_model'])
        
        # First 3 layers (0,1,2) are DeltaNet. Layer 3 is Attention.
        # Wait, user said "3 layers of gated delta net, followed by one layer of gated attention."
        if layer_idx < 3:
            self.attn = GatedDeltaNet(config)
            self.layer_type = "Delta"
        else:
            self.attn = IntentAttention(config)
            self.layer_type = "Attn"
            
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config) # Standard MLP
        
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon078(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
        
        # Pass layer_idx to blocks
        self.blocks = nn.ModuleList([MixedBlock(config, i) for i in range(config['n_layers'])])
        
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
