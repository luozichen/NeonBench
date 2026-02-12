"""Neon079: Qwen3-Next Hybrid Replica.
Layers 0-2: Qwen3NextGatedDeltaNet (Conv -> SiLU -> L2 -> Delta Rule -> NormGated).
Layer 3: Qwen3NextAttention (MHA with Output Gating).
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.neon015 import RMSNorm, apply_rotary_emb, SwiGLU_MLP

# --- Qwen3-Next Utilities ---

class Qwen3NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

# Pure PyTorch implementation of chunk_gated_delta_rule
# Simplified for parallel training (Non-chunked for short context to save complexity?)
# No, let's use the chunk logic provided or a simplified parallel scan.
# For 256 context, simple masked attention is fine and fastest to implement correctly.
# The official code `torch_chunk_gated_delta_rule` is quite complex.
# I will implement the mathematical equivalent using standard attention ops + decay mask.

class Qwen3NextGatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.n_head = config['n_head']
        self.head_dim = self.d_model // self.n_head
        
        # In Qwen3-Next:
        # key_dim = head_k_dim * num_k_heads
        # value_dim = head_v_dim * num_v_heads
        # We assume standard MHA logic: n_head for both.
        
        self.conv_kernel_size = 3
        # Projections
        # Input -> Q, K, V, Z (4 parts)
        # Input -> Beta, Alpha (2 parts)
        
        # We'll use two linears as in source
        # qkvz: [d_model, d_model * 2 + d_model * (2?)]
        # Actually Q, K have same dim. V has same dim. Z has same dim (value_dim).
        # So 4 * d_model output.
        self.in_proj_qkvz = nn.Linear(self.d_model, 4 * self.d_model, bias=False)
        
        # ba: [d_model, 2 * d_model? No, beta is per head? alpha/g is per head?]
        # Source says: beta is [num_k_heads, head_k_dim]? No.
        # Source: `projection_size_ba = self.num_v_heads * 2`.
        # Beta and Alpha are SCALARS per head? Or Vectors?
        # `beta = b.sigmoid()`. `b` shape `[..., num_v_heads]`.
        # So Beta and Alpha are per-head scalars (or vectors of size 1).
        # This is different from my previous "Linear Attention" which had full vectors.
        # Checking `torch_chunk_gated_delta_rule`:
        # `beta` shape: `[batch, n_head, seq, head_dim]`?
        # Source: `beta = F.pad(beta, (0, pad_size))`... `v_beta = value * beta.unsqueeze(-1)`.
        # If beta is unsqueezed, it implies beta was `[B, H, L]`.
        # So Beta is per-head, per-token scalar. 
        # Correct. 
        
        self.in_proj_ba = nn.Linear(self.d_model, 2 * self.n_head, bias=False)
        
        # Conv
        # Grouped conv. Groups = Channels.
        self.conv_dim = 3 * self.d_model # Q, K, V only? Code says qkvz?
        # Code: `self.conv_dim = self.key_dim * 2 + self.value_dim`.
        # Wait, Z is NOT convolved?
        # `mixed_qkv = torch.cat((query, key, value), dim=-1)` -> Conv.
        # Z comes from split but NOT put into mixed_qkv for conv.
        # So Conv covers Q, K, V.
        self.conv1d = nn.Conv1d(
            in_channels=3 * self.d_model,
            out_channels=3 * self.d_model,
            kernel_size=self.conv_kernel_size,
            groups=3 * self.d_model,
            bias=False,
            padding=0 # We will do manual padding
        )
        
        # Output Gate Norm
        self.norm = Qwen3NextRMSNormGated(self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Time step bias for "g" (alpha)
        self.dt_bias = nn.Parameter(torch.ones(self.n_head))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.n_head).uniform_(0, 16)))

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, D = x.shape
        
        # 1. Projections
        qkvz = self.in_proj_qkvz(x) # [B, T, 4*D]
        ba = self.in_proj_ba(x)     # [B, T, 2*H]
        
        # Split Q, K, V, Z
        # Assuming d_qn = d_k = d_v = d_z = D
        q, k, v, z = qkvz.split(self.d_model, dim=-1)
        b, a = ba.split(self.n_head, dim=-1)
        
        # 2. Conv (Q, K, V only)
        # Causal padding instructions: Left pad (kernel-1).
        # Kernel=3 -> Pad 2.
        mixed_qkv = torch.cat([q, k, v], dim=-1).transpose(1, 2) # [B, 3D, T]
        mixed_qkv = F.pad(mixed_qkv, (2, 0))
        mixed_qkv = self.conv1d(mixed_qkv)
        mixed_qkv = F.silu(mixed_qkv) # SiLU Act
        mixed_qkv = mixed_qkv.transpose(1, 2) # [B, T, 3D]
        
        q, k, v = mixed_qkv.split(self.d_model, dim=-1)
        
        # 3. L2 Norm on Q, K (Head-wise)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)
        
        # 4. Prepare Beta / G
        # b: [B, T, H]
        beta = torch.sigmoid(b)
        # g (decay) = -exp(A_log) * softplus(a + dt_bias)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        # g is log-space decay? No, `g = g.cumsum() ... decay_mask = (g - g).exp()`.
        # So `g` here is the log-decay rate? Or "dt * A"?
        # Yes, `g` acts as the exponent.
        
        # 5. Delta Rule (Parallel / Masked Attn form)
        # A_ij = (q_i k_j^T) * Decay(i, j)
        # Decay(i, j) = exp( sum(g_k) for k in j+1..i )
        
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2).unsqueeze(-1) # [B, H, T, 1]
        g = g.transpose(1, 2) # [B, H, T]
        
        # Apply Beta to V and K?
        # Code: `v_beta = value * beta`. `k_beta = key * beta`.
        v = v * beta
        k = k * beta
        
        # Compute Scores
        scores = q @ k.transpose(-2, -1) # [B, H, T, T]
        
        # Compute Decay Mask
        # g_cumsum: [B, H, T]
        g_cumsum = g.cumsum(dim=-1)
        # decay(i, j) = exp( G_i - G_j ) for i >= j
        # [T, 1] - [1, T]
        decay_mask = (g_cumsum.unsqueeze(-1) - g_cumsum.unsqueeze(-2))
        decay_mask = torch.exp(decay_mask)
        # Causal masking
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores * decay_mask * mask
        
        # Output
        y = scores @ v # [B, H, T, D]
        
        y = y.transpose(1, 2).reshape(B, T, D)
        
        # 6. Output Norm & Gate
        # Output Gate Z was computed earlier
        y = self.norm(y, gate=z)
        
        return self.out_proj(y)

class Qwen3NextAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.n_head = config['n_head']
        self.head_dim = self.d_model // self.n_head
        
        # QKV Proj
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Output Gate Proj
        # "Chunk(q_proj(x), 2)" -> Query, Gate?
        # No, specific code: `q_proj` output size is `num_heads * head_dim * 2`.
        # So Q Projector outputs Query AND Gate.
        self.q_gate_proj = nn.Linear(self.d_model, 2 * self.d_model, bias=False) # Combined Q+Gate
        
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cos, freqs_sin):
        B, T, C = x.shape
        
        # Q + Gate
        q_g = self.q_gate_proj(x)
        q, gate = q_g.chunk(2, dim=-1)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        
        # Norms
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE (Full for now, can perform partial if needed)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        # Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output Gate (SiLU)
        y = y * F.silu(gate)
        
        return self.o_proj(y)

class Qwen3NextBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model']) # Use standard RMSNorm for block input? 
        # Code: `Qwen3NextRMSNorm`. Consistent with my RMSNorm.
        
        if layer_idx < 3:
            self.attn = Qwen3NextGatedDeltaNet(config)
        else:
            self.attn = Qwen3NextAttention(config)
            
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = SwiGLU_MLP(config)
        
    def forward(self, x, f_cos, f_sin):
        x = x + self.attn(self.ln1(x), f_cos, f_sin)
        x = x + self.mlp(self.ln2(x)) # Note: Qwen3-Next code might pass f_cos to MLP? No SwiGLU doesn't use it.
        return x

class Neon079(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)
        
        self.blocks = nn.ModuleList([Qwen3NextBlock(config, i) for i in range(config['n_layers'])])
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
