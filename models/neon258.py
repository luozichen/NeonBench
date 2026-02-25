"""Neon258: Stochastic Depthwise Dropout Convolution.
Phase 11: Wide Conv & Progressive Scheduling.
kernel_size=9 for both Attention Q,K,V,I and MLP.
Progressively enables deeper parts of the convolution based on training steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class StochasticConv1d(nn.Module):
    def __init__(self, d_model, kernel_size=9):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=False)
        self.kernel_size = kernel_size

    def forward(self, x_pad, step=None, max_steps=None):
        wq = self.conv.weight
        
        if self.training and step is not None and max_steps is not None:
            # Stage 1: 0-30% Pointwise (T=0)
            # Stage 2: 30-70% Progressive (0 < T < 1)
            # Stage 3: 70-100% Full Conv (T=1)
            T = (step - 0.3 * max_steps) / (0.4 * max_steps)
            T = max(0.0, min(1.0, T))
            
            if T < 1.0:
                mask = torch.zeros(self.kernel_size, device=wq.device)
                mask[-1] = 1.0 # Index 8 is the current token, always keep
                
                # History tokens (y=1 to 8). y=1 is index 7, y=8 is index 0.
                for y in range(1, self.kernel_size):
                    p_keep = min(1.0, T * ((self.kernel_size - 1.0) / y)) if T > 0 else 0.0
                    if torch.rand(1).item() < p_keep:
                        mask[self.kernel_size - 1 - y] = 1.0
                
                wq = wq * mask.view(1, 1, self.kernel_size)
                
        return F.conv1d(x_pad, wq, groups=wq.shape[0])

class ConvSplitBrainMHA_258(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.block_size = config['block_size']
        
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        
        # In-Projection convolutions (Wide context blur with stochastic dropout)
        self.conv_q = StochasticConv1d(d_model, kernel_size=9)
        self.conv_k = StochasticConv1d(d_model, kernel_size=9)
        self.conv_v = StochasticConv1d(d_model, kernel_size=9)
        self.conv_i = StochasticConv1d(d_model, kernel_size=9)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 1. Strict Causal Mask
        causal_mask = torch.tril(torch.ones(self.block_size, self.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, self.block_size, self.block_size).bool())
        
        # 2. Mask A (Even Block Boundaries)
        mask_a = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(0, self.block_size, 2):
            if i + 1 < self.block_size: mask_a[i, i+1] = 1.0
        self.register_buffer("mask_a", mask_a.view(1, 1, self.block_size, self.block_size).bool())
        
        # 3. Mask B (Odd Block Boundaries)
        mask_b = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(1, self.block_size - 1, 2):
            mask_b[i, i+1] = 1.0
        self.register_buffer("mask_b", mask_b.view(1, 1, self.block_size, self.block_size).bool())

    def forward(self, x, freqs_cos, freqs_sin, is_odd_stream=False, step=None, max_steps=None):
        B, T, C = x.shape
        q_raw, k_raw, v_raw, intent_raw = self.c_attn(x).split(C, dim=2)
        
        q = self.conv_q(F.pad(q_raw.transpose(1, 2), (8, 0)), step, max_steps).transpose(1, 2)
        k = self.conv_k(F.pad(k_raw.transpose(1, 2), (8, 0)), step, max_steps).transpose(1, 2)
        v = self.conv_v(F.pad(v_raw.transpose(1, 2), (8, 0)), step, max_steps).transpose(1, 2)
        intent = self.conv_i(F.pad(intent_raw.transpose(1, 2), (8, 0)), step, max_steps).transpose(1, 2)
        
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        intent = intent.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        intent = intent.transpose(1, 2)
        
        n_causal = 1 # 75% Lookahead (like neon241)
        y_list = []
        
        if n_causal > 0:
            q_c, k_c, v_c = q[:, :n_causal], k[:, :n_causal], v[:, :n_causal]
            y_c = F.scaled_dot_product_attention(
                q_c, k_c, v_c,
                attn_mask=self.causal_mask[:, :, :T, :T]
            )
            y_list.append(y_c)
            
        if n_causal < self.n_head:
            q_l, k_l, v_l = q[:, n_causal:], k[:, n_causal:], v[:, n_causal:]
            look_mask = self.mask_b if is_odd_stream else self.mask_a
            y_l = F.scaled_dot_product_attention(
                q_l, k_l, v_l,
                attn_mask=look_mask[:, :, :T, :T]
            )
            y_list.append(y_l)
            
        attn_out = torch.cat(y_list, dim=1) if len(y_list) > 1 else y_list[0]
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

class PureHydraMLP_258(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.conv9 = StochasticConv1d(d_model, kernel_size=9)
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, step=None, max_steps=None):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        c9 = self.conv9(F.pad(x_t, (8, 0)), step, max_steps).transpose(1, 2)
        gate = F.silu(self.c_gate_proj(c9))
        return self.w2(gate * self.w1(x))

class Block_258(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = ConvSplitBrainMHA_258(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP_258(config)

    def forward(self, x, f_cos, f_sin, is_odd_stream=False, step=None, max_steps=None):
        x = x + self.attn(self.ln1(x), f_cos, f_sin, is_odd_stream=is_odd_stream, step=step, max_steps=max_steps)
        x = x + self.mlp(self.ln2(x), step=step, max_steps=max_steps)
        return x

class Neon258(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)

        self.blocks = nn.ModuleList([Block_258(config) for _ in range(config['n_layers'])])

        self.ln_f = RMSNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_emb.weight = self.head.weight

        dim = config['d_model'] // config['n_head']
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(config['block_size']).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("freqs_cos", torch.cos(freqs))
        self.register_buffer("freqs_sin", torch.sin(freqs))

    def forward(self, idx, targets=None, is_odd_stream=False):
        B, T = idx.shape
        x = self.token_emb(idx)
        
        step = getattr(self, 'current_step', None)
        max_steps = getattr(self, 'max_steps', None)
        
        f_cos, f_sin = self.freqs_cos[:T], self.freqs_sin[:T]
        for block in self.blocks:
            x = block(x, f_cos, f_sin, is_odd_stream=is_odd_stream, step=step, max_steps=max_steps)
        
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            flat_logits = logits.view(-1, self.config['vocab_size'])
            flat_targets = targets.view(-1)
            
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
            if is_odd_stream:
                mask[1::2] = False 
            else: 
                mask[0::2] = False 
            
            batch_mask = mask.repeat(B)
            loss_full = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = loss_full[batch_mask].mean()
            
        return logits, loss
