"""Neon243: Intent only, no Conv (SplitBrain 0%)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class IntentSplitBrainMHA_243(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.block_size = config['block_size']
        
        self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        
        causal_mask = torch.tril(torch.ones(self.block_size, self.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, self.block_size, self.block_size).bool())
        
        mask_a = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(0, self.block_size, 2):
            if i + 1 < self.block_size: mask_a[i, i+1] = 1.0
        self.register_buffer("mask_a", mask_a.view(1, 1, self.block_size, self.block_size).bool())
        
        mask_b = torch.tril(torch.ones(self.block_size, self.block_size))
        for i in range(1, self.block_size - 1, 2):
            mask_b[i, i+1] = 1.0
        self.register_buffer("mask_b", mask_b.view(1, 1, self.block_size, self.block_size).bool())

    def forward(self, x, freqs_cos, freqs_sin, is_odd_stream=False):
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
        
        n_causal = 4
        y_list = []
        
        if n_causal > 0:
            q_c, k_c, v_c = q[:, :n_causal], k[:, :n_causal], v[:, :n_causal]
            y_c = F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=self.causal_mask[:, :, :T, :T])
            y_list.append(y_c)
            
        if n_causal < self.n_head:
            q_l, k_l, v_l = q[:, n_causal:], k[:, n_causal:], v[:, n_causal:]
            look_mask = self.mask_b if is_odd_stream else self.mask_a
            y_l = F.scaled_dot_product_attention(q_l, k_l, v_l, attn_mask=look_mask[:, :, :T, :T])
            y_list.append(y_l)
            
        attn_out = torch.cat(y_list, dim=1) if len(y_list) > 1 else y_list[0]
        y = torch.sigmoid(intent) * attn_out
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)

class PureHydraMLP_243(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.c_gate_proj(x))
        return self.w2(gate * self.w1(x))

class Block_243(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = IntentSplitBrainMHA_243(config)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP_243(config)
    def forward(self, x, f_cos, f_sin, is_odd_stream=False):
        x = x + self.attn(self.ln1(x), f_cos, f_sin, is_odd_stream=is_odd_stream)
        x = x + self.mlp(self.ln2(x))
        return x

class Neon243(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)

        self.blocks = nn.ModuleList([Block_243(config) for _ in range(config['n_layers'])])

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
        
        f_cos, f_sin = self.freqs_cos[:T], self.freqs_sin[:T]
        for block in self.blocks:
            x = block(x, f_cos, f_sin, is_odd_stream=is_odd_stream)
        
        logits = self.head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            flat_logits = logits.view(-1, self.config['vocab_size'])
            flat_targets = targets.view(-1)
            
            mask = torch.ones(T, device=x.device, dtype=torch.bool)
            if is_odd_stream: mask[1::2] = False 
            else: mask[0::2] = False 
            
            batch_mask = mask.repeat(B)
            loss_full = F.cross_entropy(flat_logits, flat_targets, reduction='none')
            loss = loss_full[batch_mask].mean()
            
        return logits, loss
