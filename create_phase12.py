import os

template_str = """\"\"\"{name}: Progressive {pct}% SplitBrain Lookahead.
Phase 12: Pure Progressive Lookahead.
Architecture: Pure Transformer (No Conv, No Intent).
Progressively fades in {heads} lookahead heads over training according to a 30/40/30 schedule.
\"\"\"
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.neon015 import RMSNorm, apply_rotary_emb

class ProgressiveSplitBrainMHA(nn.Module):
    def __init__(self, config, n_target_lookahead):
        super().__init__()
        self.n_head = config['n_head']
        self.head_dim = config['d_model'] // config['n_head']
        d_model = config['d_model']
        self.block_size = config['block_size']
        self.n_target_lookahead = n_target_lookahead
        
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
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
        q_raw, k_raw, v_raw = self.c_attn(x).split(C, dim=2)
        q = q_raw.view(B, T, self.n_head, self.head_dim)
        k = k_raw.view(B, T, self.n_head, self.head_dim)
        v = v_raw.view(B, T, self.n_head, self.head_dim)
        
        q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        n_causal = self.n_head - self.n_target_lookahead
        
        # Determine Lookahead Probability based on 30/40/30 schedule
        p_look = 1.0 # Default eval behavior
        if self.training and step is not None and max_steps is not None:
            if step <= 0.3 * max_steps:
                p_look = 0.0
            elif step >= 0.7 * max_steps:
                p_look = 1.0
            else:
                p_look = (step - 0.3 * max_steps) / (0.4 * max_steps)
                
        y_list = []
        
        # Guaranteed Causal Heads
        if n_causal > 0:
            q_c, k_c, v_c = q[:, :n_causal], k[:, :n_causal], v[:, :n_causal]
            y_c = F.scaled_dot_product_attention(
                q_c, k_c, v_c,
                attn_mask=self.causal_mask[:, :, :T, :T]
            )
            y_list.append(y_c)
            
        # Progressive Lookahead Heads
        if self.n_target_lookahead > 0:
            q_l, k_l, v_l = q[:, n_causal:], k[:, n_causal:], v[:, n_causal:]
            look_mask = self.mask_b if is_odd_stream else self.mask_a
            
            # Fast path when p=1 or p=0
            if p_look >= 1.0:
                y_l = F.scaled_dot_product_attention(q_l, k_l, v_l, attn_mask=look_mask[:, :, :T, :T])
            elif p_look <= 0.0:
                y_l = F.scaled_dot_product_attention(q_l, k_l, v_l, attn_mask=self.causal_mask[:, :, :T, :T])
            else:
                # Stochastic Masking
                y_l_parts = []
                for h in range(self.n_target_lookahead):
                    mask = look_mask if torch.rand(1).item() < p_look else self.causal_mask
                    y_h = F.scaled_dot_product_attention(
                        q_l[:, h:h+1], k_l[:, h:h+1], v_l[:, h:h+1],
                        attn_mask=mask[:, :, :T, :T]
                    )
                    y_l_parts.append(y_h)
                y_l = torch.cat(y_l_parts, dim=1)
                
            y_list.append(y_l)
            
        y = torch.cat(y_list, dim=1) if len(y_list) > 1 else y_list[0]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class PureHydraMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        self.c_gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.c_gate_proj(x)) * self.w1(x))

class Block(nn.Module):
    def __init__(self, config, n_target_lookahead):
        super().__init__()
        self.ln1 = RMSNorm(config['d_model'])
        self.attn = ProgressiveSplitBrainMHA(config, n_target_lookahead)
        self.ln2 = RMSNorm(config['d_model'])
        self.mlp = PureHydraMLP(config)

    def forward(self, x, f_cos, f_sin, is_odd_stream=False, step=None, max_steps=None):
        x = x + self.attn(self.ln1(x), f_cos, f_sin, is_odd_stream=is_odd_stream, step=step, max_steps=max_steps)
        x = x + self.mlp(self.ln2(x))
        return x

class {name}(nn.Module):
    def __init__(self, config, warm_embeddings=None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        if warm_embeddings is not None:
             self.token_emb.weight.data.copy_(warm_embeddings)

        n_lookahead = {heads}
        self.blocks = nn.ModuleList([Block(config, n_lookahead) for _ in range(config['n_layers'])])

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
"""

with open("models/neon259.py", "w") as f:
    f.write(template_str.format(name="Neon259", pct="100", heads="4"))
with open("models/neon260.py", "w") as f:
    f.write(template_str.format(name="Neon260", pct="75", heads="3"))

if __name__ == "__main__":
    print("Generated neon259 and neon260 successfully.")
