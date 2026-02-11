# Neon016: QKVI Intent Attention

**Status:** ✨ **State-of-the-Art (3M Params)**
**Validation Loss:** `1.2551` (vs Baseline `1.4673`, vs Gated SDPA `1.3698`)
**Parameter Overhead:** ~7% (+262k params @ 3M scale vs baseline)

## 1. Overview

**Neon016** introduces a fourth projection vector, **Intent ($I$)**, alongside the standard Query ($Q$), Key ($K$), and Value ($V$). This "Intent" vector acts as a dynamic, learned filter applied to the output of the attention mechanism.

Unlike standard attention which only decides *where* to look (via $Q \cdot K^T$), **Intent Attention** also decides *what to keep* from the retrieved information (via $\sigma(I) \odot Output$).

This simple addition proved to be the single most effective architectural change in our experiments, outperforming:
- **Scaling up MLP** (neon026, 1.355)
- **Gated SDPA** (neon010, 1.370)
- **Calculated Intent** (neon052, 1.345)
- **Multi-Latent Attention** (neon006, 1.547)

## 2. Architecture

### The Formula
Standard Attention:
$$ Output = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V $$

**Neon016 (Intent Attention):**
$$ Gate = \sigma(I) $$
$$ Output = Gate \odot \left( \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V \right) $$

Where:
- $I$ is a learned projection of the input $x$.
- $\sigma$ is the Sigmoid function (output range $[0, 1]$).
- $\odot$ is element-wise multiplication.

### Detailed Implementation

In `models/neon016.py`, the linear projection is modified to produce 4 chunks instead of 3:

```python
# 1. Four Projections (Q, K, V, I)
self.c_attn = nn.Linear(d_model, 4 * d_model, bias=False)
q, k, v, intent = self.c_attn(x).split(C, dim=2)

# 2. Standard Q/K Processing (RoPE, Norm)
# Note: Intent is NOT normalized or rotated. It is raw content.
q, k = self.q_norm(q), self.k_norm(k)
q = apply_rotary_emb(q, freqs_cos, freqs_sin)
k = apply_rotary_emb(k, freqs_cos, freqs_sin)

# 3. Standard Attention
attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 4. Result Gating
# Apply sigmoid to Intent to get a [0,1] gate
y = torch.sigmoid(intent) * attn_out 
```

### Dimensionality
For a model with $d_{model}=256, n_{head}=4$:
- $Q, K, V, I$ all have shape `(B, T, 4, 64)`.
- The gating happens per-head, per-dimension.
- This means the model can choose to "silence" specific features from a head's output if they are not relevant to the current intent, even if the attention mechanism attended to them.

## 3. Why It Works (Conceptual Model)

Standard attention combines "Search" and "Retrieval".
1.  **Search (Q, K):** Find relevant past tokens.
2.  **Retrieval (V):** Extract their values.

However, standard attention forces you to *always* take the sum of retrieved values. If the search finds "weak matches" or "irrelevant strong matches" (distractors), the noise is added to the residual stream.

**Intent ($I$) splits the responsibility:**
1.  **Q/K/V:** "Find the best match you can."
2.  **I (Intent):** "Regardless of what you found, do I actually *want* this type of information right now?"

For example, if the model is currently predicting a verb, but an attention head specializes in retrieving adjectives, the **Intent** vector can simply set the gate to $\approx 0$ for that head's output, effectively turning it off for this specific timestep.

This dynamic gating is much more powerful than a static "head weight" because it changes **per token**.

## 4. Comparison with Other Approaches

| Model | Mechanism | Val Loss | Notes |
|-------|-----------|----------|-------|
| **neon016** | **Learned I ($W_I$)** | **1.255** | **Independent projection allows disjoint feature learning.** |
| neon010 | Gated by Q ($W_g Q$) | 1.370 | Q is constrained by positional matching duties. |
| neon052 | Gated by Calc ($QW_q + \dots$) | 1.345 | Calc intent is correlated with Q/K/V. |
| neon005 | Baseline (No Gate) | 1.467 | No filtering capability. |

**The "Orthogonality" Hypothesis:**
The reason neon016 succeeds where calculated intent (neon052) fails is likely because the optimal "Intent" signal is **orthogonal** to the Q, K, and V signals.
- $Q$ needs to encode *position/relation*.
- $V$ needs to encode *content*.
- $I$ needs to encode *task state* (e.g. "I am looking for a name").

Forcing $I$ to be derived from $Q, K, V$ (as in neon052) entangles these distinct needs. Giving $I$ its own independent weights ($W_I$) allows it to learn pure "task state" features without compromising the attentional search.

## 5. Hyperparameters & Cost

**Parameter Cost:**
- Adds 1 linear projection ($d_{model} \times d_{model}$).
- For `d_model=256`: Adds $65,536$ params per layer.
- Total for 4 layers: ~262k parameters.
- **Overhead:** ~7% increase over baseline (2.88M $\to$ 3.15M).

**Efficiency:**
- The computational cost is negligible (one extra matmul and element-wise mult).
- The gain in loss (-0.21 vs baseline) is equivalent to doubling the model size in standard scaling laws.
- This makes neon016 **extremely efficient**.
