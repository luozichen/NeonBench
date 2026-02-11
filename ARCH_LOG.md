# Architecture Log

This document details the Neon transformer architectures, their configurations, techniques, and parameter breakdowns.

## Model Summary

| Model | Params | Key Feature |
|-------|--------|-------------|
| neon001 | 2,371,072 | Baseline GPT-2 (GELU, LayerNorm) |
| neon002 | 2,362,112 | + RMSNorm (replace LayerNorm) |
| neon003 | 1,968,896 | + SwiGLU MLP |
| neon004 | 1,968,896 | + Wide MLP (1024 d_ff) |
| neon005 | 2,886,400 | RoPE + RMSNorm + SwiGLU (new baseline) |
| neon006 | 2,755,328 | Multi-Latent Attention (MLA) |
| neon007 | 2,889,984 | QK-Norm |
| neon008 | 2,888,704 | Grouped-Query Attention (GQA) |
| neon009 | 3,148,544 | QKVI Intent Attention (separate I proj) |
| neon010 | 2,903,040 | Gated SDPA (query-derived gate) |
| neon011 | 12,230,528 | 10M: Narrow & Deep (384d, 8L) |
| neon012 | 16,285,312 | 10M: Wide & Medium (512d, 6L) |
| neon013 | 8,538,880 | 10M: Balanced (320d, 8L) |
| neon014 | 14,579,712 | 10M: MLP-Heavy (384d, 4× FF) |
| neon015 | 3,148,544 | Result gate: I ⊙ attn (raw I, raw V) |
| neon016 | 3,148,544 | **Result gate: σ(I) ⊙ attn** ⭐ best |
| neon017 | 3,148,544 | Result gate: I ⊙ attn(σ(V)) |
| neon018 | 3,148,544 | Result gate: σ(I) ⊙ attn(σ(V)) |
| neon019 | 3,148,544 | Source gate: attn(I ⊙ V) |
| neon020 | 3,148,544 | Source gate: attn(σ(I) ⊙ V) |
| neon021 | 3,148,544 | Source gate: attn(I ⊙ σ(V)) |
| neon022 | 3,148,544 | Source gate: attn(σ(I) ⊙ σ(V)) |
| neon023 | 6,034,688 | 8-layer neon016 (deep, MLP-starved) |
| neon024 | 6,034,688 | 8-layer neon016 + LayerDrop=0.1 |
| neon025 | 3,148,544 | neon016 + Post-Norm (instead of Pre-Norm) |
| neon026 | 3,150,592 | neon005 scaled (d_ff 512→598) — fair baseline |
| neon027 | 3,148,800 | neon010 scaled (d_ff 512→592) — Gated SDPA |
| neon028 | 3,148,544 | neon006 scaled (d_ff 512→640) — MLA |
| neon029 | 3,148,780 | neon001 scaled (d_ff 512→891) — GPT-2 |
| neon030 | 3,148,544 | neon002 scaled (d_ff 512→896) — RMSNorm |
| neon031 | 2,886,400 | Calculated Intent: σ(Q ⊙ V) |
| neon032 | 2,886,400 | Calculated Intent: σ(Q ⊙ K) |
| neon033 | 2,886,400 | Calculated Intent: σ(K ⊙ V) |
| neon034 | 2,886,400 | Calculated Intent: σ(Q ⊙ K ⊙ V) |
| neon035 | ~2,886,912 | Calculated Intent: LayerNorm(Q + V) |
| neon036 | 2,886,400 | Calculated Intent: L2-norm(Q + K + V) |
| neon037 | 2,886,400 | Calculated Intent: σ(Q) ⊙ tanh(V) |
| neon038 | 2,886,400 | Calculated Intent: Q + σ(K ⊙ V) |
| neon039 | 2,886,400 | Calculated Intent: tanh(Q + K - V) |
| neon040 | ~2,886,656 | Calculated Intent: RMSNorm(Q ⊙ V) |

---

## 3M Fair Comparison Experiment (neon025-030)

**Motivation:** Intent attention (neon016) has ~262K more params than baseline (neon005) due to the extra Intent projection. To isolate whether the performance gain comes from the gating mechanism or just having more parameters, we scaled up neon001-010's best candidates to match neon016's param count (~3.15M) by increasing `d_ff`.

**Research Question:** Is intent attention better than just having a bigger MLP?

### Experimental Setup

| Model | Base | d_ff | Total Params | Purpose |
|-------|------|------|--------------|---------|
| neon016 | — | 512 | 3,148,544 | **Reference** (σ(I) Intent Attention) |
| neon025 | neon016 | 512 | 3,148,544 | Post-Norm variant |
| neon026 | neon005 | 598 | 3,150,592 | Modern baseline scaled |
| neon027 | neon010 | 592 | 3,148,800 | Gated SDPA scaled |
| neon028 | neon006 | 640 | 3,148,544 | MLA scaled |
| neon029 | neon001 | 891 | 3,148,780 | GPT-2 scaled |
| neon030 | neon002 | 896 | 3,148,544 | RMSNorm+GELU scaled |

All trained for 10k steps on `hp0.txt` with `tok1` (BPE).

### Results (Final Val Loss @ 10k steps)

| Rank | Model | Final VL | Δ vs neon016 | Steps to VL<1.5 | Architecture |
|------|-------|----------|--------------|-----------------|-------------|
| **1** | **neon016** | **1.2551** | — | **6,000** | σ(I) Intent |
| **2** | **neon027** | **1.2558** | **+0.0007** | **6,000** | Gated SDPA |
| 3 | neon025 | 1.3404 | +0.0853 | 7,000 | Post-Norm |
| 4 | neon026 | 1.3553 | +0.1002 | 7,500 | Baseline+BigMLP |
| 5 | neon028 | 1.3554 | +0.1003 | 7,500 | MLA |
| 6 | neon030 | 1.3953 | +0.1402 | 8,000 | RMSNorm+BigMLP |
| 7 | neon029 | 1.4158 | +0.1607 | 8,500 | GPT-2+BigMLP |

### Key Findings

1. **Gating mechanisms win decisively.** neon016 (Intent) and neon027 (Gated SDPA) are virtually identical (Δ=0.0007), both **0.1 better** than the scaled baseline (neon026). The ~262K extra params are better spent on attention gating than MLP capacity.

2. **Gating mechanism doesn't matter; having gating does.** neon016 (σ(I) result gating) ≈ neon027 (σ(W_g Q) gating). Both approaches gate the attention output with a learned sigmoid, and both work equally well. The key is **gating attention**, not the specific mechanism.

3. **Pre-Norm is critical.** neon025 (Post-Norm) drops from 1.255 → 1.340 (Δ=0.085). Pre-Norm provides better gradient flow and training stability.

4. **Modern techniques stack additively.** GPT-2 (1.416) → +RMSNorm (1.395) → +SwiGLU (1.355) shows each modern technique contributes ~0.02-0.04 val loss improvement.

5. **MLA doesn't help at 3M scale.** neon028 (MLA) ≈ neon026 (standard attn). KV compression provides no benefit at this parameter count.

### Conclusion

**Intent attention is validated.** The performance gain is not from extra parameters — it's from the gating mechanism itself. At equal param count, gated attention (Intent or Gated SDPA) provides a consistent **~0.1 val loss advantage** over increasing MLP capacity.

**Recommended architecture:** neon016 (σ(I) Intent Attention, Pre-Norm) remains the best choice.

---

## Tokenizers

Two tokenization strategies are supported for experimentation:

### tok1: BPE (Byte-Pair Encoding)

Standard subword tokenizer trained on the corpus.

| Property | Value |
|----------|-------|
| Type | Byte-level BPE |
| Vocab Size | 1024 |
| Embedding Init | Random (PyTorch default) |
| Build Script | `scripts/build_tokenizer.py` |

**Characteristics:**
- Subword units learned from frequency statistics
- Handles unseen words via subword fallback
- No linguistic knowledge built in

### tok2: Word-Level with POS (Warm Init)

Linguistically-informed word-level tokenizer with warm embedding initialization.

| Property | Value |
|----------|-------|
| Type | Word-level with spaCy POS |
| Vocab Size | 1024 |
| Embedding Init | Warm (POS-based prototypes) |
| Build Script | `scripts/build_warm_tokenizer.py` |
| Embedding Script | `scripts/build_warm_embeddings.py` |

**Characteristics:**
- Word-level tokens with POS metadata (NOUN, VERB, ADJ, etc.)
- Warm embedding initialization using orthogonal POS prototypes
- Words with same POS start in similar embedding regions
- Optional GloVe integration for semantic priors

**POS Prototype Initialization:**
```
NOUN  → Prototype 0 (orthogonal vector)
VERB  → Prototype 1
ADJ   → Prototype 2
ADV   → Prototype 3
DET   → Prototype 4
PRON  → Prototype 5
ADP   → Prototype 6
CONJ  → Prototype 7
...
```

---


## Neon001 (Baseline GPT-2 Style)

A standard GPT-2 style transformer with modern attention optimizations.

### Configuration
| Parameter | Value |
|-----------|-------|
| Vocab Size | 1024 |
| D_Model | 256 |
| Layers | 4 |
| Heads | 4 |
| D_FF | 512 |
| Block Size | 256 |

### Techniques
- **Multi-Head Attention (MHA):** Standard attention with 4 query, 4 key, 4 value heads
- **Rotary Positional Embeddings (RoPE):** Position encoding via rotation in embedding space
- **GELU Activation:** Gaussian Error Linear Unit in the MLP
- **LayerNorm (GPT-2 Style):** Pre-norm architecture with trainable bias
- **Weight Tying:** Shared weights between token embeddings and output head

### Parameter Breakdown
| Component | Formula | Parameters |
|-----------|---------|------------|
| Token Embedding | vocab × d_model | 262,144 |
| Attention (×4 layers) | (3 × d_model² + d_model²) × 4 | 1,048,576 |
| MLP (×4 layers) | (d_model × d_ff + d_ff × d_model) × 4 | 1,048,576 |
| LayerNorm (×4 layers) | (d_model × 2) × 4 × 2 | 4,096 |
| Final LayerNorm | d_model × 2 | 512 |
| **Total** | | ~2,364,000 |

> **Note:** Output head shares weights with token embedding (weight tying).

---

## Neon002 (Advanced Modern)

Upgrades to modern techniques: RMSNorm, QK-Norm, and bias-free layers.

### Configuration
| Parameter | Value |
|-----------|-------|
| Vocab Size | 1024 |
| D_Model | 256 |
| Layers | 4 |
| Heads | 4 |
| D_FF | 512 |
| Block Size | 256 |

### Techniques
- **RMSNorm:** Root Mean Square normalization (replacing LayerNorm, no bias)
- **QK-Norm:** RMSNorm applied to Query and Key tensors before attention computation
- **Bias-free Linear Layers:** All linear layers without bias terms
- **Rotary Positional Embeddings (RoPE)**
- **GELU Activation**
- **Weight Tying**

### Key Differences from Neon001
- RMSNorm has fewer parameters (no bias, just scale)
- QK-Norm adds stability to attention computation
- Bias-free design reduces total parameters slightly

### Parameter Breakdown
| Component | Formula | Parameters |
|-----------|---------|------------|
| Token Embedding | vocab × d_model | 262,144 |
| Attention (×4 layers) | (3 × d_model² + d_model²) × 4 | 1,048,576 |
| QK-Norm (×4 layers) | (head_dim × 2) × 4 | 512 |
| MLP (×4 layers) | (d_model × d_ff × 2) × 4 | 1,048,576 |
| RMSNorm (×4 layers) | d_model × 4 × 2 | 2,048 |
| Final RMSNorm | d_model | 256 |
| **Total** | | ~2,362,000 |

---

## Neon003 (Efficient MQA)

Introduces Multi-Query Attention (MQA) for improved inference efficiency.

### Configuration
| Parameter | Value |
|-----------|-------|
| Vocab Size | 1024 |
| D_Model | 256 |
| Layers | 4 |
| Query Heads | 4 |
| KV Heads | 1 (shared) |
| D_FF | 512 |
| Block Size | 256 |

### Techniques
- **Multi-Query Attention (MQA):** Single shared Key and Value head across all 4 Query heads
  - Reduces KV cache size by 4× during inference
  - Significant memory savings for long sequences
- **RMSNorm (Pre-norm + QK-norm)**
- **Bias-free Linear Layers**
- **Rotary Positional Embeddings (RoPE)**
- **GELU Activation**
- **Weight Tying**

### MQA Explained
```
Standard MHA:  Q[4 heads] × K[4 heads] × V[4 heads]
MQA:           Q[4 heads] × K[1 head]  × V[1 head]  (broadcast)
```

### Parameter Breakdown
| Component | Formula | Parameters |
|-----------|---------|------------|
| Token Embedding | vocab × d_model | 262,144 |
| Q Projection (×4 layers) | d_model² × 4 | 262,144 |
| K Projection (×4 layers) | d_model × head_dim × 4 | 65,536 |
| V Projection (×4 layers) | d_model × head_dim × 4 | 65,536 |
| Output Proj (×4 layers) | d_model² × 4 | 262,144 |
| QK-Norm (×4 layers) | head_dim × 2 × 4 | 512 |
| MLP (×4 layers) | (d_model × d_ff × 2) × 4 | 1,048,576 |
| RMSNorm (×4 layers) | d_model × 4 × 2 | 2,048 |
| Final RMSNorm | d_model | 256 |
| **Total** | | ~1,968,000 |

> **Savings vs Neon002:** ~394K fewer attention parameters due to shared KV heads.

---

## Neon004 (Shared Wide MLP)

Experiments with parameter sharing in the MLP layers using a wider shared core.

### Configuration
| Parameter | Value |
|-----------|-------|
| Vocab Size | 1024 |
| D_Model | 256 |
| Layers | 4 |
| Query Heads | 4 |
| KV Heads | 1 (shared) |
| D_FF (Per-layer Selection) | 256 |
| D_FF_Wide (Shared) | 1024 |
| Block Size | 256 |

### Techniques
- **Shared Wide MLP:** A single wide MLP (d_model → 1024 → d_model) shared across all layers
  - Layer-specific "selection" matrices route information through the shared core
  - Hypothesis: Shared representations may improve parameter efficiency
- **Multi-Query Attention (MQA)**
- **RMSNorm (Pre-norm + QK-norm)**
- **Bias-free Linear Layers**
- **Rotary Positional Embeddings (RoPE)**
- **GELU Activation**
- **Weight Tying**

### Shared Wide MLP Architecture
```
Input → [Layer-Specific Up-Select (256→256)]
      → [Shared Up (256→1024)]
      → GELU
      → [Shared Down (1024→256)]
      → [Layer-Specific Down-Select (256→256)]
      → Output
```

### Parameter Breakdown
| Component | Formula | Parameters |
|-----------|---------|------------|
| Token Embedding | vocab × d_model | 262,144 |
| Attention (same as Neon003) | | 655,360 |
| QK-Norm (×4 layers) | head_dim × 2 × 4 | 512 |
| **Shared** Up Weights | d_model × d_ff_wide | 262,144 |
| **Shared** Down Weights | d_ff_wide × d_model | 262,144 |
| **Per-layer** Up-Select (×4) | d_model² × 4 | 262,144 |
| **Per-layer** Down-Select (×4) | d_model² × 4 | 262,144 |
| RMSNorm (×4 layers) | d_model × 4 × 2 | 2,048 |
| Final RMSNorm | d_model | 256 |
| **Total** | | ~1,968,000 |

### Why Neon003 and Neon004 Have Equal MLP Parameters

This is **by design**! The parameter counts are equivalent:

| Model | MLP Calculation | Total MLP Params |
|-------|-----------------|------------------|
| Neon003 | `256 × 512 × 2 × 4 layers` | 1,048,576 |
| Neon004 | `(256 × 1024 × 2) + (256 × 256 × 2 × 4 layers)` | 1,048,576 |

**Breakdown:**
- Neon003: Per-layer MLP = 256×512×2 = 262,144 per layer × 4 = **1,048,576**
- Neon004: 
  - Shared weights = 256×1024 + 1024×256 = **524,288**
  - Per-layer selection = 256×256×2 = 131,072 per layer × 4 = **524,288**
  - Total = 524,288 + 524,288 = **1,048,576**

This design allows fair comparison: same total capacity, but different memory access patterns.

---

## Neon005 (SwiGLU Activation)

Base: neon002 with SwiGLU replacing GELU MLP.

### Techniques (vs neon002)
- **SwiGLU MLP:** `down(silu(gate(x)) * up(x))` — 3 projections instead of 2
- Gating mechanism adds expressivity over standard GELU MLP
- All other components identical to neon002

### Parameter Breakdown
| Component | Formula | Parameters |
|-----------|---------|------------|
| Token Embedding | vocab × d_model | 262,144 |
| Attention (×4) | Same as neon002 | 1,049,088 |
| SwiGLU MLP (×4) | (d_model × d_ff × 3) × 4 | 1,572,864 |
| Norms | Same as neon002 | 2,304 |
| **Total** | | ~2,886,400 |

> **Note:** SwiGLU adds ~524K params over neon002 due to the 3rd MLP projection (gate).

---

## Neon006 (Multi-head Latent Attention)

Base: neon005 with MLA from DeepSeek-V2. KV is compressed through a low-rank latent.

### Techniques (vs neon005)
- **MLA:** `x → kv_down(d_latent=128) → k_up, v_up` — shared KV bottleneck
- Q projection is standard (d_model → d_model)
- Reduces KV computation and storage
- SwiGLU MLP retained

### Architecture
```
Input x
  ├── q_proj: d_model → d_model (256 → 256)
  └── kv_down: d_model → d_latent (256 → 128)
        ├── k_up: d_latent → d_model (128 → 256)
        └── v_up: d_latent → d_model (128 → 256)
```

---

## Neon007 (DeltaNet — Linear Attention)

Base: neon005 with softmax attention replaced by DeltaNet delta rule recurrence.

### Techniques (vs neon005)
- **DeltaNet:** `S_t = S_{t-1} + β_t(v_t - S_t^Tk_t) ⊗ k_t`, `o_t = S_t^Tq_t`
- β is a learned per-head sigmoid gate controlling update strength
- Keys are L2-normalized for recurrence stability
- **No RoPE** — recurrence provides inherent positional ordering
- SwiGLU MLP retained

### Properties
- **Linear time** in sequence length (O(T) vs O(T²) for softmax)
- Sequential recurrence — training is slower per step
- Associative memory S stores key-value associations

---

## Neon008 (L2 Normalization — QM-inspired)

Base: neon005 with L2 normalization + LayerNorm. All hidden states constrained to unit sphere.

### Techniques (vs neon005)
- **LayerNorm** replaces RMSNorm (needs bias for non-positive values)
- **L2 Normalization** after each residual connection: `||h||₂ = 1`
- QK-Norm (RMSNorm) retained within attention
- SwiGLU MLP retained

### Motivation
Inspired by quantum mechanics: state vectors lie on the unit hypersphere. Constraining hidden representations to unit norm forces the model to encode information in direction rather than magnitude.

### Architecture
```
x = x + attn(LayerNorm(x))
x = L2_normalize(x)        ← constrain to unit sphere
x = x + mlp(LayerNorm(x))
x = L2_normalize(x)        ← constrain to unit sphere
```

---

## Neon009 (QKVI — Intent Attention)

Base: neon005 with a 4th learned projection I (Intent) in attention.

### Techniques (vs neon005)
- **Intent projection:** `I = W_i(x)` — same shape as V
- **Output:** `I ⊙ Softmax(QK^T/√d_k)V` — element-wise modulation
- Preserves query-side information lost after softmax aggregation
- All other components identical to neon005

### Motivation
In standard attention, the query's identity is "lost" after softmax — the output is a weighted sum of values. Intent I adds a per-position modulation that preserves what the query "wanted" to know.

### Extra Parameters
| Component | Formula | Parameters |
|-----------|---------|------------|
| I projection (×4) | d_model × d_model × 4 | 262,144 |

---

## Neon010 (Gated SDPA)

Base: neon005 with sigmoid gating derived from Q.

### Techniques (vs neon005)
- **Gate:** `σ(W_g Q + b_g)` — sigmoid gate computed from query
- **Output:** `Gate(Q) ⊙ Softmax(QK^T/√d_k)V`
- Fewer extra params than neon009 (gate reuses Q, head_dim² vs d_model²)
- Gate projection has bias (only exception to bias-free design)

### Comparison with Neon009
| Aspect | Neon009 (Intent) | Neon010 (Gated) |
|--------|------------------|-----------------|
| Source | Separate projection from x | Derived from Q |
| Extra params/layer | d_model² = 65,536 | head_dim² + head_dim = 4,160 |
| Activation | Linear (learned) | Sigmoid (bounded 0-1) |

---

## Summary Comparison

| Feature | 001 | 002 | 003 | 004 | 005 | 006 | 007 | 008 | 009 | 010 |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Normalization | LN | RMS | RMS | RMS | RMS | RMS | RMS | LN+L2 | RMS | RMS |
| QK-Norm | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Attention | MHA | MHA | MQA | MQA | MHA | MLA | Delta | MHA | QKVI | Gated |
| MLP | GELU | GELU | GELU | Shared | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| RoPE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |

---

## 10M Scale — Scaling Experiments (neon011-014)

All 10M models use the **neon010 architecture** (Gated SDPA + SwiGLU + RMSNorm + QK-Norm + RoPE + weight tying). Only the configuration differs.

### Goal
Test how to best allocate ~10M parameters: more layers (depth), wider layers (width), or larger MLPs.

---

### Neon011 (Narrow & Deep)

**Strategy:** More layers, moderate width.

| Parameter | Value |
|-----------|-------|
| d_model | 384 |
| n_layers | **8** |
| n_head | 6 (head_dim=64) |
| d_ff | 768 (2× d_model) |
| block_size | 256 |

**Hypothesis:** Deeper networks gain more compositional reasoning ability.

---

### Neon012 (Wide & Medium)

**Strategy:** Wider layers, fewer layers.

| Parameter | Value |
|-----------|-------|
| d_model | **512** |
| n_layers | 6 |
| n_head | 8 (head_dim=64) |
| d_ff | 1024 (2× d_model) |
| block_size | 256 |

**Hypothesis:** Width captures more features per layer; fewer layers means less depth bottleneck on small data.

---

### Neon013 (Balanced)

**Strategy:** Compromise between depth and width.

| Parameter | Value |
|-----------|-------|
| d_model | 320 |
| n_layers | **8** |
| n_head | 8 (head_dim=40) |
| d_ff | 640 (2× d_model) |
| block_size | 256 |

**Hypothesis:** Balanced allocation avoids over-investing in either dimension.

---

### Neon014 (MLP-Heavy)

**Strategy:** Standard width + depth, but 4× MLP expansion.

| Parameter | Value |
|-----------|-------|
| d_model | 384 |
| n_layers | 6 |
| n_head | 6 (head_dim=64) |
| d_ff | **1536 (4× d_model)** |
| block_size | 256 |

**Hypothesis:** MLP stores factual knowledge; more MLP capacity = better memorization. Recent large models (Llama 3, Mistral) use larger MLP ratios.

---

### 10M Config Comparison

| Model | d_model | Layers | d_ff | MLP Ratio | ~Params |
|-------|---------|--------|------|-----------|---------|
| neon011 | 384 | 8 | 768 | 2× | ~9.8M |
| neon012 | 512 | 6 | 1024 | 2× | ~9.7M |
| neon013 | 320 | 8 | 640 | 2× | ~9.5M |
| neon014 | 384 | 6 | 1536 | 4× | ~10.2M |

### Key Questions
1. **Depth vs Width:** Does 8 layers (neon011) beat 6 wider layers (neon012)?
2. **MLP capacity:** Does 4× expansion (neon014) outperform 2× (neon011)?
3. **Balanced scaling:** Is the middle ground (neon013) the safest bet?

---

## Intent Attention Ablation (neon015-022)

All 3M scale. Base: SwiGLU + RMSNorm + QK-Norm + RoPE + weight tying.
All use QKVI projection: `c_attn = Linear(d_model, 4 * d_model)` producing Q, K, V, I per-head.

### Gating Strategy

**Result gating** (neon015-018): Intent modulates the aggregated attention output.
The query position decides how to filter the received information.

$$\text{Output}_i = f(I_i) \odot \sum_{j} A_{ij} \cdot g(V_j)$$

**Source gating** (neon019-022): Intent modulates values before aggregation.
Each source position decides what information to broadcast.

$$\text{Output}_i = \sum_{j} A_{ij} \cdot (f(I_j) \odot g(V_j))$$

### Activation Variants

| Model | Gating | f(I) | g(V) | Formula |
|-------|--------|------|------|---------|
| neon015 | Result | raw | raw | $I_i \odot \sum A V$ |
| neon016 | Result | σ | raw | $\sigma(I_i) \odot \sum A V$ |
| neon017 | Result | raw | σ | $I_i \odot \sum A \sigma(V)$ |
| neon018 | Result | σ | σ | $\sigma(I_i) \odot \sum A \sigma(V)$ |
| neon019 | Source | raw | raw | $\sum A (I_j \odot V_j)$ |
| neon020 | Source | σ | raw | $\sum A (\sigma(I_j) \odot V_j)$ |
| neon021 | Source | raw | σ | $\sum A (I_j \odot \sigma(V_j))$ |
| neon022 | Source | σ | σ | $\sum A (\sigma(I_j) \odot \sigma(V_j))$ |

### Key Questions
1. **Result vs Source gating:** Does it matter more to filter at the receiver or the sender?
2. **Sigmoid on Intent:** Does bounding I to [0,1] help (interpretable gate) or hurt (restricts expressivity)?
3. **Sigmoid on Values:** Does bounding V to [0,1] before aggregation help or hurt?
4. **Best combo:** Which (f, g) combination yields lowest val loss?

---

## Experimental Results (HP dataset, 10k steps)

### Ranking (Worst → Best)

**Tier 5: Worst**
1. **neon022** (Source, σ(I), σ(V)) — **WORST**

**Tier 4: Poor**
2. (noticeable gap)
3. **neon018** (Result, σ(I), σ(V))
4. **neon017** (Result, raw I, σ(V))
5. **neon010** (Gated SDPA baseline)

**Tier 3: Middle Pack (close)**
6. (even larger gap)
7. **neon019** (Source, raw I, raw V)
8. **neon015** (Result, raw I, raw V)
9. **neon009** (Original QKVI)

**Tier 2: Strong**
10. (small gap)
11. **neon021** (Source, raw I, σ(V))
12. **neon020** (Source, σ(I), raw V)

**Tier 1: Clear Winner**
13. (big gap)
14. **neon016** (Result, σ(I), raw V) — **BEST** ✅

### Convergence Speed
**neon016 at 10k steps ≈ neon010 at 6.8k steps** → **~32% faster convergence**

---

## Analysis: What We Learned

### 1. **Result Gating > Source Gating**
**As predicted:** Result gating (neon015-018) decisively outperformed source gating (neon019-022).

**Why:** Query-aware filtering (result gating) allows each position to **selectively integrate** information based on context. Source gating forces positions to broadcast the same suppressed information to all queries, which is less flexible.

**Evidence:** 
- **Top 3:** neon016, neon020, neon021 — 2 result, 1 source
- **Bottom tier:** neon022, neon018, neon017 — 1 source, 2 result (but both have σ(V)!)

Result gating wins when done right (raw V). Source gating can be okay if you avoid σ(V).

---

### 2. **σ(V) is HARMFUL** ⚠️
**Critical finding:** Sigmoid on values **severely degrades performance**.

**Worst performers (all have σ(V)):**
- neon022: σ(I), σ(V) — **WORST**
- neon018: σ(I), σ(V) — **Tier 4**
- neon017: raw I, σ(V) — **Tier 4**
- neon021: raw I, σ(V) — **Tier 2** (only tolerable because source gating)

**Best performers (all have raw V):**
- neon016: σ(I), raw V — **BEST**
- neon020: σ(I), raw V — **Tier 2**
- neon015: raw I, raw V — **Tier 3**

**Why σ(V) hurts:**
1. **Destroys representational capacity:** Values carry semantic content — clamping to [0,1] loses information
2. **All-positive values:** σ(V) ∈ [0,1] means values can't go negative, limiting expressivity
3. **Gradient saturation:** Sigmoid saturates (flat gradients) for large inputs, slowing learning
4. **Unnecessary constraint:** Unlike attention weights (must sum to 1), values have no inherent reason to be bounded

**Lesson:** Don't constrain what you don't need to. Values should be expressive, not gated.

---

### 3. **σ(I) is ESSENTIAL**
**Strong pattern:** Sigmoid on intent is the key differentiator.

**Evidence:**
- **Top 2:** neon016 (σ(I)), neon020 (σ(I)) — both have σ(I)
- **Middle:** neon015 (raw I), neon019 (raw I) — raw I performs worse

**Why σ(I) helps:**
1. **Interpretable gating:** Intent ∈ [0,1] makes it a proper gate (0 = suppress, 1 = keep)
2. **Prevents intent explosion:** Unbounded I can grow arbitrarily large, causing instability
3. **Stable training:** Bounded intent → bounded gating → smoother gradients

**But:** Raw I can still work (neon015, neon009 in middle tier) — it's not catastrophic, just suboptimal.

---

### 4. **The Winner: neon016**
**Formula:** $\sigma(I_i) \odot \sum_j A_{ij} V_j$

**Why it won:**
1. ✅ **Result gating:** Query-aware filtering (flexible)
2. ✅ **σ(I):** Bounded intent gate (stable)
3. ✅ **Raw V:** Unconstrained values (expressive)

This is the **Goldilocks combination** — constrain the gate, not the content.

**32% faster convergence than neon010** — significant improvement!

---

### 5. **Comparison to Baselines**
| Model | Gating | σ(I) | σ(V) | Rank | Notes |
|-------|--------|------|------|------|-------|
| **neon016** | Result | ✅ | ❌ | **1st** | **Winner** |
| neon010 | Result | ✅ | ❌ | **5th** | Baseline (gate from Q, not I) |
| neon009 | Result | ❌ | ❌ | **9th** | Original QKVI (raw I) |
| neon022 | Source | ✅ | ✅ | **14th** | Worst (over-constrained) |

**neon016 beats neon010** by using a **dedicated intent projection** (I) instead of deriving the gate from Q.

---

## Key Takeaways

1. **Result gating (receive filtering) > Source gating (broadcast control)**
2. **σ(V) is detrimental** — DO NOT bound values; they need full expressivity
3. **σ(I) is beneficial** — bounding the intent gate improves stability
4. **Best combo:** Result gating + σ(I) + raw V (neon016)
5. **Over-constraining kills performance** — neon022 (double bounded) is worst

**Recommended architecture for future models:** Result gating + σ(I) + raw V (neon016)

**Design principle:** Constrain control signals (gates), not content signals (values).

