# Architecture Log

This document details the four Neon transformer architectures, their configurations, techniques, and parameter breakdowns.

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
