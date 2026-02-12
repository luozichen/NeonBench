# NeonBench: Architectural Exploration Log

NeonBench is a repository dedicated to exploring novel transformer and recurrent architectures at the ~3M parameter scale. This log tracks every experiment, focusing on parameter efficiency and architectural breakthroughs.

## 📋 Comprehensive Model Inventory
*Parameter counts exclude embeddings to allow fair comparison across tokenizers.*

| Model | Params (Ex-Emb) | Technical Description |
| :--- | :--- | :--- |
| **neon001** | 2.11M | Baseline: Pre-Norm, LayerNorm, RoPE, GELU, Bias=T. |
| neon002 | 2.10M | + RMSNorm, QK-Norm, Bias=F. |
| neon003 | 1.71M | + Multi-Query Attention (MQA). |
| neon004 | 1.71M | + Shared Wide MLP (across layers). |
| **neon005** | 2.62M | + SwiGLU (SiLU), RMSNorm. (Modern Baseline) |
| neon006 | 2.49M | + MLA (KV Compression). |
| neon007 | 2.63M | + DeltaNet (Associative Memory Recurrence). |
| neon008 | 2.63M | + L2 Normalized Unit Sphere States. |
| **neon009** | 2.89M | + MoE (4 Experts, Top-2). |
| neon010 | 2.64M | + Gated SDPA (Gate from Q). |
| neon011-014| 8M - 15M | Wiki103 Scaling variants (Deep vs Wide). |
| **neon015** | 2.89M | + QKVI Intent Attention (Direct Gating). |
| **neon016** | 2.89M | **Learned Intent**: Sigmoid Result Gate. |
| neon017-019| 2.89M | Result Gate variants (tanh, swish, square). |
| **neon020** | 2.89M | **Source Gate**: Sigmoid applied to V before attention. |
| neon021-026| 2.89M | Intent/Source Gate hybrid variants. |
| neon027 | 2.89M | Gated SDPA Baseline (Scaled). |
| neon029-030| 2.89M | Normalization Sweeps (LN vs RMS). |
| neon031-039| 2.62M | **Calculated Intent**: Gate derived from Q/K/V interactions. |
| neon040-054| 2.64M - 2.90M | **Gated Calc Intent**: σ(W(Q+K+V)+b) and pre-norm variants. |
| **neon055** | 2.89M | Gated Calculated Intent (σ(Q+K+V)). Very efficient. |
| **neon061** | 9.72M | **Wide MLP**: d_ff=2736 (~10x d_model). |
| neon062 | 2.62M | MLP-Free: 8 layers of purely attention-based mixing. |
| neon063 | 3.94M | Att-MLP: MLP replaced by 2nd attention layer. |
| neon064 | 2.76M | Hadamard-Merged Heads (8 -> 4). |
| **neon065** | **4.20M** | **Big Single Head**: 1 Head, d_head=512. Projects d_model -> 4*512. |
| neon066-069| 2.89M | Frankenstein "Head Merging" experiments. |
| neon070 | 2.84M | **Pure Hydra Gate**: MLP gated exclusively by attention context. |
| neon071 | 2.98M | Gated Linear Attention (GLA) approximation. |
| **neon072** | **3.21M** | **ResHydra MLP**: Linear Gate + Attention Gate (Residual). |
| neon073-075| 2.84M | SSM/Recurrent variants (Mamba-ish, HGRN). |
| neon076 | 2.83M | Light Residual Hydra (d_model=240). |
| **neon077** | 2.82M | **Conv-Gated Hydra**: Linear + Conv Gate. Matches Baseline. |
| neon078 | 2.86M | Hybrid DeltaNet (3:1 Delta/Attn). |
| neon079 | 2.87M | Qwen3-Next Hybrid Replica. |

---

## 📊 Benchmarks

### Benchmark: HP0 / Tok1 (1k Vocab)
*Vocabulary Size: 1,024. Embeddings (with Head): ~0.26M.*

| Model | Params (Ex-Emb) | Val Loss | Technical Summary |
| :--- | :--- | :--- | :--- |
| neon001 | 2.11M | 1.7509 | Baseline GPT-2. |
| neon005 | 2.62M | 1.4673 | SwiGLU Baseline. |
| neon009 | 2.89M | 1.3010 | MoE (4 Experts). |
| neon016 | 2.89M | 1.2551 | Result Intent Gate. |
| neon020 | 2.89M | 1.2809 | Source Intent Gate. |
| neon046 | 2.64M | 1.3524 | Gated Calc Intent (Q+K+V). |

### Benchmark: HP0 / Tok3 (2k Vocab)
*Vocabulary Size: ~2,048. Embeddings: ~0.52M.*

| Model | Params (Ex-Emb) | Val Loss | Technical Summary |
| :--- | :--- | :--- | :--- |
| neon016 | 2.89M | 1.1610 | Result Intent Gate. |
| neon027 | 2.89M | 1.1683 | Gated SDPA Baseline. |
| neon055 | 2.89M | 1.1601 | Gated Calc Intent. |

### Benchmark: HP0 / Tok4 (4k Vocab)
*Vocabulary Size: 4,096. Embeddings: ~1.05M.*

| Model | Params (Ex-Emb) | Val Loss | Technical Summary |
| :--- | :--- | :--- | :--- |
| **neon016** | 2.89M | 0.9174 | **Baseline (Intent Attention)**. |
| **neon061** | 9.72M | **0.2364** | **Wide MLP (Brute Force)**. |
| **neon065** | **4.20M** | **0.7833** | **Big Single Head (Efficiency Winner)**. |
| **neon072** | **3.21M** | **0.8577** | **ResHydra MLP (Best Hybrid)**. |
| **neon077** | 2.82M | 0.9172 | Conv-Gated Hydra (Baseline Match). |
| neon079 | 2.87M | 1.1056 | Qwen3-Next Hybrid (Underperforms). |

### Benchmark: Wiki103 / Tok1 (1k Vocab)
*WikiText-103 Dataset (100MB).*

| Model | Params (Ex-Emb) | Val Loss | Technical Summary |
| :--- | :--- | :--- | :--- |
| neon010 | 2.64M | 2.5903 | Small model baseline. |
| neon011 | 11.85M | 2.1547 | Narrow & Deep (8 layers, d=384). |
| neon012 | 15.77M | 2.1280 | Wide & Medium (6 layers, d=512). |
| neon014 | 14.12M | 2.1199 | MLP-Heavy (6 layers, d=384, d_ff=1536). |

### Benchmark: Wiki103 / Tok4 (4k Vocab)
*Note: Harder dataset with larger vocab. Typically higher losses.*

| Model | Params (Ex-Emb) | Val Loss | Technical Summary |
| :--- | :--- | :--- | :--- |
| neon061 | 9.72M | 3.0940 | Wide MLP. |
| neon065 | 4.20M | 3.3171 | Big Single Head. |
| neon068 | 2.89M | 2.89?? | Frankenstein 4-Head. |
