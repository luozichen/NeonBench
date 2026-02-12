# NeonBench: Architectural Exploration Log

NeonBench is a repository dedicated to exploring novel transformer and recurrent architectures at the ~3M parameter scale. This log tracks every experiment, focusing on parameter efficiency and architectural breakthroughs.

## 📋 Master Model Inventory
*Parameter counts exclude embeddings to ensure absolute consistency across benchmarks.*

| Model | Params (Ex-Emb) | Technical Description |
| :--- | :--- | :--- |
| **neon001** | 2.11M | Baseline: Pre-Norm, LayerNorm, RoPE, GELU, Bias=T. |
| neon002 | 2.10M | + RMSNorm, QK-Norm, Bias=F. |
| neon003 | 1.71M | + Multi-Query Attention (MQA). |
| neon004 | 1.71M | + Shared Wide MLP (d_ff=1024 shared). |
| **neon005** | 2.62M | + SwiGLU (SiLU), RMSNorm. (Modern Baseline) |
| neon006 | 2.49M | + MLA (Multi-head Latent Attention). |
| neon007 | 2.63M | + DeltaNet (Associative Memory Recurrence). |
| neon008 | 2.63M | + L2 Normalized Unit Sphere States. |
| **⭐ neon009** | 2.89M | **QKVI Attention**: I is a Learnt Intention. |
| **⭐ neon010** | 2.64M | **Calculated Intent**: Gated SDPA (Gate derived from Q). |
| neon011 | 11.84M | Narrow & Deep (8 layers × 384 dim, 2× MLP). |
| neon012 | 15.76M | Wide & Medium (6 layers × 512 dim, 2× MLP). |
| neon013 | 8.21M | Balanced (8 layers × 320 dim, 2× MLP). |
| neon014 | 14.19M | MLP-Heavy (6 layers × 384 dim, 4× MLP expansion). |
| **neon015** | 2.89M | Result gating, raw I, raw V. Formula: $I_i \odot \Sigma_j(A_{ij} V_j)$. |
| **⭐ neon016** | 2.89M | **Result gating, σ(I), raw V.** Identical to neon9 but with Sigmoid non-linearity. |
| neon017 | 2.89M | Result gating, raw I, σ(V). |
| neon018 | 2.89M | Result gating, σ(I), σ(V). |
| neon019 | 2.89M | Source gating, raw I, raw V. Formula: $\Sigma_j A_{ij} (I_j \odot V_j)$. |
| **neon020** | 2.89M | **Source gating, σ(I), raw V.** |
| neon021 | 2.89M | Source gating, raw I, σ(V). |
| neon022 | 2.89M | Source gating, σ(I), σ(V). |
| neon023 | 5.77M | 8-layer deep variant. LayerDrop support (ERNIE 5 inspired). No drop. |
| neon024 | 5.77M | same as 23, but WITH layerdrop!! |
| neon025 | 2.89M | **Try Post-Norm**: Same as neon16 but with PostNorm (Exaone inspired). |
| neon026 | 2.89M | neon005 scaled to neon16 size via d_ff increase. (No-Intent Control). |
| **neon027** | 2.89M | neon010 (Gated SDPA) scaled to neon16 size. (Calculated-Intent Control). |
| neon028 | 2.89M | neon006 (MLA) scaled to neon16 size. |
| neon029 | 2.89M | neon001 (GPT-2) scaled to ~3M total params (inc. embeddings). |
| neon030 | 2.89M | neon002 (RMSNorm + GELU) scaled to ~3M total params. |
| neon031 | 2.62M | Calculated Intent — σ(Q ⊙ V). Zero extra params. |
| neon032 | 2.62M | Calculated Intent — σ(Q ⊙ K). |
| neon033 | 2.62M | Calculated Intent — σ(K ⊙ V). |
| neon034 | 2.62M | Calculated Intent — σ(Q ⊙ K ⊙ V). |
| neon035 | 2.62M | Calculated Intent — LayerNorm(Q + V). |
| neon036 | 2.62M | Calculated Intent — normalize(Q + K + V). |
| neon037 | 2.62M | Calculated Intent — σ(Q) ⊙ tanh(V). |
| neon038 | 2.62M | Calculated Intent — Q + σ(K ⊙ V). |
| neon039 | 2.62M | Calculated Intent — tanh(Q + K - V). |
| neon040 | 2.62M | Calculated Intent — RMSNorm(Q ⊙ V). |
| neon041 | 2.64M | Gated Calculated Intent — σ(W_g(Q ⊙ V) + b_g). Tiny learned gate. |
| neon042 | 2.64M | Gated Calculated Intent — σ(W_g(Q ⊙ K) + b_g). |
| neon043 | 2.64M | Gated Calculated Intent — σ(W_g(K ⊙ V) + b_g). |
| neon044 | 2.64M | Gated Calculated Intent — σ(W_g(Q ⊙ K ⊙ V) + b_g). |
| neon045 | 2.64M | Gated Calculated Intent — σ(W_g(Q + V) + b_g). |
| **⭐ neon046** | 2.64M | **Gated Calculated Intent** — σ(W_g(Q + K + V) + b_g). (Milestone). |
| neon047 | 2.64M | Gated Calculated Intent — σ(W_g(σ(Q) ⊙ tanh(V)) + b_g). |
| neon048 | 2.64M | Gated Calculated Intent — σ(W_g(Q + σ(K ⊙ V)) + b_g). |
| neon049 | 2.64M | Gated Calculated Intent — σ(W_g(Q + K - V) + b_g). |
| neon050 | 2.64M | Gated Calculated Intent — σ(W_hc (Q⊙V) + b) with RMSNorm pre-gate. |
| neon051 | 2.63M | Linear Combination Intent — σ(w_q Q + w_k K + w_v V + b). |
| neon052 | 2.67M | Matrix Intent — σ(Q W_q + K W_k + V W_v + b). |
| neon053 | 2.89M | QKVI Intent Attention with SiLU gating. |
| neon054 | 2.64M | Gated Calculated Intent with SiLU — σ(W_g(Q + K + V) + b_g). |
| **neon055** | 2.89M | neon046 with larger d_ff (592). (Final Calc-Intent test). |
| neon056 | 2.90M | Double-Gated (Magnitude * Direction). |
| neon057 | 2.89M | Differential Intent (Sigmoid of absolute diffs). |
| neon058 | 2.64M | Residual Intent: Output + W_i(SiLU(Q)). |
| neon059 | 2.89M | Norm-Gated: Context [QKV, norms]. |
| neon060 | 2.89M | Max-Pooled: Max(Q, K, V). |
| **⭐ neon061** | 9.72M | **Wide MLP ("Stable Winner")**: d_ff ratio approx 16x. |
| neon062 | 2.62M | MLP-Free: Double layers, no MLP. |
| neon063 | 3.94M | Attention-in-MLP: MLP replaced by 2nd Attention step. |
| neon064 | 2.76M | Hadamard Head Merge: n_head=8 merged pairwise. |
| **neon065** | 4.20M | Big Single Head: 1 Head, d_head=512. |
| neon066 | 2.89M | Fair Fight Big Head: d_head=512, d_ff reduced to match params. |
| neon067 | 2.89M | 2 Heads (Head Dim 128). |
| **neon068** | 2.89M | **8 Heads (Head Dim 32)**. (Best Multi-Head Baseline). |
| neon069 | 2.89M | 16 Heads (Head Dim 16). |
| **neon070** | 2.84M | **Hydra MLP**: Gate = Sigmoid(Attn(x)). Context-aware activation. |
| neon071 | 2.98M | Wide Hydra: d_ff=640. |
| **neon072** | 3.21M | Gated-Residual Hydra: SiLU(Linear) + Sigmoid(Attn). |
| neon073 | 2.84M | Multi-Head Hydra. |
| neon074 | 2.84M | Swish-Gated Hydra. |
| neon075 | 2.84M | Negative Hydra: Inhibitory Tanh gating. |
| neon076 | 2.83M | Light Residual Hydra: neon072 with d_model=240. |
| **⭐ neon077** | 2.82M | **Conv-Gated Hydra**: Linear + Causal Conv Gate. **Personal SOTA.** |
| neon078 | 2.86M | **Qwen3-Next Style Hybrid**: Layers 0-2 (DeltaNet), Layer 3 (Attn). |
| neon079 | 2.87M | **Qwen3-Next Hybrid Replica**: Full Gated DeltaNet components. |
| **neon080** | 2.89M | **Scaling Study (Width)**: Match neon016 via d_ff=384. |
| **⭐ neon081** | 2.87M | **Context-scaled Hydra**: Match neon016 via k=9, d_ff=378. **[NEW SOTA]** |
| **neon082** | 2.89M | **Scaling Study (Fair Hydra)**: ResHydra (neon072) with d_ff=416. |
| **neon083** | 2.87M | **Modulation Hydra**: `SiLU(Linear) * Sigmoid(Conv9)`. |
| **neon084** | 2.88M | **Dilated Hydra**: `kernel=5, dilation=4` (RF=17). |
| **neon085** | 2.89M | **Dual-Scale Hydra**: Parallel `k=3` and `k=9` gate paths. |

---

## 📊 Benchmarks

### 🧪 Benchmark: HP0 / Tok1 (1k Vocab)
*Vocabulary Size: 1,024. Embeddings (with Head): ~0.26M.*

| Model | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- |
| neon001 | 2.11M | 1.7509 | Baseline GPT-2. |
| neon002 | 2.10M | 1.7434 | RMSNorm Baseline. |
| neon003 | 1.71M | 1.8868 | MQA Baseline. |
| neon004 | 1.71M | 1.9451 | Shared MLP. |
| neon005 | 2.62M | 1.4673 | SwiGLU Baseline. |
| neon006 | 2.49M | 1.5467 | MLA Baseline. |
| neon007 | 2.63M | 3.0147 | DeltaNet (Fail). |
| neon008 | 2.63M | 6.0381 | Unit Sphere (Fail). |
| **⭐ neon009** | 2.89M | **1.3010** | **QKVI Attention**. |
| **⭐ neon010** | 2.64M | 1.3698 | **Calculated Intent**. |
| **⭐ neon016** | 2.89M | **1.2551** | **Result Gating σ(I).** |
| neon017 | 2.89M | 1.3764 | Result raw I, σ(V). |
| neon018 | 2.89M | 1.3808 | Result σ(I), σ(V). |
| neon019 | 2.89M | 1.3150 | Source raw I, raw V. |
| **neon020** | 2.89M | 1.2809 | Source Gating σ(I). |
| neon021 | 2.89M | 1.2842 | Source raw I, σ(V). |
| neon022 | 2.89M | 1.4234 | Source σ(I), σ(V). |
| neon023 | 5.77M | 0.5260 | Overfit deep. |
| neon024 | 5.77M | 1.0800 | Deep + LayerDrop. |
| neon025 | 2.89M | 1.3404 | Post-Norm Study. |
| neon026 | 2.89M | 1.3553 | No-Intent Control. |
| neon027 | 2.89M | 1.2558 | Scaled Calc-Intent. |
| neon028 | 2.89M | 1.3554 | MLA Control. |
| neon029 | 2.89M | 1.4158 | LayerNorm Baseline. |
| neon030 | 2.89M | 1.3953 | RMSNorm Baseline. |
| neon031 | 2.62M | 1.3975 | Calc σ(Q⊙V). |
| neon032 | 2.62M | 1.3854 | Calc σ(Q⊙K). |
| neon033 | 2.62M | 1.3875 | Calc σ(K⊙V). |
| neon034 | 2.62M | 1.4229 | Calc σ(Q⊙K⊙V). |
| neon035 | 2.62M | 1.3866 | Calc LN(Q+V). |
| neon036 | 2.62M | 1.4049 | Calc norm(Q+K+V). |
| neon037 | 2.62M | 1.4264 | Calc σ(Q)⊙tanh(V). |
| neon038 | 2.62M | 1.3784 | Calc Q+σ(KV). |
| neon039 | 2.62M | 1.4417 | Calc tanh(gap). |
| neon040 | 2.62M | 1.5139 | Calc RMS(Q⊙V). |
| neon041 | 2.64M | 1.3754 | Gated Calc (QV). |
| neon042 | 2.64M | 1.3774 | Gated Calc (QK). |
| neon043 | 2.64M | 1.3780 | Gated Calc (KV). |
| neon044 | 2.64M | 1.3905 | Gated Calc (QKV_prod). |
| neon045 | 2.64M | 1.3618 | Gated Calc (Q+V). |
| **⭐ neon046** | 2.64M | 1.3524 | **Gated Calc (Q+K+V)**. |
| neon047 | 2.64M | 1.3758 | Gated Calc Bounded. |
| neon048 | 2.64M | 1.3609 | Gated Calc Biased. |
| neon049 | 2.64M | 1.3594 | Gated Calc Gap. |
| neon050 | 2.64M | 1.3740 | Gated Calc + Norm. |
| neon051 | 2.63M | 1.3938 | Linear Combination. |
| neon052 | 2.67M | 1.3447 | Matrix Intent. |
| neon053 | 2.89M | 1.3129 | QKVI SiLU. |
| neon054 | 2.64M | 1.4444 | Gated Calc SiLU. |
| neon055 | 2.89M | 1.2417 | Scaled Calc Intent. |
| neon056 | 2.90M | 1.3369 | Double Gated. |
| neon057 | 2.89M | 1.3418 | Differential Intent. |
| neon058 | 2.64M | 1.4620 | Residual Additive. |
| neon059 | 2.89M | 1.2588 | Norm Gated. |
| neon060 | 2.89M | 1.3029 | Max Pooled. |

### 🧪 Benchmark: HP0 / Tok3 (2k Vocab)
*Vocabulary Size: ~2,048. Embeddings: ~0.52M.*

| Model | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- |
| **⭐ neon016** | 2.89M | 1.1610 | Learned Intent. |
| neon027 | 2.89M | 1.1683 | Gated SDPA Baseline. |
| neon055 | 2.89M | 1.1601 | Scaled Gated Calc Intent. |

### 🧪 Benchmark: HP0 / Tok4 (4k Vocab)
*Vocabulary Size: 4,096. Embeddings: ~1.05M.*

| Model | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- |
| **⭐ neon016** | 2.89M | 0.9174 | Standard Baseline. |
| neon027 | 2.89M | 0.9297 | Gated SDPA Baseline. |
| neon055 | 2.89M | 0.9434 | Gated Calc Intent. |
| **⭐ neon061** | 9.72M | **0.2364** | Wide MLP Winner. |
| neon062 | 2.62M | 1.2116 | MLP-Free Stack. |
| neon063 | 3.94M | 1.1605 | Attention-in-MLP. |
| neon064 | 2.76M | 1.2970 | Hadamard Merge. |
| neon065 | 4.20M | 0.7833 | Big Single Head. |
| neon066 | 2.89M | 1.0632 | Fair Fight Big Head. |
| neon067 | 2.89M | 1.0060 | 2 Heads. |
| **neon068** | 2.89M | **0.9214** | **8 Heads Baseline**. |
| neon069 | 2.89M | 0.9259 | 16 Heads. |
| neon070 | 2.84M | 1.1269 | Pure Hydra MLP. |
| neon071 | 2.98M | 1.0495 | Wide Hydra. |
| neon072 | 3.21M | 0.8577 | ResHydra Hybrid. |
| neon073 | 2.84M | 1.1571 | Multi-Head Hydra. |
| neon074 | 2.84M | 1.0653 | Swish Hydra. |
| neon075 | 2.84M | 1.0084 | Negative Hydra. |
| neon076 | 2.83M | 1.0399 | Light Hydra. |
| **⭐ neon077** | 2.82M | **0.9172** | **Conv-Gated Hydra**. |
| neon078 | 2.86M | 1.4483 | Hybrid DeltaNet. |
| neon079 | 2.87M | 1.1056 | Qwen3-Next Hybrid. |
| **neon080** | 2.89M | 0.8875 | Scaling Study (Width). |
| **⭐ neon081** | **2.87M** | **0.8812** | **Scaling Study (Context)**. **NEW SOTA.** |
| **neon082** | 2.89M | 0.9886 | Scaling Study (Fair Hydra). |

### 🧪 Benchmark: Wiki103 / Tok1 & Tok4
*WikiText-103 Dataset (100MB).*

| Model | Tok | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- | :--- |
| neon010 | tok1 | 2.64M | 2.5903 | Gated SDPA. |
| neon011 | tok1 | 11.84M | 2.1547 | Narrow & Deep Wiki. |
| neon012 | tok1 | 15.76M | 2.1280 | Wide & Medium Wiki. |
| neon013 | tok1 | 8.21M | 2.2307 | Balanced Wiki. |
| neon014 | tok1 | 14.19M | 2.1199 | MLP-Heavy Wiki. |
| **⭐ neon061** | tok4 | 9.72M | 3.0940 | Wide MLP Wiki. |
| neon065 | tok4 | 4.20M | 3.3171 | Big Single Head Wiki. |
| neon066 | tok4 | 2.89M | 3.3377 | Fair Fight Big Head Wiki. |
| neon063 | tok4 | 3.94M | 3.3141 | Attention-in-MLP Wiki. |
| neon062 | tok4 | 2.62M | 3.3475 | MLP-Free Wiki. |
| neon064 | tok4 | 2.76M | 3.4275 | Hadamard Merge Wiki. |

---

## 📈 Key Discovery Timeline

1.  **Intent Evolution (001-022)**: We proved that **Result Gating** (gating the attention output) is significantly better than **Source Gating** (gating before attention). σ(I) is essential.
2.  **Calculated Intent (031-055)**: We attempted to "calculate" intent from Q/K/V interactions to save parameters. `neon010` and `neon046` proved that these "calculated" signals can match full learned gating, by saving learnt intent parameters and scaling other parts of the model.
3.  **The Head Discovery (065-069)**: We found that at our ~3M scale, **1 Massive Head (512-dim)** outperforms the standard 4-head configuration, but mostly due to internal parameter scaling. Under a "Fair Fight" (`neon066`), 4 heads remained the most optimal.
4.  **Hydra Era (070-077)**: Introduced context-aware gating in the MLP. `neon077` (Conv-Gated Hydra) successfully matched the Attention baseline using a lightweight convolutional heuristic.
5.  **Scaling Breakthrough (080-081)**: Proved that context is the primary bottleneck. `neon081` (**k=9**) shattered the baseline, achieving 0.88 val loss at 3M parameters.
6.  **Modern Hybrids (078-079)**: Replicating state-of-the-art architectures like Qwen3-Next to benchmark against our simplified blocks.
