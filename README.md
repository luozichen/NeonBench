NeonBench is a repository dedicated to exploring novel transformer and recurrent architectures at the ~3M parameter scale. This log tracks every experiment, focusing on parameter efficiency and architectural breakthroughs.

**Total Architectures Tested**: 211  
**Total Models Trained**: 256

## 📋 Master Model Inventory
*Parameter counts exclude embeddings to ensure absolute consistency across benchmarks.*

| Model | Params (Ex-Emb) | Technical Description |
| :--- | :--- | :--- |
| **neon001** | 2.37M | Baseline: Pre-Norm, LayerNorm, RoPE, GELU, Bias=T. |
| neon002 | 2.36M | + RMSNorm, QK-Norm, Bias=F. |
| neon003 | 1.97M | + Multi-Query Attention (MQA). |
| neon004 | 1.97M | + Shared Wide MLP (d_ff=1024 shared). |
| **neon005** | 2.89M | + SwiGLU (SiLU), RMSNorm. (Modern Baseline) |
| neon006 | 2.76M | + MLA (Multi-head Latent Attention). |
| neon007 | 2.89M | + DeltaNet (Associative Memory Recurrence). |
| neon008 | 2.89M | + L2 Normalized Unit Sphere States. |
| **⭐ neon009** | 3.15M | **QKVI Attention**: I is a Learnt Intention. |
| **⭐ neon010** | 2.90M | **Calculated Intent**: Gated SDPA (Gate derived from Q). |
| neon011 | 12.23M | Narrow & Deep (8 layers × 384 dim, 2× MLP). Gated SDPA. |
| neon012 | 16.29M | Wide & Medium (6 layers × 512 dim, 2× MLP). Gated SDPA. |
| neon013 | 8.54M | Balanced (8 layers × 320 dim, 2× MLP). Gated SDPA. |
| neon014 | 14.58M | MLP-Heavy (6 layers × 384 dim, 4× MLP expansion). Gated SDPA. |
| **neon015** | 3.15M | Result gating, raw I, raw V. Formula: $I_i \odot \Sigma_j(A_{ij} V_j)$. |
| **⭐ neon016** | 3.15M | **Result gating, σ(I), raw V.** Identical to neon9 but with Sigmoid non-linearity. ([Detailed Docs](docs/neon016.md)) |
| neon017 | 3.15M | Result gating, raw I, σ(V). |
| neon018 | 3.15M | Result gating, σ(I), σ(V). |
| neon019 | 3.15M | Source gating, raw I, raw V. Formula: $\Sigma_j A_{ij} (I_j \odot V_j)$. |
| **neon020** | 3.15M | **Source gating, σ(I), raw V.** |
| neon021 | 3.15M | Source gating, raw I, σ(V). |
| neon022 | 3.15M | Source gating, σ(I), σ(V). |
| neon023 | 6.03M | 8-layer deep variant. LayerDrop support (ERNIE 5 inspired). No drop. |
| neon024 | 6.03M | same as 23, but WITH layerdrop!! |
| neon025 | 3.15M | **Try Post-Norm**: Same as neon16 but with PostNorm (Exaone inspired). |
| neon026 | 3.15M | neon005 scaled to neon16 size via d_ff increase. (No-Intent Control). |
| **neon027** | 3.15M | neon010 (Gated SDPA) scaled to neon16 size. (Calculated-Intent Control). |
| neon028 | 3.15M | neon006 (MLA) scaled to neon16 size. |
| neon029 | 3.15M | neon001 (GPT-2) scaled to ~3M total params (inc. embeddings). |
| neon030 | 3.15M | neon002 (RMSNorm + GELU) scaled to ~3M total params. |
| neon031 | 2.89M | Calculated Intent — σ(Q ⊙ V). Zero extra params. |
| neon032 | 2.89M | Calculated Intent — σ(Q ⊙ K). |
| neon033 | 2.89M | Calculated Intent — σ(K ⊙ V). |
| neon034 | 2.89M | Calculated Intent — σ(Q ⊙ K ⊙ V). |
| neon035 | 2.89M | Calculated Intent — LayerNorm(Q + V). |
| neon036 | 2.89M | Calculated Intent — normalize(Q + K + V). |
| neon037 | 2.89M | Calculated Intent — σ(Q) ⊙ tanh(V). |
| neon038 | 2.89M | Calculated Intent — Q + σ(K ⊙ V). |
| neon039 | 2.89M | Calculated Intent — tanh(Q + K - V). |
| neon040 | 2.89M | Calculated Intent — RMSNorm(Q ⊙ V). |
| neon041 | 2.90M | Gated Calculated Intent — σ(W_g(Q ⊙ V) + b_g). Tiny learned gate. |
| neon042 | 2.90M | Gated Calculated Intent — σ(W_g(Q ⊙ K) + b_g). |
| neon043 | 2.90M | Gated Calculated Intent — σ(W_g(K ⊙ V) + b_g). |
| neon044 | 2.90M | Gated Calculated Intent — σ(W_g(Q ⊙ K ⊙ V) + b_g). |
| neon045 | 2.90M | Gated Calculated Intent — σ(W_g(Q + V) + b_g). |
| **⭐ neon046** | 2.90M | **Gated Calculated Intent** — σ(W_g(Q + K + V) + b_g). (Milestone). |
| neon047 | 2.90M | Gated Calculated Intent — σ(W_g(σ(Q) ⊙ tanh(V)) + b_g). |
| neon048 | 2.90M | Gated Calculated Intent — σ(W_g(Q + σ(K ⊙ V)) + b_g). |
| neon049 | 2.90M | Gated Calculated Intent — σ(W_g(Q + K - V) + b_g). |
| neon050 | 2.90M | Gated Calculated Intent — σ(W_hc (Q⊙V) + b) with RMSNorm pre-gate. |
| neon051 | 2.89M | Linear Combination Intent — σ(w_q Q + w_k K + w_v V + b). |
| neon052 | 2.94M | Matrix Intent — σ(Q W_q + K W_k + V W_v + b). |
| neon053 | 3.15M | QKVI Intent Attention with SiLU gating. |
| neon054 | 2.90M | Gated Calculated Intent with SiLU — σ(W_g(Q + K + V) + b_g). |
| **neon055** | 3.15M | neon046 with larger d_ff (592). (Final Calc-Intent test). |
| neon056 | 3.17M | Double-Gated (Magnitude * Direction). |
| neon057 | 3.15M | Differential Intent (Sigmoid of absolute diffs). |
| neon058 | 2.90M | Residual Intent: Output + W_i(SiLU(Q)). |
| neon059 | 3.15M | Norm-Gated: Context [QKV, norms]. |
| neon060 | 3.15M | Max-Pooled: Max(Q, K, V). |
| **⭐ neon061** | 9.98M | **Wide MLP ("Stable Winner")**: d_ff ratio approx 16x. |
| neon062 | 1.57M | MLP-Free: Double layers, no MLP. |
| neon063 | 4.20M | Attention-in-MLP: MLP replaced by 2nd Attention step. |
| neon064 | 3.02M | Hadamard Head Merge: n_head=8 merged pairwise. |
| **neon065** | 4.46M | Big Single Head: 1 Head, d_head=512. |
| neon066 | 2.11M | Fair Fight Big Head: d_head=512, d_ff reduced to match params. |
| neon067 | 3.15M | 2 Heads (Head Dim 128). |
| **neon068** | 3.15M | **8 Heads (Head Dim 32)**. (Best Multi-Head Baseline). |
| neon069 | 3.15M | 16 Heads (Head Dim 16). |
| **neon070** | 3.10M | **Hydra MLP**: Gate = Sigmoid(Attn(x)). Context-aware activation. |
| neon071 | 3.25M | Wide Hydra: d_ff=640. |
| **neon072** | 3.48M | Gated-Residual Hydra: SiLU(Linear) + Sigmoid(Attn). |
| neon073 | 3.10M | Multi-Head Hydra. |
| neon074 | 3.10M | Swish-Gated Hydra. |
| neon075 | 3.10M | Negative Hydra: Inhibitory Tanh gating. |
| neon076 | 3.07M | Light Residual Hydra: neon072 with d_model=240. |
| **⭐ neon077** | 3.09M | **Conv-Gated Hydra**: Linear + Causal Conv Gate. **Personal SOTA.** |
| neon078 | 3.12M | **Qwen3-Next Style Hybrid**: Layers 0-2 (DeltaNet), Layer 3 (Attn). |
| neon079 | 3.13M | **Qwen3-Next Hybrid Replica**: Full Gated DeltaNet components. |
| **neon080** | 3.15M | **Scaling Study (Width)**: Match neon016 via d_ff=384. |
| **⭐ neon081** | 3.13M | **Context-scaled Hydra**: Match neon016 via k=9, d_ff=378. **[MILESTONE]** ([Detailed Docs](docs/neon081.md)) |
| **neon082** | 3.16M | **Scaling Study (Fair Hydra)**: ResHydra (neon072) with d_ff=416. |
| **neon083** | 3.13M | **Modulation Hydra**: `SiLU(Linear) * Sigmoid(Conv9)`. |
| **neon084** | 3.15M | **Dilated Hydra**: `kernel=5, dilation=4` (RF=17). |
| **⭐ neon085** | **3.15M** | **Dual-Scale Hydra**: Parallel `k=3` and `k=9` gate paths. **[PROJECT SOTA]** ([Detailed Docs](docs/neon085.md)) |
| **neon086** | 3.14M | **Res-Hydra**: Context gate with residual `x` connection. |
| **neon087** | 3.12M | **Pyramidal Hydra**: Triple scale `k=3, 9, 27` (RF=27). |
| **neon088** | 3.15M | **Competitive Hydra**: `Max(k3, k9)` feature selection. |
| **neon089** | 3.15M | **Dense Pyramidal Hydra**: Four parallel scales `k=3, 5, 7, 9`. |
| **neon090** | 3.15M | **Recursive Hydra Gating**: Asymmetric cascaded gate logic. |
| **neon091** | 9.98M | **10M Hydra**: Scaled `neon081` (k=9) to match `neon061`. |
| **⭐ neon092** | 9.98M | **10M Dual-Scale Hydra**: Scaled `neon085` (k=3+9) to match `neon061`. **[10M SOTA]** ([Detailed Docs](docs/neon092.md)) |
| **neon093** | 9.98M | **10M Deep Standard**: 8-layer pure Transformer baseline for scaling audit. |
| **neon094** | 9.98M | **10M Hydra-Base**: Dual-Scale Hydra MLP (k=3+9) but with **Standard Attention**. |
| **neon095** | 3.15M | **Progressive Hydra**: Kernel size increases with depth (k=3, 5, 9, 17). |
| **neon096** | 3.15M | **Heterogeneous Stack**: Alternating Dual-Scale Hydra and SwiGLU layers. |
| **neon097** | 3.15M | **Triple-Scale Hydra**: Parallel k=3, 5, and 9 gate paths. |
| **neon098** | 3.15M | **Dilated Hydra (RF=65)**: Massive reach with k=3 (dense) + k=17 (dilated, d=4). |
| **neon099** | 3.15M | **Residual Hydra**: Multiplicative residual gating logic. |
| **⭐ neon100** | **3.15M** | **Pure Hydra**: Convolutional-only gate (no linear identity path). **Project SOTA.** |
| **neon101** | 3.15M | **Progressive Specialization**: 2x SwiGLU -> 2x Dual-Scale Hydra. |
| **neon102** | 3.15M | **Sandwich Hydra**: Hydra-SwiGLU-SwiGLU-Hydra stack. |
| **neon103** | 3.15M | **Inverted Sandwich**: SwiGLU-Hydra-Hydra-SwiGLU stack. |
| **neon104** | 3.15M | **Late Bloomer Hydra**: 3x SwiGLU -> 1x Hydra (L3). |
| **neon105** | 3.15M | **Early Starter Hydra**: 1x Hydra (L0) -> 3x SwiGLU. |
| **neon106** | 3.15M | **Dual-Decision Pure Hydra**: Independent sigmoid gates for k=3 and k=9. |
| **neon107** | 3.15M | **Massive Reach Pure Hydra**: Pure architecture with k=17 dilated (RF=65). |
| **neon108** | 3.15M | **Pure Hydra Single-Scale** (k=9). |
| neon109 | 3.14M | **Pure Hydra High-Reach** (k=20). |
| **neon110** | 3.15M | **Pure Hydra Swish** (MLP-Only SOTA). |
| neon111 | 3.08M | **Space-Aware Matrix Attention** (Failed). |
| neon112 | 3.14M | **Wide MLP** / Bottleneck Gate experiment. |
| **⭐ neon113** | **3.15M** | **Conv-Attention**: Locally-aware convolution (k=3) on Q/K/V/I. |
| **⭐ neon114** | **3.15M** | **Sharp-Value Conv-Attention**: Convolves Q/K/I, keeps V sharp. |
| neon115 | 3.15M | **Multi-Head Conv-Attention** (Independent head-dim convs). |
| **⭐ neon116** | **3.15M** | **Full Multi-Head Conv-Attention**: Dual-Level Context (Attn+MLP Conv). **[PROJECT SOTA]** |
| neon117 | 3.15M | **Activated Multi-Head Conv-Attention** (SiLU post-conv). |
| neon118 | 3.16M | **L2-Norm Multi-Head Conv-Attention**. |
| neon119 | 3.16M | **Dynamic Soft-Gating**: Predicted SiLU beta for selection. |
| neon120 | 3.16M | **Activated Intent**: SiLU activation on Intent gate. |
| neon121 | 3.15M | **Context-Aware Intent Only**: Sharp Q/K/V, Convolved Intent. |
| neon122 | 3.15M | **Zero-Centered Norm**: LayerNorm-style centering on Q/K. |
| neon123 | 3.15M | **Residual Gated Attention**: Intent-controlled bypass. |
| **neon124** | **3.23M** | **Multi-Query Intent (MQI)**: Shared Intent gate across all heads. |
| neon125 | 3.17M | **Bottleneck Intent**: Low-rank linear projections. |
| neon126 | 3.15M | **Attention-Context Only**: No MLP Conv ablation. |
| neon127 | 3.15M | **Biased Attention Context**: Learnable biases in Attn Convs. |
| neon128 | 3.42M | **Gateless Context baseline**: Convolved Q/K/V, no Intent gate. |
| **neon129** | 3.15M | **Hyper-Synergy**: Full + MQI + Bias. |
| **⭐ neon130** | **3.15M** | **Sharp-V Hyper-Synergy**: MQI + Sharp V context. **[Co-SOTA]** |
| neon131 | 3.15M | **Qwen-NexT Synergy**: Adds Zero-Centered Q/K stability. |
| **neon132** | 3.14M | **Causal Spectral Hydra**: Multi-scale causal conv bank (k=3, 9, 27). |
| **neon133** | 3.15M | **Commander Head**: Low-rank dynamic weights predicted on-the-fly. |
| **neon134** | 3.15M | **Mamba-Hydra Hybrid**: Matrix-Parallel Transition Scan. |
| **neon135** | 3.16M | **Holographic Projection**: Complex-valued interference attention. |
| **neon136** | 3.16M | **Hydra MoE**: 2-Expert Mixture of Experts (Dense MoE). |
| **neon137** | 3.16M | **Hierarchical Context Stack**: Layers 0-2 Intent Attn, Layer 3 Full Conv-Attn. |
| **neon138** | 3.15M | **Strategic Colossus**: Unified k=33 gate for Intent and MLP. |
| **⭐ neon139** | 3.16M | **Sequential Kernel Expansion**: Progressive k=3→5→7→9 through layers. |
| **neon140** | 3.16M | **Parallel Spectral Heads**: Per-head k=3, 5, 7, 9 specialization. |
| **neon141** | 3.15M | **Denoising Bottleneck Hydra**: k=3→SiLU→k=3→σ gate. |
| **neon142** | 3.16M | **Global Hum Hydra**: Causal Mean Pool global bias for local gates. |
| **⭐ neon143** | 3.16M | **Silent Hydra**: Attention-Free convolutional gating. **HP0 Specialist.** |
| **neon144** | 3.15M | **Sigmoid Bottleneck Hydra**: k=3→Sigmoid→k=3→σ gate. |
| **neon145** | 3.16M | **Multi-Head Denoising Bottleneck**: MHI + Denoising gate. |
| **neon146** | 3.15M | **Multi-Head Global Hum**: MHI + Global bias signal. |
| **neon147** | 3.16M | **Multi-Head Sigmoid Bottleneck**: MHI + Bounded denoising. |
| **neon148** | 3.16M | **Asymmetric Search (Sharp-Q)**: Q sharp (k=1), K wide (k=11). |
| **neon149** | 3.15M | **Dilated Receptive Fields**: k=3, dilation=4 (9-token reach). |
| **neon150** | 3.16M | **Intent Recurrence**: Cross-layer Intent signal propagation. |
| **neon151** | 3.15M | **Inception Value**: Multi-Fidelity Content (Sharp + Gated Conv V). |
| **neon152** | 3.16M | **Multi-Head Asymmetric Search**: MHI + Sharp-Q. |
| **neon153** | 3.15M | **Multi-Head Dilated Context**: MHI + Dilated k=3, d=4. |
| **neon154** | 3.16M | **Multi-Head Intent Recurrence**: MHI + Cross-layer focus. |
| **neon155** | 3.15M | **Multi-Head Inception Value**: MHI + Multi-Fidelity Content. |
| **neon156** | 3.15M | **Spectral Silent Hydra**: Attention-Free + Multi-Scale bank (k=3,5,9). |
| **neon157** | 3.16M | **Wide-Merge Silent Hydra**: Merged V+G projections. |
| **neon158** | 3.16M | **Dilated Silent Hydra**: Attention-Free + Dilated k=3, d=4. |
| **neon159** | 3.15M | **Clean-Room Silent Hydra**: Denoising SiLU Bottleneck gate. |
| **⭐ neon160** | 3.16M | **The Ghost**: 3x Silent Hydra + 1x Softmax Attention (L3). |
| **neon161** | 3.15M | **Deep Silent Hydra**: 8-layer Attention-Free. |
| **neon162** | 3.15M | **Deep Hybrid Ghost**: 7x Silent + 1x Attention (L7). 8 layers. |
| **neon163** | 3.15M | **Alternating Ghost**: Silent→Attn→Silent→Attn (8 layers). |
| **neon164** | 3.15M | **Pyramidal Silent Hydra**: 8 layers, k=3→17. |
| **neon165** | 3.15M | **Res-Gated Silent Hydra**: 8 layers, gate=σ(conv(x)+x). |
| **neon166** | 3.15M | **Deep Spectral Hydra**: 8 layers, Multi-Scale bank (k=3,5,9). |
| | | |
| **---** | **---** | **5M PARAMETER CLASS MODELS** |
| **neon167** | **5.28M** | **Giant Synergy**: Scaled neon116. (d_model=272, d_ff=1072). |
| **neon168** | 5.28M | **Sharp Intent & Value Giant**: Q/K blurred, V/I sharp. |
| **⭐ neon169** | **5.30M** | **Ascending Giant**: Hierarchical Attn kernels (k=3 to 9). |
| **neon170** | 5.30M | **Descending Giant**: Hierarchical Attn kernels (k=9 to 3). |
| **neon171** | **5.28M** | **Ascending MLP Giant**: Hierarchical MLP kernels (k=3 to 9). |
| **neon172** | 5.28M | **Descending MLP Giant**: Hierarchical MLP kernels (k=9 to 3). |
| **neon173** | 5.29M | **Dual Ascending Giant**: Hierarchical Attn + MLP kernels. |
| **neon174** | **5.29M** | **MQI Att-Hierarchy**: Shared Intent + Attn Hierarchy (d_ff=1140). |
| **neon175** | 5.28M | **MQI MLP-Hierarchy**: Shared Intent + MLP Hierarchy (d_ff=1140). |
| **neon176** | **5.29M** | **MQI Dual-Hierarchy**: Shared Intent + Dual Hierarchy (d_ff=1140). |
| **neon177** | 5.28M | **MQA Giant**: 5-Layer Multi-Query Attention attempt. |
| **neon178** | 5.28M | **Spectral Synergy**: Multi-scale Spectral Pyramid heads. |
| **neon179** | 5.28M | **Sharp Intent**: Blurred Q/K/V with Sharp Intent gate. |
| **⭐ neon180** | **5.28M** | **Sharp-V Giant**: Sharp Value with Blurred Q/K/I. **Wiki CO-SOTA.** |
| **neon181** | 5.28M | **Sharp Search**: Sharp Q/K with Blurred Value/Intent. |
| **neon182** | 5.27M | **Pure Attention**: No convolutions in projections. |
| **neon183** | 5.28M | **RoPE Before Conv**: RoPE applied before convolution step. |
| **neon184** | 5.28M | **No RoPE**: Positional info from convolutions only. |
| **⭐ neon185** | **5.28M** | **SwiGLU-Conv**: SiLU Gated MLP + Sigmoid Gated Attn. **[PROJECT SOTA]** |
| **neon186** | 5.28M | **SiLU Gated Attn**: SiLU Gated Attn + Sigmoid Gated MLP. |
| **neon187** | 5.28M | **Full SiLU Architecture**: SiLU Gated Attn + SiLU Gated MLP. |
| | | |
| **---** | **---** | **SEMANTIC GATE EXPLORATION (188-206)** |
| **neon188** | 5.28M | **Dual-Attention Bipolar Gate**: QKV tanh(5x)+x/10 mult gate + QKVI add residual. |
| **neon189** | 5.28M | **Pure Tanh Gate**: `x * tanh(QKV_attn(x))`. Semantic axis reversal. |
| **neon190** | 5.28M | **Signed Residual**: `x + tanh(QKV_attn(x)) * QKVI_attn(x)`. Polarity modulation. |
| **neon191** | 5.28M | **Bottleneck MLP Gate**: `x * tanh(W2·SiLU(W1·x))`. Lightweight d→d/4→d gate. |
| **neon192** | 5.28M | **Soft Flip + Skip**: `α·x + (1-α)·(x·tanh(QKV_attn(x)))`. Learned α. |
| **neon193** | 5.28M | **Residual Flip**: Single QKVI attn branches into gate + residual. |
| **neon194** | 5.28M | **Gated Polarity Residual**: `res * (α·tanh(proj(res)) + (1-α))`. |
| **neon195** | 5.00M | **Bipolar Intent**: `tanh(I)` replaces `sigmoid(I)`. Range [-1, +1]. |
| **neon196** | 5.00M | **Sparse Bipolar**: `tanh³(I)` — sparse polarity gating. |
| **neon197** | 5.00M | **Gated Polarity Residual (5M)**: Fair-comparison version of neon194. |
| **⭐ neon198** | **5.00M** | **Shifted Tanh Intent**: `tanh(I) + 1`. Range [0, 2], init=1. |
| **neon199** | 5.00M | **Biased Bipolar Intent**: `tanh(I + 0.55)`. Init ≈ 0.5, range [-1, +1]. |
| **neon200** | 5.00M | **Split Intent**: Half sigmoid + half tanh per head. |
| **neon201** | 5.00M | **Learnable Blend**: `α·sigmoid(I) + (1-α)·tanh(I)`, per-head α. |
| **neon202** | 5.00M | **Silent Hydra 5M**: Attention-free conv gating at 5M scale. |
| **neon203** | 5.00M | **Shifted Tanh²**: `(tanh(I)+1)²`. Range [0, 4], init=1. |
| **neon204** | 5.00M | **SiLU Intent**: `SiLU(I)` as attention gate. Unbounded amplification. |
| **neon205** | 5.00M | **Double Gate**: `sigmoid(I) · SiLU(I)`. Fine-grained control. |
| **neon206** | 5.00M | **Standard SwiGLU MLP**: No conv in MLP — tests Hydra conv value. |
| **neon207** | 5.01M | **Texture MLP (k=3)**: Conv on MLP value path too. Both branches see local context. |
| **neon208** | 5.01M | **Texture MLP (k=9)**: Symmetric conv9 on both gate and value MLP paths. |
| **neon209** | 5.00M | **Recursive Intent**: Intent passed from Layer L to L+1. Persistent search thread. |
| **neon210** | 5.00M | **Differential Attention**: `softmax(Q1K1) - λ·softmax(Q2K2)`. Noise cancellation. |
| **neon211** | 5.00M | **Reflective Attention**: Post-gate bottleneck. Gate from attn output, not input. d=280. |

---

## 📊 Benchmarks

### 🧪 Benchmark: HP0 / Tok1 (1k Vocab)
*Vocabulary Size: 1,024. Embeddings (with Head): ~0.26M.*

| Model | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- |
| **neon023** | 5.77M | **0.5260** | Overfit deep. |
| **neon024** | 5.77M | **1.0800** | Deep + LayerDrop. |
| **⭐ neon055** | 2.89M | **1.2417** | Scaled Calc Intent. |
| **⭐ neon016** | 2.89M | **1.2551** | **Result Gating σ(I).** |
| **neon027** | 2.89M | 1.2558 | Scaled Calc-Intent. |
| **neon059** | 2.89M | 1.2588 | Norm Gated. |
| **⭐ neon020** | 2.89M | **1.2809** | **Source Gating σ(I).** |
| **⭐ neon046** | 2.64M | **1.3524** | **Gated Calc (Q+K+V)**. |
| **⭐ neon009** | 2.89M | **1.3010** | **QKVI Attention**. |
| **neon060** | 3.15M | 1.3029 | Max Pooled. |
| **⭐ neon015** | 2.89M | 1.3042 | Dedicated Intent Head. |
| **neon053** | 2.89M | 1.3129 | QKVI SiLU. |
| **neon021** | 3.15M | 1.2842 | Source raw I, σ(V). |
| **neon019** | 3.15M | 1.3150 | Source raw I, raw V. |
| **neon045** | 2.64M | 1.3618 | Gated Calc (Q+V). |
| **neon025** | 2.89M | 1.3404 | Post-Norm Study. |
| **neon056** | 3.17M | 1.3369 | Double-Gated. |
| **neon057** | 3.15M | 1.3418 | Differential Intent. |
| **neon052** | 2.67M | 1.3447 | Matrix Intent. |
| **neon026** | 2.89M | 1.3553 | No-Intent Control. |
| **neon028** | 2.89M | 1.3554 | MLA Control. |
| **neon049** | 2.64M | 1.3594 | Gated Calc Gap. |
| **neon048** | 2.64M | 1.3609 | Gated Calc Biased. |
| **⭐ neon010** | 2.64M | 1.3698 | **Calculated Intent**. |
| **neon050** | 2.64M | 1.3740 | Gated Calc + Norm. |
| **neon041** | 2.64M | 1.3754 | Gated Calc (QV). |
| **neon047** | 2.64M | 1.3758 | Gated Calc Bounded. |
| **neon017** | 2.89M | 1.3764 | Result raw I, σ(V). |
| **neon042** | 2.64M | 1.3774 | Gated Calc (QK). |
| **neon038** | 2.62M | 1.3784 | Calc Q+σ(KV). |
| **neon043** | 2.64M | 1.3780 | Gated Calc (KV). |
| **neon018** | 2.89M | 1.3808 | Result σ(I), σ(V). |
| **neon032** | 2.62M | 1.3854 | Calc σ(Q⊙K). |
| **neon035** | 2.62M | 1.3866 | Calc LN(Q+V). |
| **neon033** | 2.62M | 1.3875 | Calc σ(K⊙V). |
| **neon044** | 2.64M | 1.3905 | Gated Calc (QKV_prod). |
| **neon051** | 2.63M | 1.3938 | Linear Combination. |
| **neon030** | 2.89M | 1.3953 | RMSNorm Baseline. |
| **neon031** | 2.62M | 1.3975 | Calc σ(Q⊙V). |
| **neon036** | 2.62M | 1.4049 | Calc norm(Q+K+V). |
| **neon029** | 2.89M | 1.4158 | LayerNorm Baseline. |
| **neon034** | 2.62M | 1.4229 | Calc σ(Q⊙K⊙V). |
| **neon022** | 2.89M | 1.4234 | Source σ(I), σ(V). |
| **neon037** | 2.62M | 1.4264 | Calc σ(Q)⊙tanh(V). |
| **neon039** | 2.62M | 1.4417 | Calc tanh(gap). |
| **neon054** | 2.64M | 1.4444 | Gated Calc SiLU. |
| **neon058** | 2.64M | 1.4620 | Residual Additive. |
| **neon005** | 2.62M | 1.4673 | SwiGLU Baseline. |
| **neon040** | 2.62M | 1.5139 | Calc RMS(Q⊙V). |
| **neon006** | 2.49M | 1.5467 | MLA Baseline. |
| **neon002** | 2.10M | 1.7434 | RMSNorm Baseline. |
| **neon001** | 2.11M | 1.7509 | Baseline GPT-2. |
| **neon003** | 1.71M | 1.8868 | MQA Baseline. |
| **neon004** | 1.71M | 1.9451 | Shared MLP. |
| **neon007** | 2.63M | 3.0147 | DeltaNet (Fail). |
| **neon008** | 2.63M | 6.0381 | Unit Sphere (Fail). |

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
| **⭐ neon143** | **3.16M** | **0.6604** | **Silent Hydra (Attention-Free)**. [HP0 CHAMPION] |
| **neon156** | 3.15M | 0.6975 | Spectral Silent Hydra. |
| **neon157** | 3.16M | 0.7063 | Wide-Merge Silent Hydra. |
| **⭐ neon160** | **3.16M** | **0.7149** | **The Ghost**. Hybrid (3x Silent + 1x Attn). |
| **⭐ neon139** | **3.16M** | **0.7159** | **Sequential Kernel Expansion** (k=3→9). |
| **⭐ neon130** | **3.15M** | **0.7265** | **Sharp-V Hyper-Synergy**. MQI Efficiency. [Co-SOTA] |
| **⭐ neon116** | **3.15M** | **0.7269** | **Full Multi-Head Conv-Attention**. |
| **neon131** | **3.15M** | **0.7297** | **Qwen-NexT Synergy**. Zero-Centered stability. |
| **neon153** | 3.15M | 0.7368 | Multi-Head Dilated Context. |
| **neon141** | 3.15M | 0.7374 | Denoising Bottleneck Hydra. |
| **neon148** | 3.16M | 0.7428 | Asymmetric Search (Sharp-Q). |
| **neon145** | 3.16M | 0.7469 | Multi-Head Denoising Bottleneck. |
| **neon129** | 3.15M | 0.7513 | Hyper-Synergy (Full+MQI+Bias). |
| **neon164** | 3.15M | 0.7525 | Pyramidal Silent Hydra (8L). |
| **neon127** | **3.15M** | **0.7555** | **Biased Attention Conv**. Stability gain. |
| **neon161** | 3.15M | 0.7573 | Deep Silent Hydra (8L). |
| **neon159** | 3.15M | 0.7583 | Clean-Room Silent Hydra. |
| **⭐ neon114** | **3.15M** | **0.7652** | **Sharp-Value Conv-Attention**. |
| **⭐ neon115** | **3.15M** | **0.7663** | **Multi-Head Conv-Attention**. |
| **neon152** | 3.16M | 0.7695 | Multi-Head Asymmetric Search. |
| **neon113** | 3.15M | 0.7707 | Conv-Attention (Shared). |
| **neon124** | **3.23M** | **0.7727** | **Multi-Query Intent (MQI)**. Sharing works. |
| **neon165** | 3.15M | 0.7733 | Res-Gated Silent Hydra (8L). |
| **neon149** | 3.15M | 0.7774 | Dilated Receptive Fields. |
| **neon137** | 3.16M | 0.7796 | Hierarchical Context Stack. |
| **neon166** | 3.15M | 0.7823 | Deep Spectral Hydra (8L). |
| **neon162** | 3.15M | 0.7910 | Deep Hybrid Ghost (8L). |
| **neon128** | 3.42M | 0.7905 | Gateless Context baseline. |
| **neon132** | 3.14M | **0.8000** | **Causal Spectral Hydra**. Multi-scale bank. |
| **neon150** | 3.16M | 0.8048 | Intent Recurrence. |
| **neon117** | 3.15M | 0.8050 | Activated Multi-Head Conv-Attn (SiLU). |
| **neon125** | 3.17M | 0.8124 | Bottleneck Intent. |
| **neon121** | 3.15M | 0.8145 | Context-Aware Intent Only. |
| **neon123** | 3.15M | 0.8203 | Residual Gated Attention. |
| **neon144** | 3.15M | 0.8266 | Sigmoid Bottleneck Hydra. |
| **neon122** | 3.15M | 0.8283 | Zero-Centered Norm. |
| **neon154** | 3.16M | 0.8286 | Multi-Head Intent Recurrence. |
| **⭐ neon110** | 3.15M | 0.8365 | Pure Hydra Swish (MLP-Only SOTA). |
| **⭐ neon108** | 3.15M | 0.8366 | Pure Hydra Single-Scale. |
| **neon100** | 3.15M | 0.8437 | Dual-Scale Pure Hydra. |
| **neon147** | 3.16M | 0.8477 | Multi-Head Sigmoid Bottleneck. |
| **neon133** | **3.15M** | **0.8586** | **Commander Head**. Dynamic weights. |
| neon106 | 3.15M | 0.8608 | Dual-Gated Pure Hydra runner-up. |
| neon102 | 3.15M | 0.8655 | Sandwich Hydra Test. |
| **⭐ neon085** | **3.15M** | **0.8670** | **Dual-Scale Hydra**. (Previous 3M SOTA). |
| **neon105** | 3.15M | 0.8671 | Early Starter (Hydra L0). |
| **neon095** | 3.15M | 0.8703 | Progressive Kernels (k=3-17). |
| **neon089** | 3.15M | 0.8768 | Dense Pyramidal (k=3,5,7,9). |
| **neon090** | 3.15M | 0.8786 | Recursive Hydra Gating. |
| **neon109** | 3.14M | 0.8807 | Pure Hydra High-Reach (k=20). |
| **⭐ neon081** | **3.13M** | **0.8812** | **Context-scaled Hydra**. (Wiki Champion). |
| **neon097** | 3.15M | 0.8817 | Triple-Scale Gate (k=3,5,9). |
| **neon096** | 3.15M | 0.8832 | Heterogeneous Stack Hydra. |
| **neon138** | 3.15M | 0.8804 | Strategic Colossus (k=33). |
| **neon151** | 3.15M | 0.8801 | Inception Value (Multi-Fidelity). |
| **neon080** | 3.15M | 0.8875 | Scaling Study (Width). |
| **neon142** | 3.16M | 0.8931 | Global Hum Hydra. |
| **neon098** | 3.15M | 0.8940 | Dilated Hydra (RF=65). |
| **neon088** | 3.15M | 0.8944 | Competitive Hydra (Max-Pool). |
| **neon155** | 3.15M | 0.8952 | Multi-Head Inception Value. |
| **neon163** | 3.15M | 0.9071 | Alternating Ghost (8L). |
| **neon087** | 3.12M | 0.9018 | Pyramidal Hydra (k=3,9,27). |
| **neon107** | 3.15M | 0.9103 | Massive Reach Pure Hydra (RF=65). |
| **neon086** | 3.14M | 0.9168 | Res-Hydra (Residual Context). |
| **⭐ neon077** | **3.09M** | **0.9172** | **Conv-Gated Hydra**. Matches Baseline. |
| **⭐ neon016** | **3.15M** | **0.9174** | **Learned Intent [Tok4 Baseline]**. |
| **neon112** | 3.14M | 0.9231 | Bottleneck-Gated Wide Hydra. |
| **neon103** | 3.15M | 0.9245 | Inv Sandwich (S-H-H-S). |
| **neon101** | 3.15M | 0.9253 | Block Hetero (2-Swi / 2-Hyd). |
| **neon119** | 3.16M | 0.9331 | Dynamic Swish-Beta. |
| **neon104** | 3.15M | 0.9342 | Late Bloomer (3-Swi / 1-Hyd). |
| **neon126** | **3.15M** | **0.9627** | **No MLP Conv [ARCHITECTURAL FAIL]**. |
| **neon082** | 3.16M | 0.9886 | Fair Hydra Scaling Study. |
| **neon120** | 3.16M | 0.9710 | Activated Intent (SiLU). |
| **neon099** | 3.15M | 0.9961 | Residual Multiplicative Gating. |
| neon111 | 3.08M | 0.9968 | Space-Aware Matrix (Failed). |
| **neon146** | 3.15M | 1.0156 | Multi-Head Global Hum. |
| **neon134** | 3.15M | 1.0224 | **Mamba Hybrid**. Linear recurrence scan. |
| **neon140** | 3.16M | 1.0654 | Parallel Spectral Heads. |
| **neon118** | 3.16M | 1.1725 | L2-Norm Conv-Attention. |
| **neon158** | 3.16M | 1.2885 | Dilated Silent Hydra. |
| **neon135** | 3.16M | 1.4692 | **Holographic Projection**. Failed experiment. |

| **⭐ neon092** | **9.98M** | **0.1961** | **10M Dual-Scale Hydra [SOTA]**. |
| **neon091** | 9.98M | 0.1962 | 10M Hydra Scaling (k=9). |
| **neon094** | 9.98M | 0.2067 | 10M Hydra-Base (No Intent). |
| **neon061** | 9.98M | 0.2364 | Legacy Wide MLP baseline. |
| **neon093** | 9.98M | 0.2512 | 10M 8-Layer Deep standard. |

### 🧪 Benchmark: Wiki103 / Tok4
*WikiText-103 Dataset (100MB). Tok4 = 4,096 Vocab.*

| Model | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- |
| **⭐ neon185** | **5.28M** | **3.1364** | **Wiki103 5M SOTA**. SwiGLU-Conv. |
| **neon187** | 5.28M | 3.1381 | Full SiLU (Swish). |
| **neon167** | **5.28M** | **3.1484** | **Wiki103 5M Baseline**. Giant Synergy. |
| **⭐ neon180** | **5.28M** | **3.1485** | **Sharp-V Giant**. Wiki CO-SOTA. |
| **⭐ neon169** | **5.30M** | **3.1485** | **Ascending Attention Hierarchy**. |
| **neon171** | 5.28M | 3.1492 | Ascending MLP. |
| **neon183** | 5.28M | 3.1496 | RoPE Before Convolution. |
| **neon173** | 5.29M | 3.1502 | Dual Ascending (MHI). |
| **neon186** | 5.28M | 3.1528 | SiLU Attn + Sigmoid MLP. |
| **neon170** | 5.30M | 3.1533 | Descending Attention Hierarchy. |
| **neon176** | 5.29M | 3.1538 | Dual Ascending (MQI + Wide MLP). |
| **neon175** | 5.28M | 3.1565 | MQI MLP-Hierarchy. |
| **neon179** | 5.28M | 3.1568 | Sharp Intent Giant. |
| **neon174** | 5.29M | 3.1601 | MQI Att-Hierarchy. |
| **neon181** | 5.28M | 3.1663 | Sharp Search (Q,K) Giant. |
| **neon172** | 5.28M | 3.1665 | Descending MLP. |
| **neon168** | 5.28M | 3.1678 | Sharp Intent & Value Giant. |
| **neon178** | 5.28M | 3.1712 | Spectral Synergy Giant. |
| **neon177** | 5.28M | 3.1776 | 5-Layer MQA Giant. |
| **neon184** | 5.28M | 3.1793 | No RoPE. |
| **neon182** | 5.27M | 3.1893 | Pure Attention (No Convs). |
| **neon188** | 5.28M | 3.1838 | Dual-Attention Bipolar Gate. |
| | | | |
| **⭐ neon092** | **9.98M** | **3.0575** | **10M Wiki SOTA**. Full Synergy. |
| **neon091** | 9.98M | 3.0797 | 10M Hydra Wiki. |
| **neon061** | 9.98M | 3.0940 | Legacy 10M Baseline. |
| **neon093** | 9.98M | 3.0955 | 8-Layer Deep Standard. |
| **neon094** | 9.98M | 3.0977 | 10M Hydra-Base (No Intent). |
| | | | |
| **neon116** | 3.15M | 3.2499 | Full Conv-Attention (3M class). |
| **neon139** | 3.16M | 3.2637 | Sequential Kernel Expansion. |
| **neon127** | 3.15M | 3.2722 | Biased Conv-Attention. |
| **neon130** | 3.15M | 3.2733 | Sharp-V Hyper-Synergy. |
| **neon108** | 3.15M | 3.2803 | Pure Hydra Single-Scale. |
| **⭐ neon081** | **3.13M** | **3.2844** | **Wiki103 3M SOTA**. |
| **neon131** | 3.15M | 3.2840 | Qwen-NexT Synergy. |
| **⭐ neon077** | **3.09M** | **3.2880** | Conv-Gated Hydra Wiki. |
| **neon016** | 3.15M | 3.2885 | Wiki Tok4 Baseline. |
| **neon085** | 3.15M | 3.2905 | Dual-Scale Hydra Wiki. |
| **neon110** | 3.15M | 3.2927 | Pure Hydra Swish. |
| **neon100** | 3.15M | 3.2812 | Pure Hydra. |
| **neon129** | 3.15M | 3.2524 | Hyper-Synergy. |
| **neon160** | 3.16M | 3.3071 | The Ghost (Hybrid). |
| **neon063** | 4.20M | 3.3141 | Attention-in-MLP Wiki. |
| **neon065** | 4.46M | 3.3171 | Big Single Head Wiki. |
| **neon162** | 3.15M | 3.3369 | Deep Hybrid Ghost (8L). |
| **neon066** | 2.11M | 3.3377 | Fair Fight Big Head Wiki. |
| **neon062** | 1.57M | 3.3475 | MLP-Free Wiki. |
| **neon064** | 3.02M | 3.4275 | Hadamard Merge Wiki. |
| **neon143** | 3.16M | 3.5287 | Silent Hydra. |
| **neon156** | 3.15M | 3.5405 | Spectral Silent Hydra. |
| **neon161** | 3.15M | 3.5429 | Deep Silent Hydra (8L). |

### 🧪 Benchmark: Wiki103 / Tok5
*WikiText-103 Dataset (100MB). Tok5 = 8,192 Vocab (Character-Level).*

| Model | Params (Ex-Emb) | Val Loss | Summary |
| :--- | :--- | :--- | :--- |
| **⭐ neon185** | **5.00M** | **3.4278** | **Wiki103 Tok5 SOTA**. SwiGLU-Conv Baseline. |
| **⭐ neon198** | **5.00M** | **3.4333** | Shifted Tanh Intent `tanh(I)+1`. Range [0,2], init=1. |
| **neon194** | 5.28M | 3.4364 | Gated Polarity Residual (5.28M). |
| **⭐ neon205** | **5.00M** | **3.4381** | **Double Gate `σ(I)·SiLU(I)`**. Fine-grained control. |
| **neon197** | 5.00M | 3.4400 | Gated Polarity Residual (5M fair ver). |
| **neon201** | 5.00M | 3.4401 | Learnable Blend `α·σ(I)+(1-α)·tanh(I)`. |
| **neon199** | 5.00M | 3.4404 | Biased Bipolar Intent `tanh(I+0.55)`. |
| **neon195** | 5.00M | 3.4468 | Bipolar Intent `tanh(I)`. |
| **neon204** | 5.00M | 3.4504 | SiLU Intent `SiLU(I)`. Unbounded amplification. |
| **neon167** | 5.28M | 3.4540 | Giant Synergy (5M Baseline, no SwiGLU MLP). |
| **neon196** | 5.00M | 3.4568 | Sparse Bipolar `tanh³(I)`. |
| **neon200** | 5.00M | 3.4585 | Split Intent (half sigmoid / half tanh). |
| **neon192** | 5.28M | 3.4591 | Soft Flip + Learned Skip α. |
| **neon190** | 5.28M | 3.4636 | Signed Residual. |
| **neon203** | 5.00M | 3.4661 | Shifted Tanh² `(tanh(I)+1)²`. Range [0,4]. |
| **neon193** | 5.28M | 3.4727 | Residual Flip (branch attn). |
| **neon206** | 5.00M | 3.5146 | Standard SwiGLU MLP (no Hydra conv). |
| **neon189** | 5.28M | 3.5296 | Pure Tanh Gate (init problem). |
| **neon191** | 5.28M | 3.5674 | Bottleneck MLP Gate. |
| **neon202** | 5.00M | 3.7389 | Silent Hydra 5M (Attention-Free). |

---

## 📈 Key Discovery Timeline

1.  **Intent Evolution (001-022)**: We proved that **Result Gating** (gating the attention output) is significantly better than **Source Gating** (gating before attention). σ(I) is essential.
2.  **Calculated Intent (031-055)**: We attempted to "calculate" intent from Q/K/V interactions to save parameters. `neon010` and `neon046` proved that these "calculated" signals can match full learned gating, by saving learnt intent parameters and scaling other parts of the model.
3.  **The Head Discovery (065-069)**: We found that at our ~3M scale, **1 Massive Head (512-dim)** outperforms the standard 4-head configuration, but mostly due to internal parameter scaling. Under a "Fair Fight" (`neon066`), 4 heads remained the most optimal.
4.  **Hydra Era (070-077)**: Introduced context-aware gating in the MLP. `neon077` (Conv-Gated Hydra) successfully matched the Attention baseline using a lightweight convolutional heuristic.
5.  **Scaling Breakthrough (080-081)**: Proved that context is the primary bottleneck. `neon081` (**k=9**) shattered the baseline, achieving 0.88 val loss at 3M parameters.
6.  **Modern Hybrids (078-079)**: Replicating state-of-the-art architectures like Qwen3-Next to benchmark against our simplified blocks.
7.  **The Gauntlet Synergy (091-094)**: Proved that **Double Gating** (Intent Attention + Hydra MLP) creates a synergistic effect. `neon092` crushed both deep standard models (`neon093`) and ablation baselines, proving architectural intelligence beats raw parameter scaling.
8.  **Pure Hydra Discovery (100-105)**: Discovered that the SiLU-identity gate is optional. `neon100` (Pure Convolutional Gating) achieved the new project SOTA at 3M scale. The increased parameter budget from removing the identity gate allows for much wider MLPs.
9.  **Locally-Aware Attention (113-116)**: Discovered that adding $k=3$ depthwise convolutions to $Q, K, V,$ and $Intent$ projections AFTER linear projection creates a "Locally-Aware Search." `neon116` achieved a massive SOTA jump from 0.88 to **0.72**, proving that attention is most effective when it sees its neighbors.
10. **The Force Multiplier Discovery (126)**: Proved that Locally-Aware Attention is NOT a standalone winner. Ablation `neon126` (0.96 loss) showed that without the **Hydra MLP** providing local context foundationally, the attention mechanism "flies blind." Local context is a dual-layer requirement.
11. **Hyper-Synergy & MQI (130-131)**: Optimized the architecture via **Multi-Query Intent (MQI)**, sharing a single intent gate across all heads. `neon130` matched the project SOTA (0.72) while using the saved parameters to push MLP width to $d_{ff}=572$, establishing the current most efficient 3M architecture.
12. **The Blue Sky Pivot (132-135)**: Moving beyond fixed convolutions.
    - **Commander Head (133)**: Achieved a solid **0.85** loss, proving that predicting kernels on-the-fly is a powerful lever for local intelligence.
    - **Holographic (135)**: Demonstrated that complex interference is highly sensitive and difficult to regularize (1.46 loss).
    - **Mamba/Fourier (134/132)**: Discovered that fast recurrent scans require careful masking (NaN fix) and dimension alignment to match the stability of spatial convolutions.
13. **The Silent Hydra Revolution (136-166)**:
    - **Attention-Free Pioneer**: `neon143` (Silent Hydra) achieved **0.66** val loss on HP0 — the best in the entire project — by replacing Softmax Attention with pure convolutional gating.
    - **The Ghost Architecture**: `neon160` (3x Silent + 1x Attention) proved that a single attention layer at the end is sufficient for grounding.
    - **Sequential Kernel Expansion**: `neon139` (progressive k=3→9) achieved **0.71**, proving that hierarchical context builds deeper abstraction than uniform kernels.
    - **Deep Variants (161-166)**: Scaling to 8 layers showed diminishing returns for attention-free models on small datasets, but pyramidal kernels (`neon164`) remained competitive.
    - **Wiki Gap**: Silent Hydra models excelled on HP0 but showed a significant gap on Wiki103 (~3.5 vs ~3.2), confirming that attention is critical for complex generalization.
14. **The 5M Upscale & Hierarchical Abstraction (167-176)**:
    - **Quantity Meets Quality**: Scaling from 3M to 5M parameters and standardizing on **4x MLP width** resulted in an immediate SOTA jump on Wikipedia.
    - **Hierarchical Sensing**: Discovered that **Ascending Kernels** (starting sharp at k=3 and expanding to k=9 with depth) outperform uniform and descending kernels.
    - **The MHI Rebound**: Crucially, discovered that **Multi-Head Intent (MHI)** is superior to Multi-Query Intent (MQI) at the 5M scale, despite MQI allowing for wider MLPs. Head-specific gating diversity is key for high-level reasoning.
    - **Current Champions**: `neon167`, `neon169`, and `neon180` form a 3-way tie for the record at **3.148**.
15. **The Search-is-King Discovery (179-182)**:
    - **Convolutional Mandate**: Proved that **Blurred Search (Q, K)** and **Blurred Gating (I)** are mandatory for Wikipedia. Moving to raw dot-product (`neon182`) or sharp matching (`neon181`) caused immediate regressions.
    - **Value Flexibility**: Discovered that keeping **Value Sharp** (`neon180`) is the only viable ablation, as intelligence at 5M lies in the *selection mechanism*, not the *content smoothing*.
    - **The Depth Trap**: Confirmed that adding a 5th layer via MQA (`neon177`) is strictly worse than keeping 4 layers with full head-specific gating diversity (MHI).
16. **The Positional Study (183-184)**:
    - **RoPE Placement**: `neon183` (RoPE before convolution) matched the baseline, while `neon184` (no RoPE, relying solely on convolutions for position) showed only minor degradation (3.179 vs 3.148), confirming that 1D convolutions provide substantial positional information.
17. **The SwiGLU Mandate (185-187)**:
    - **SiLU MLP Winner**: `neon185` (Swish MLP) achieved a new SOTA **3.136** by replacing Sigmoid gating with SiLU (SwiGLU-style) in the MLP.
    - **Sigmoid Attention Winner**: However, using SiLU in the Attention gate (`neon186`, `neon187`) degraded performance relative to the base. Interpretation: **Attention Probability** is naturally bounded $[0,1]$ and benefits from Sigmoid, whereas **MLP features** are unbounded and benefit from SiLU's non-saturation.
18. **The Semantic Gate Exploration (188-206)**:
    - **Core Question**: Can replacing `sigmoid(Intent)` with alternative activation functions improve the attention gate? Explored 19 variants across multiplicative gates, bipolar intent, amplification, and structural ablations.
    - **Sigmoid Wins**: After exhaustive testing, `sigmoid(I)` remains the best attention gate. The top challenger `neon198` (`tanh(I)+1`, range [0,2]) achieved **3.4333** vs baseline **3.4278** — within noise margin but not a clear win.
    - **Double Gate Discovery**: `neon205` (`σ(I)·SiLU(I)`) achieved **3.4381** — the strongest "novel" activation, proving that combining sigmoid's probabilistic selection with SiLU's non-saturating scaling gives fine-grained control.
    - **Init Matters**: Models initializing the gate at 1.0 (identity) consistently outperformed those starting at 0.5 or 0.0. `neon189` (pure tanh, init=0) suffered a 2000-step convergence delay.
    - **Hydra Conv Confirmed**: `neon206` (standard SwiGLU MLP without Hydra conv) scored **3.5146** vs baseline **3.4278** — a **0.087 gap** proving the Hydra convolutional gate in the MLP is worth ~0.09 val loss points.
    - **Attention-Free Gap**: `neon202` (Silent Hydra at 5M) scored **3.7389** — a massive **0.31 gap** from baseline, confirming that attention is critical for Wikipedia-scale generalization even with 4.6x MLP width.
