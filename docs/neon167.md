# 🐉 Neon167: Scaled Full Conv-Attention (The Giant)

Neon167 is the flagship of the **5M Parameter Class**, representing the successful transition of the project's architectural discoveries from small-scale (3M) to medium-scale (5M) general knowledge modeling. It currently holds the **Co-SOTA** title for Wiki103 generalization.

## 📊 Performance Summary

| Dataset | Metric | neon116 (3M Baseline) | **neon167 (5M Champion)** | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Wiki103 / Tok4** | Val Loss | 3.2499 | **3.1484** | **+3.1%** |

## 🏗️ Architectural Core

Neon167 is a direct, vertical scale-up of the **Full Blur** architecture first established in `neon116`. It confirms that the dual-level context requirement (Locally-Aware Attention + Hydra MLP) scales linearly with parameter count.

### 1. The "Full Blur" Synergy
Unlike models that attempt to harvest parameters by removing convolutions (like `neon130` or `neon180`), Neon167 applies $k=3$ depthwise convolutions to every search and content signal in the attention block.
- **Attention Blur**: $Q, K, V,$ and $I$ (Intent) all receive a local context filter before processing.
- **MLP Foundation**: The Hydra MLP utilizes a **$k=9$** depthwise convolution to generate its gating signal, ensuring the "database" of facts is activated based on a semantic window rather than isolated tokens.
- Discovery: At the 5M scale, this "Blur everything" strategy provides the most robust regularization for the complex structural jumps within Wikipedia text.

### 2. Giant Scaling Laws
Through the transition to 5M parameters, we observed several key "Giant" properties:
- **Quantity Meets Quality**: Increasing the MLP width from the 3M standard (d_ff=507) to the 5M standard (**d_ff=1072**) resulted in an immediate jump in factual retrieval precision.
- **Resilience to Depth**: While `neon116` was sensitive to initialization, the wider 5M footprint of `neon167` exhibits significantly higher stability during the late-stage convergence on Wiki103.

### 3. Verification of Multi-Head Intent (MHI)
Neon167 utilizes **Multi-Head Intent (MHI)**, where each of the **4 attention heads** has its own independent gating projection. 
- Discovery: Despite the success of MQI (Shared Intent) at 3M, Neon167 proves that at larger scales, **Parallel Filtering Diversity** is more valuable than the parameter savings used to widen the MLP further.

## ⚖️ Parameter Calibration
Neon167 is precisely calibrated to **5,283,056** parameters:
- **d_model**: 272
- **n_layers**: 4
- **d_ff**: 1072 (4x expansion)
- **Head Diversity**: 4 Heads (68-dim each)

## 🔭 Summary
Neon167 proves that the **Synergy Architecture** is not just a small-scale heuristic but a viable foundation for scaling. It serves as the baseline against which all hierarchical and spectral "Goliaths" are measured.

## 🎓 Pedagogical Walkthrough: The Data Pipeline

To understand Neon167, let us trace the life of a token through its 5.28M parameters.

### Phase 1: Ingestion & Embedding
1.  **Tokenization**: Raw text is broken into chunks of **256 tokens** (the context window).
2.  **Embedding**: Each token ID is converted into a **272-dimensional** vector via a learned lookup table ($1024 \times 272$).
3.  **Result**: We now have a matrix of shape `[Batch, 256, 272]`.

### Phase 2: The Transformer Block (Iterated 4 Times)
The data enters one of the four blocks. Each block is a two-step process: **Attention** followed by **MLP**.

#### Step A: Locally-Aware Conv-Attention
1.  **Normalization**: The input is passed through **RMSNorm** to stabilize gradients.
2.  **Projection**: A linear layer transforms the 272-dim input into **1,088 dimensions** ($4 \times 272$), which are split into **Query (Q)**, **Key (K)**, **Value (V)**, and **Intent (I)**.
3.  **The QKVI Blur**: Each of these four signals is passed through a **$k=3$ Depthwise Convolution**. 
    *   *Dimension Impact*: Every channel looks at its 2 neighbors. Shape remains `[Batch, 256, 272]`.
4.  **Head Splitting**: The 272 channels are divided into **4 Attention Heads** (each **68-dim** wide).
5.  **RoPE**: **Rotary Positional Embeddings** are applied to Q and K so the model knows the relative distance between tokens.
6.  **Search**: We perform **Scaled Dot-Product Attention** ($Q \times K^T$). This creates a $256 \times 256$ attention matrix per head, which is then multiplied by V.
7.  **The Intent Gate**: The output of the attention search is multiplied by `Sigmoid(Intent)`. This is the project's signature **Result Gating**.
8.  **Output Projection**: The matched results from all 4 heads are concatenated and projected back into the 272-dim residual stream.

#### Step B: The Hydra MLP
1.  **Normalization**: Another **RMSNorm** layer.
2.  **The Topic Sensor**: The input is passed through a **$k=9$ Depthwise Convolution**. This allows the MLP to "sense" the topic of the entire phrase.
3.  **The Bifurcation**:
    *   **Lane 1 (The Content)**: Linear layer $W_1$ expands 272 dimensions to **1,072 dimensions**.
    *   **Lane 2 (The Gate)**: The convolved signal is projected to 1,072 dimensions via `c_gate_proj` and passed through a **Sigmoid**.
4.  **Hydra Gating**: The content ($W_1$) is multiplied by the gate. This is a sparse-like activation where the context determines which "facts" are relevant.
5.  **Down-Projection**: $W_2$ transforms the 1072-dim activation back down to **272 dimensions**.

### Phase 3: Prediction
After 4 layers of this dual-context processing, the final 272-dim vector is normalized and passed through a language model head ($272 \times 1024$) to predict the probability of the next token in the sequence.

---
*Developed as the baseline for the NeonBench 5M Generalization Study.*
