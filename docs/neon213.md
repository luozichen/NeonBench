# neon213: The Growable 20M Model

**neon213** is the first **26M parameter** model (20M non-embedding) in the NeonBench series, trained on the **FineWeb-Edu** dataset using a **Progressive Growth** strategy. It represents a significant scale-up from the previous 5M class.

| Property | Value |
|---|---|
| **Parameters (Total)** | **26.49M** |
| **Parameters (Active)** | **20.20M** (Non-Embedding) |
| **Architecture** | **Growable SwiGLU-Conv** (neon185 base) |
| **Dataset** | **FineWeb-Edu** (Sample-10GT) |
| **Tokenizer** | **tok6** (16,384 Vocab, GPT-2 subset) |
| **Dimensions** | $d_{model}=384, n_{head}=6, n_{layers}=8, d_{ff}=1536$ |
| **Context** | **Growable** ($k=1 \to 9$) |
| **Status** | **Active (FP16 Checkpoint Available)** |

---

## 🏗️ Architecture

The architecture is based on **neon185 (SwiGLU-Conv)**, which features:
1.  **SwiGLU MLP**: `w2(SiLU(w1(x)) * w3(x))` gating.
2.  **Hydra Convolution**: Depthwise convolutions on MLP gates to provide local context.
3.  **Conv-Attention**: Depthwise convolutions on Q/K/V/I projections.
4.  **Sigmoid Attention Gate**: Learned `sigmoid(Intent)` gate on attention output.

### Growable Kernels
Unlike previous static models, neon213 features **configurable kernel sizes** (`conv_k`, `mlp_k`). This allows the model to start with pointwise operations ($k=1$) and grow its receptive field during training.

```python
# Conv-Attention Layer
self.conv_q = nn.Conv1d(d, d, kernel_size=k, groups=d)  # k grows 1->9
self.conv_k = nn.Conv1d(d, d, kernel_size=k, groups=d)
self.conv_v = nn.Conv1d(d, d, kernel_size=k, groups=d)
self.conv_i = nn.Conv1d(d, d, kernel_size=k, groups=d)

# SwiGLU MLP Layer
self.conv_gate = nn.Conv1d(d, d, kernel_size=k, groups=d) # k grows 1->9
```

---

## 💡 Key Innovations

### 1. Learned Intent Gating

Standard Gated Scaled Dot-Product Attention (as seen in architectures like Qwen 3.5) derives its output gate from existing projections — typically the Query. The gate is *calculated*, not independently learned:

$$\text{Gated-SDPA}: \quad y = \sigma(W_g \cdot Q) \odot \text{Attn}(Q, K, V)$$

In neon213, the gate is a **fully independent learned projection** called **Intent ($I$)**. Intent has its own dedicated weights (`c_attn` slice) and its own dedicated convolution (`conv_i`), giving it a completely separate representational capacity from Q, K, and V:

$$\text{Intent-Gated}: \quad y = \sigma(\text{Conv}(I)) \odot \text{Attn}(Q, K, V)$$

This means the model can learn **what information to keep** (Intent) independently from **what information to search for** (Query) and **what information to retrieve** (Value). The gate is not a byproduct of the search — it is a first-class citizen with its own parameters.

### 2. Depthwise Convolutions as Communication Channels

The depthwise convolutions applied to Q, K, V, and I are **not simple blurs**. Each convolution kernel is a set of **fully learned, unconstrained weights** — including negative values. This means each dimension can independently decide:
- **How much** of a neighboring token's signal to incorporate (weight magnitude).
- **Whether to amplify or inhibit** that signal (positive vs. negative weights).
- **Which temporal direction** to prioritize (the causal padding ensures only past tokens are visible).

In practice, this creates an **additional token-to-token communication pathway** that operates *before* the attention mechanism. While attention allows tokens to selectively read from any position, the convolutions provide a **fixed, local, per-dimension** channel for adjacent tokens to share information — a form of inductive bias that complements the global, content-based routing of attention.

### 3. Progressive Kernel Growth

Convolutions at large kernel sizes ($k=9$) are powerful but difficult to train from scratch — the model must simultaneously learn *what* to convolve and *how far* to look. neon213 solves this with **Progressive Kernel Growth**:

1. Training begins with **pointwise kernels** ($k=1$), which are equivalent to no convolution at all. The model first learns the fundamentals of attention and MLP gating without any local context.
2. Kernels are then **gradually expanded** ($k=1 \to 3 \to 5 \to 7 \to 9$) using **zero-padding initialization** — the new kernel positions are filled with zeros, so the model's behavior is perfectly preserved at the moment of expansion.
3. The model then **learns to use the new context** during the subsequent training steps, gradually discovering how to exploit wider local neighborhoods.

This approach is analogous to curriculum learning: the model masters simple patterns first, then progressively gains the capacity to leverage richer local context.

---

## 📈 Progressive Growth Training

Training was split into **9 Stages** to stabilize convergence and save compute. The model grew in **Depth** (Layers) and **Context** (Kernel Size).

| Stage | Layers | Kernel ($k$) | Steps | Description |
|---|---|---|---|---|
| **1** | 4 | 1 | 5,000 | **Deep & Narrow**: Learning simple relations. |
| **2** | 5 | 1 | 3,000 | **Depth Growth 1**: Identity initialization. |
| **3** | 6 | 1 | 3,000 | **Depth Growth 2**: Identity initialization. |
| **4** | 7 | 1 | 3,000 | **Depth Growth 3**: Identity initialization. |
| **5** | 8 | 1 | 3,000 | **Full Depth**: Reached 8 layers. |
| **6** | 8 | 3 | 3,000 | **Context Explosion**: Expanded to $k=3$. |
| **7** | 8 | 5 | 3,000 | **Context Expansion**: Expanded to $k=5$. |
| **8** | 8 | 7 | 3,000 | **Context Expansion**: Expanded to $k=7$. |
| **9** | 8 | 9 | 5,000 | **Final Refinement**: Full context $k=9$. |

**Total Steps**: ~31,000 (Batch Size 64, ~2M tokens/step).

### Growth Mechanics
1.  **Layer Growth**: New layers are initialized with **zero output projection**, effectively acting as identity functions ($x + 0 = x$) to preserve the forward pass.
2.  **Kernel Growth**: New kernel weights are **zero-padded** on the causal side (left), ensuring the convolution output remains identical to the smaller kernel state at the moment of expansion.

---

## 💾 Checkpoint & Quantization

The final model checkpoint exceeded the GitHub 100MB file limit (**101 MB**).
To resolve this, the checkpoint was converted to **Float16 (Half Precision)**.

- **Original Size**: 101.16 MB
- **FP16 Size**: 62.60 MB
- **Format**: Standard PyTorch `state_dict`.
- **Compatibility**: `NeonModelEngine` automatically handles the `fp16` $\to$ `fp32` cast during loading.

---

## 📊 Performance

| Metric | Value | Notes |
|---|---|---|
| **Val Loss** | **3.69** | FineWeb-Edu (Harder sample than Wiki103). |
| **Generation** | Coherent | Produces grammatically correct English paragraphs. |

**Sample Generation**:
> *"**The meaning of life is** not obvious. It is a story about the way things change. It is a very great story about the great place to go and the future — everything you really expect from the past — of history and"*

> *"**The meaning of life is** determined. Its meaning is derived from the social class and the economic base of it. The word’s meaning is derived from the social class and describes a society’s ability to grow. It contains"*
