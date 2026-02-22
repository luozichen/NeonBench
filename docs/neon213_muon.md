# neon213-Muon: The SOTA Reborn

**neon213-Muon** is the optimized evolution of the neon213 architecture. By replacing AdamW with the **Muon Optimizer** (Momentum Orthogonalized by Newton-Schulz/Polar Express) and extending the kernel growth to **$k=21$**, this model achieved a project-wide **State-of-the-Art (SOTA)** Val Loss of **3.44**.

| Property | Value |
|---|---|
| **Parameters (Total)** | **26.49M** |
| **Parameters (Active)** | **20.20M** (Non-Embedding) |
| **Optimizer** | **Muon V4** (Orthogonalized) |
| **Final Val Loss** | **3.4432** (FineWeb-Edu) |
| **Participation Ratio (V)** | **25.9** (vs 12.2 in AdamW) |
| **Kernels (max)** | **$k=21$** (Progressive Growth) |
| **Status** | **Project SOTA (FP16 Checkpoint: `neon213_muon_sota_fp16.pth`)** |

---

## 🚀 The Muon Breakthrough

The primary bottleneck in the original neon213 (AdamW) was **Dimensional Collapse**. Despite having a wide 384-dimensional residual stream, the model was only utilizing a fraction of its capacity.

### 1. Participation Ratio (PR) Audit
Participation Ratio measures how many dimensions are effectively carrying signal.
*   **AdamW Baseline**: V-PR ~12.2 (Only 3% of the 384 dimensions utilized).
*   **Muon SOTA**: V-PR **25.9** (Cold Start) / **34.8** (Peak).

Muon enforces **orthogonality** in the 2D weight matrices (Attention Projections and MLPs), forcing the model to distribute information across the entire latent space rather than collapsing onto a few dominant features.

### 2. SOTA Comparison
| Metric | Neon213 (AdamW) | **Neon213 (Muon)** | Improvement |
|---|---|---|---|
| **Best Val Loss** | 3.53 | **3.44** | **-0.09 Loss** |
| **V-PR (Utilization)** | ~12.2 | **25.9** | **+112% Diversity** |
| **Residual PR (L3)** | ~1.0? | **67.9** | **Massive** |

---

## 🏗️ Progressive Growth $k=1 \to 21$

While the original neon213 capped kernels at $k=9$, the Muon version proved stable enough to grow to **$k=21$**. This provides the model with an expansive local receptive field that handles complex grammatical structures and long-range local co-occurrence much more effectively.

### Training Schedule (Stage 12)
1.  **Hybrid Growth**: $k=1 \to 3 \to 5 \dots \to 21$ (Odd increments every 3,000 steps).
2.  **Long Tail (Stage 12)**: 30,000 steps at $k=21$ using a smooth Cosine decay.
3.  **Hysteresis Recovery**: After each kernel growth "shock," the Muon-backed heads recovered much faster than AdamW, consistently finding deeper minima.

---

## 🧠 Architecture Highlights

*   **Intent-Gated Attention**: A dedicated 4th projection ($I$) provides a learned $\sigma(I)$ gate on the attention output.
*   **Hydra-MLP**: Depthwise convolutions ($k=21$) in the MLP gate provide a "local filter" before the SwiGLU non-linearity.
*   **Orthogonal Projections**: All dense layers are kept near-orthogonal by the optimizer, maximizing "feature diversity."

---

## 🌐 Deployment & Visualizer

The model is deployed on **NeonConnect** with **FP16 Quantization**, dropping the checkpoint size from **~101MB** to **~63MB** with zero measurable loss in perplexity.

*   **Viz Demo**: [luozichen.pythonanywhere.com](http://luozichen.pythonanywhere.com)
*   **Key Viz Metric**: Check the **"Heads"** tab for the Muon SOTA model—you will see significantly more "rainbow-like" activity in the feature maps compared to the AdamW baseline, indicating high dimensional utilization.

---

## 📜 Usage
```python
from models.neon213 import Neon213
config = {
    'd_model': 384, 'n_head': 6, 'n_layers': 8, 'd_ff': 1536,
    'conv_k': 21, 'mlp_k': 21, 'vocab_size': 16384
}
model = Neon213(config)
# Load neon213_muon_sota_fp16.pth
```
