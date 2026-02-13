# 🐉 Neon116: Full Multi-Head Conv-Attention

Neon116 represents a fundamental breakthrough in the project, achieving the first sub-0.80 loss at the 3M parameter scale. Discovery **#9 and #10** in the timeline, it proved that context is a "Dual-Level" requirement: both the Attention mechanism and the MLP must be locally-aware to achieve peak efficiency.

## 📊 Performance Summary

| Dataset | Metric | neon081 (Prev. SOTA) | **neon116 (Current SOTA)** | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **HP0 / Tok4** | Val Loss | 0.8812 | **0.7269** | **+17.5%** |

## 🏗️ Architectural Core

Neon116 introduces the concept of **Locally-Aware Search**. While previous models used convolutions only in the MLP (Hydra) or only on the Intent gate, Neon116 applies $k=3$ depthwise convolutions to every search projection:

### 1. The Conv-Attention Formula
Every head in the attention mechanism operates on convolved representations:
- $Q_{conv} = \text{Conv}_{k=3}(W_q x)$
- $K_{conv} = \text{Conv}_{k=3}(W_k x)$
- $V_{conv} = \text{Conv}_{k=3}(W_v x)$
- $I_{conv} = \text{Conv}_{k=3}(W_i x)$

The final output is computed as:
$$y = \sigma(I_{conv}) \odot \text{Softmax}\left(\frac{Q_{conv} K_{conv}^T}{\sqrt{d}}\right) V_{conv}$$

### 2. The "Force Multiplier" Discovery
The most significant finding from Neon116 (and its ablation `neon126`) is that **Locally-Aware Attention is not a standalone champion.** 
- When the MLP convolution was removed in `neon126`, the loss skyrocketed to **0.96**.
- This proves that the Attention mechanism "flies blind" unless the **Hydra MLP** has already set a stable local context foundation. The two mechanisms act as a force multiplier for each other.

## ⚖️ Parameter Efficiency
Neon116 was calibrated to **3,154,688** parameters. 
- **The Tradeoff**: Adding 4 depthwise convolutions per layer required reducing the MLP width ($d_{ff}$) from 572 down to **507**.
- **The Result**: Despite having a narrower MLP, the "contextualized" projections provided such high-quality signals to the attention heads that the model's intelligence increased by over 17%.

## 🔭 Summary
Neon116 proves that for small-scale models, **where you look** (locally-aware projections) is more important than **how wide you are** (MLP capacity).

---
*Developed as part of the NeonBench Dual-Context Study.*
