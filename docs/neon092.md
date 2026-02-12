# 🐉 Neon092: Dual-Scale Hydra Architecture

Neon092 represents the current Project SOTA at the 10M parameter scale. It is a **Hybrid Selective Convolutional Transformer** that bridges the gap between local syntactic processing and global structural context by replacing the standard SwiGLU MLP with a Dual-Scale Convolutional Gating mechanism.

## 📊 Performance Summary

| Dataset | Metric | neon061 (Baseline) | **neon092 (SOTA)** | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **HP0 / Tok4** | Val Loss | 0.2364 | **0.1961** | **+17.0%** |
| **Wiki103 / Tok4** | Val Loss | 3.0940 | **3.0575** | **+1.2% (~1600 steps)** |

## 🏗️ Architectural Core

The fundamental innovation is the **Dual-Scale Hydra MLP**. Unlike a standard MLP which projects tokens pointwise, Neon092 uses two parallel depthwise convolutional heads in the gate path.

### 1. The Gating Formula
The output $y$ of the MLP is defined as:
$$y = W_2 \left( (\text{LinearGate}(x) + \text{ConvGate}(x)) \odot W_1(x) \right)$$

Where **ConvGate** is:
$$\text{ConvGate}(x) = \sigma(W_{proj}(\text{DepthwiseConv}_{k=3}(x) + \text{DepthwiseConv}_{k=9}(x)))$$

### 2. The Dual-Scale Logic
*   **Head 1 ($k=3$):** Captures local n-gram syntax. It acts as a "syntactic filter," ensuring the MLP only fires if the local word ordering (bigrams/trigrams) is valid.
*   **Head 2 ($k=9$):** Captures phrase-level structure and topical context. It allows the model to "see" up to 8 tokens into the past within the MLP layer itself, bypassing the need for an attention head to resolve small structural dependencies.

## ⚖️ Parameter Efficiency (The "Fair Fight")
Neon092 was calibrated to be a bit-for-bit exact match to the **9,718,528** non-embedding parameters of the previous record holder (`neon061`).

To pay for the convolutional heads and projections, Neon092 slightly reduced the MLP width:
- **neon061**: $d_{ff} = 2736$ (Brute-force width)
- **neon092**: $d_{ff} = 2049$ (Selective intelligence)

The fact that `092` outperforms `061` on identical parameter budgets proves that **Context Intelligence** (Gating) is a more efficient use of silicon than **Raw Capacity** (Width).

## 🔭 Visualizing the "Hydra"
When viewed in the [Neon Visualizer](scripts/visualizer.py), the $k=9$ gate often shows high activation weights for structural markers (punctuation, prepositions) while the $k=3$ gate stays active for local noun-phrase clusters. This effectively "shouts" to the model when a specific structural pattern is recognized.

---
*Developed as part of the NeonBench Scaling & Generalization Study.*
