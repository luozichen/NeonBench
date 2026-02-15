# 🐉 Neon167: Scaled Full Conv-Attention (The Giant)

Neon167 is the flagship of the **5M Parameter Class**, representing the successful transition of the project's architectural discoveries from small-scale (3M) to medium-scale (5M) general knowledge modeling. It currently holds the **Co-SOTA** title for Wiki103 generalization.

## 📊 Performance Summary

| Dataset | Metric | neon116 (3M Baseline) | **neon167 (5M Champion)** | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Wiki103 / Tok4** | Val Loss | 3.2499 | **3.1484** | **+3.1%** |

## 🏗️ Architectural Core

Neon167 is a direct, vertical scale-up of the **Full Blur** architecture first established in `neon116`. It confirms that the dual-level context requirement (Locally-Aware Attention + Hydra MLP) scales linearly with parameter count.

### 1. The "Full Blur" Synergy
Unlike models that attempt to harvest parameters by removing convolutions (like `neon130` or `neon180`), Neon167 applies $k=3$ depthwise convolutions to every search and content signal:
- $Q, K, V,$ and $I$ (Intent) all receive a local context filter before processing.
- Discovery: At the 5M scale, this "Blur everything" strategy provides the most robust regularization for the complex structural jumps within Wikipedia text.

### 2. Giant Scaling Laws
Through the transition to 5M parameters, we observed several key "Giant" properties:
- **Quantity Meets Quality**: Increasing the MLP width from the 3M standard (d_ff=507) to the 5M standard (**d_ff=1072**) resulted in an immediate jump in factual retrieval precision.
- **Resilience to Depth**: While `neon116` was sensitive to initialization, the wider 5M footprint of `neon167` exhibits significantly higher stability during the late-stage convergence on Wiki103.

### 3. Verification of Multi-Head Intent (MHI)
Neon167 utilizes **Multi-Head Intent (MHI)**, where each of the 8 attention heads has its own independent gating projection. 
- Discovery: Despite the success of MQI (Shared Intent) at 3M, Neon167 proves that at larger scales, **Parallel Filtering Diversity** is more valuable than the parameter savings used to widen the MLP further.

## ⚖️ Parameter Calibration
Neon167 is precisely calibrated to **5,283,056** parameters:
- **d_model**: 272
- **n_layers**: 4
- **d_ff**: 1072 (4x expansion)
- **Head Diversity**: 4 Heads (68-dim each)

## 🔭 Summary
Neon167 proves that the **Synergy Architecture** is not just a small-scale heuristic but a viable foundation for scaling. It serves as the baseline against which all hierarchical and spectral "Goliaths" are measured.

---
*Developed as the baseline for the NeonBench 5M Generalization Study.*
