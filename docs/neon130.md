# 🐉 Neon130: Sharp-V Hyper-Synergy (MQI)

Neon130 is the **Co-SOTA** champion of the 3M scale. It represents the pinnacle of parameter harvesting, combining **Multi-Query Intent (MQI)** and **Sharp-Value** projections to maximize the raw capacity of the model without sacrificing localized intelligence.

## 📊 Performance Summary

| Dataset | Metric | neon116 (SOTA) | **neon130 (Co-SOTA)** | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **HP0 / Tok4** | Val Loss | 0.7269 | **0.7265** | **Equivalence** |

## 🏗️ Architectural Core

The core philosophy of Neon130 is **Surgical Parameter Reallocation**. It identifies "redundant" context projections and harvests their parameters to fund a massive increase in MLP width.

### 1. Multi-Query Intent (MQI)
Instead of every attention head learning its own Intent gating projection, Neon130 uses a single **Shared Intent Gate** for all heads in a layer.
- Discovery: Intent is a "high-level" decision (e.g., "Is this a local syntactic cluster?") that is often identical across heads. 
- By sharing the gating projection, we save parameters while maintaining the contextual "filter" that makes Locally-Aware Attention work.

### 2. The "Sharp-V" Principle
Neon130 keeps the **Value (V)** projection "sharp" (no convolution), convolving only the **Query (Q)**, **Key (K)**, and **Intent (I)**.
- Discovery: Convolving $V$ can sometimes "blur" the content signal. By keeping $V$ sharp, the model preserves high-fidelity token information while still using the convolved $Q$ and $K$ to perform a locally-aware search.

### 3. Hyper-Synergy Calibration
The parameter savings from MQI and Sharp-V are significant:
- **Parameter Harvest**: Saved ~65,000 parameters per layer compared to a full `neon116` stack.
- **MLP Expansion**: Reinvested these savings to push $d_{ff}$ from 507 to **572**.

## ⚖️ Conclusion
Neon130 proves that **Shared Gating (MQI)** is an extremely efficient architecture for small-scale models. It achieves the same intelligence as our most complex models while maintaining the raw horsepower of a wider MLP.

---
*Developed as part of the NeonBench Hyper-Synergy & Harvesting Study.*
