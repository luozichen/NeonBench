# neon081: Context-Scaled Conv-Gated Hydra

`neon081` represents a significant technical breakthrough in the NeonBench project, achieving a Validation Loss of **0.8812** on HP0 (Tok4) at the ~3M parameter scale. This decisively outperforms the `neon016` (Learned Intent) baseline of **0.9174** while using slightly fewer parameters.

## 🏗️ Architecture

The model is built on the **Hydra MLP** architecture, where the standard SwiGLU transition is gated by a context-aware mechanism.

### The Gating Mechanism
The core innovation in `neon081` is the expansion of the convolutional receptive field within the MLP gate.

$$Gate = \text{SiLU}(W_g x) + \sigma(\text{Conv1D}_{\text{k=9}}(x))$$

- **Linear Path**: Provides a per-token semantic gate (standard SiLU).
- **Convolutional Path**: Uses a **Kernel Size of 9** depthwise convolution to provide local temporal context.
- **Why k=9?**: The scaling study (neon080-082) proved that `neon081` (k=9) outperformed `neon080` (k=3, wider MLP) and `neon082` (Attention gating), indicating that the MLP is primarily "starved" for structural context rather than raw mapping width.

## 📊 Parameters (Fair Fight)

To ensure a 1:1 comparison with the legacy `neon016` baseline, `neon081` was precisely calibrated:

| Metric | value |
| :--- | :--- |
| `d_model` | 256 |
| `n_layers` | 4 |
| `n_head` | 4 |
| `d_ff` | 378 |
| `kernel_size` | 9 |
| **Non-Emb Params** | **2,871,040** |
| Baseline (`016`) | 2,886,400 |

Despite having ~15k fewer parameters than the baseline, `neon081` achieved a **~4% relative improvement** in validation loss.

## 🧪 Benchmark Results (Tok4)

| Model | Val Loss | Status |
| :--- | :--- | :--- |
| neon016 (Baseline) | 0.9174 | Legacy SOTA |
| neon077 (k=3) | 0.9172 | Baseline Match |
| neon080 (Width) | 0.8875 | Improved |
| **neon081 (Context)** | **0.8812** | **Technical SOTA** |

## 💡 Key Takeaway
At the 3M parameter scale, broad convolutional context in the MLP gate is a more efficient use of the parameter budget than adding more attention heads or increasing the hidden dimension.
