# neon085: Dual-Scale Conv-Gated Hydra

`neon085` is the current state-of-the-art for the NeonBench project, achieving an unprecedented Validation Loss of **0.8670** on HP0 (Tok4). This represents a relative improvement of **~5.5%** over the `neon016` baseline at an identical parameter count of 2.89M.

## 🏗️ Architecture

Building on the success of `neon081`, `neon085` introduces **Multi-Scale Feature Extraction** into the MLP gating mechanism.

### The Duel-Scale Gate
Instead of a single convolution, `neon085` processes the input token stream through two parallel convolutional paths of different granularities:

$$Gate = \text{SiLU}(W_g x) + \sigma(\text{Conv1D}_{\text{k=3}}(x) + \text{Conv1D}_{\text{k=9}}(x))$$

- **Small Scale (k=3)**: Captures immediate token combinations (bigrams/trigrams).
- **Large Scale (k=9)**: Captures structural context and phrase-level patterns.
- **Why this wins**: The summation of features from both scales allows the MLP to trigger based on both local syntax and broader context simultaneously. This proved significantly more effective than "Modulation" (`neon083`) or simple "Dilation" (`neon084`).

## 📊 Parameters (The 1:1 Match)

`neon085` was precisely calibrated to be an exact bit-for-bit parameter match to the `neon016` baseline:

| Metric | Value |
| :--- | :--- |
| `d_model` | 256 |
| `n_layers` | 4 |
| `n_head` | 4 |
| `d_ff` | **381** |
| `kernel_sizes` | 3, 9 |
| **Non-Emb Params** | **2,886,400** |
| Baseline (`016`) | **2,886,400** |

This 1:1 match eliminates the "parameter starvation" variable, proving that the **Architectural Efficiency** of the multi-scale gate is the true driver of the performance gain.

## 🧪 Benchmark Comparison (Tok4)

| Model | Val Loss | delta vs Baseline |
| :--- | :--- | :--- |
| neon016 (Baseline) | 0.9174 | - |
| neon081 (k=9) | 0.8812 | -0.0362 |
| **neon085 (k=3+9)** | **0.8670** | **-0.0504** |

## 💡 Conclusion
The "Intent" era proved that gating the result of calculations is better than gating the source. The "Hydra" era is proving that **Convolution** is the most efficient way to generate that gate for small-scale language models.
