# Training Instructions (NeonBench)

This document describes how to train all Neon models.

---

## Model Overview

### 2.9M Models (neon001-010) — Architecture Experiments
| Model | Base | Key Change |
|-------|------|------------|
| **neon001** | Baseline | MHA + LayerNorm + GELU |
| **neon002** | Modern | MHA + RMSNorm + QK-Norm + GELU |
| **neon003** | neon002 | MQA (shared KV) + GELU |
| **neon004** | neon003 | Shared wide MLP + GELU |
| **neon005** | neon002 | SwiGLU (replaces GELU) |
| **neon006** | neon005 | MLA (Multi-head Latent Attention) |
| **neon007** | neon005 | DeltaNet (linear attention + delta rule) ⚠️ slow |
| **neon008** | neon005 | L2 Norm + LayerNorm (QM-inspired) |
| **neon009** | neon005 | QKVI Intent Attention |
| **neon010** | neon005 | Gated SDPA (gate from Q) |

### 10M Models (neon011-014) — Scaling Experiments
All use the neon010 architecture (Gated SDPA + SwiGLU). Only config differs.

| Model | Strategy | d_model | Layers | Heads | d_ff | MLP Ratio |
|-------|----------|---------|--------|-------|------|-----------|
| **neon011** | Narrow & Deep | 384 | 8 | 6 | 768 | 2× |
| **neon012** | Wide & Medium | 512 | 6 | 8 | 1024 | 2× |
| **neon013** | Balanced | 320 | 8 | 8 | 640 | 2× |
| **neon014** | MLP-Heavy | 384 | 6 | 6 | 1536 | 4× |

---

## 1. Prepare Data

### Harry Potter (small, ~1.5M tokens)
Already available at `data/hp/hp0.txt`.

### WikiText-103 (large, ~100M tokens)
Convert parquet files to text:
```bash
# Install pandas + pyarrow if needed
pip3 install pandas pyarrow

# Convert training split to text
python3 scripts/prepare_wiki.py --data_dir data/wiki103 --split train --output data/wiki103/wiki103.txt
```

---

## 2. Build Tokenizers

```bash
# HP tokenizer (BPE, vocab 1024)
python3 scripts/build_tokenizer.py --data data/hp/hp0.txt --vocab_size 1024 --save_path tokenizers/hp_tok1.json

# WikiText tokenizer (BPE, vocab 1024)
python3 scripts/build_tokenizer.py --data data/wiki103/wiki103.txt --vocab_size 1024 --save_path tokenizers/wiki103_tok1.json
```

---

## 3. Parameter Count

```bash
python3 scripts/count_params.py
```

---

## 4. Train Models

### Train 2.9M models on Harry Potter
```bash
for model in neon001 neon002 neon003 neon004 neon005 neon006 neon007 neon008 neon009 neon010; do
    python3 train.py --model $model --data data/hp/hp0.txt --tokenizer tokenizers/hp_tok1.json --tok_name tok1
done
```

> **Note:** neon007 (DeltaNet) is ~30× slower due to sequential recurrence.

### Test: Train neon010 on WikiText-103
Run this first to verify the pipeline works before the expensive 10M models:
```bash
python3 train.py --model neon010 --data data/wiki103/wiki103.txt --tokenizer tokenizers/wiki103_tok1.json --tok_name tok1
```

### Train 10M models on WikiText-103
```bash
for model in neon011 neon012 neon013 neon014; do
    python3 train.py --model $model --data data/wiki103/wiki103.txt --tokenizer tokenizers/wiki103_tok1.json --tok_name tok1
done
```

### Output Files
- **Log:** `logs/{model}_tok1_{data}_log.txt`
- **Checkpoint:** `checkpoints/{model}_tok1_{data}_best.pth`

---

## 5. Plot Results

```bash
# Plot all 2.9M models (HP)
python3 scripts/plot_results.py --data_name hp0

# Plot 10M models (WikiText)
python3 scripts/plot_results.py --data_name wiki103 --models neon011,neon012,neon013,neon014

# Compare neon010 across HP and WikiText
python3 scripts/plot_results.py --data_name wiki103 --models neon010,neon011,neon012,neon013,neon014
```

---

## 6. Interactive Inference

```bash
python3 scripts/inference.py \
    --model neon010 \
    --tokenizer tokenizers/hp_tok1.json \
    --checkpoint checkpoints/neon010_tok1_hp0_best.pth
```

| Flag | Default | Description |
|------|---------|-------------|
| `--max_tokens` | 500 | Max tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | 50 | Top-k sampling |
