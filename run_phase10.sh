#!/bin/bash
# Phase 10: Layer-Specific Lookahead (neon253-256)

echo "Starting Phase 10 Evaluation Run..."

python3 train_parity.py --model neon253 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json
python3 train_parity.py --model neon254 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json
python3 train_parity.py --model neon255 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json
python3 train_parity.py --model neon256 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json

echo "Phase 10 Complete!"
