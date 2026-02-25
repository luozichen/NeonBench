#!/bin/bash
# Phase 11: Wide Convolution & Stochastic Dropout (neon257-258)

echo "Starting Phase 11 Evaluation Run..."

python3 train_parity.py --model neon257 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json
python3 train_parity.py --model neon258 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json

echo "Phase 11 Complete!"
