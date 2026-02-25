#!/bin/bash
# Phase 12: Pure Progressive Lookahead (neon259-260)
# Uses the Wiki103 / Tok5 dataset.

echo "Starting Phase 12: Pure Progressive Lookahead Evaluation Run..."

python3 train_parity.py --model neon259 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json
python3 train_parity.py --model neon260 --data data/wiki103/wiki103_tok5.bin --tokenizer tokenizers/wiki103_tok5.json

echo "Phase 12 Complete!"
