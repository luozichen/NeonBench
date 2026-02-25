#!/bin/bash
# Phase 10: Layer-Specific Lookahead (neon253-256)

echo "Starting Phase 10 Evaluation Run..."

python3 train_parity.py --model neon253
python3 train_parity.py --model neon254
python3 train_parity.py --model neon255
python3 train_parity.py --model neon256

echo "Phase 10 Complete!"
