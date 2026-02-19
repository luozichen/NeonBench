"""Inspect random tokens from a binary dataset file.
Usage: python3 scripts/inspect_tokens.py --bin data/fineweb/fineweb_tok6.bin --tokenizer tokenizers/fineweb_tok6.json
"""
import argparse
import os
import numpy as np
from tokenizers import Tokenizer
import random

def main():
    parser = argparse.ArgumentParser(description="Inspect random tokens")
    parser.add_argument("--bin", type=str, required=True, help="Input .bin file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer JSON")
    parser.add_argument("--count", type=int, default=100, help="Number of tokens to show")
    args = parser.parse_args()

    if not os.path.exists(args.bin):
        print(f"Error: {args.bin} not found.")
        return
    
    # Get file size to pick random offset
    file_size_bytes = os.path.getsize(args.bin)
    total_tokens = file_size_bytes // 2  # uint16 is 2 bytes
    print(f"Total tokens in file: {total_tokens:,}")
    
    start = 0
    if total_tokens > args.count:
        start = random.randint(0, total_tokens - args.count)
    
    print(f"Reading {args.count} tokens starting at index {start:,}...")
    
    # Read tokens
    # offset in bytes = start * 2
    with open(args.bin, 'rb') as f:
        f.seek(start * 2)
        data = f.read(args.count * 2)
        tokens = np.frombuffer(data, dtype=np.uint16)
        
    print(f"Token IDs: {tokens.tolist()}")
    
    # Decode
    tokenizer = Tokenizer.from_file(args.tokenizer)
    decoded = tokenizer.decode(tokens)
    
    print("\n--- Decoded Text ---")
    print(decoded)
    print("--------------------")
    
    print("\n--- Individual Token Splits ---")
    for i, t in enumerate(tokens):
        s = tokenizer.decode([t])
        print(f"{t:5d} -> '{s}'")

if __name__ == "__main__":
    main()
