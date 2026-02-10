"""Pre-tokenize a text file and save as binary .bin file for fast loading.
Usage: python3 scripts/pretokenize.py --data data/wiki103/wiki103.txt --tokenizer tokenizers/wiki103_tok1.json --output data/wiki103/wiki103_tok1.bin
"""
import argparse
import os
import struct
import numpy as np
from tokenizers import Tokenizer

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize text to binary")
    parser.add_argument("--data", type=str, required=True, help="Input text file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer JSON")
    parser.add_argument("--output", type=str, required=True, help="Output .bin file")
    parser.add_argument("--batch_size", type=int, default=1000, help="Lines per batch")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Count lines first
    print(f"Counting lines in {args.data}...")
    total_lines = 0
    with open(args.data, 'r', encoding='utf-8') as f:
        for _ in f:
            total_lines += 1
    print(f"Total lines: {total_lines:,}")

    # Tokenize in batches of lines using encode_batch (parallel Rust)
    all_ids = []
    processed = 0

    with open(args.data, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)

            if len(batch) >= args.batch_size:
                encoded = tokenizer.encode_batch(batch)
                for enc in encoded:
                    all_ids.extend(enc.ids)
                processed += len(batch)
                batch = []
                if processed % 50000 == 0:
                    print(f"  Processed {processed:,}/{total_lines:,} lines ({100*processed/total_lines:.1f}%) — {len(all_ids):,} tokens")

        # Last batch
        if batch:
            encoded = tokenizer.encode_batch(batch)
            for enc in encoded:
                all_ids.extend(enc.ids)
            processed += len(batch)

    print(f"\nTotal tokens: {len(all_ids):,}")

    # Save as uint16 numpy array (vocab_size < 65536)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(args.output)

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Saved to {args.output} ({file_size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
