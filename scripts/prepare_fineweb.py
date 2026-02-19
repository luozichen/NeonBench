"""Prepare FineWeb-Edu parquet file into a single .txt file for training.
Usage: python3 scripts/prepare_fineweb.py --parquet data/fineweb/000_00000.parquet --output data/fineweb/fineweb.txt
"""
import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Convert FineWeb parquet to text")
    parser.add_argument("--parquet", type=str, default="data/fineweb/000_00000.parquet",
                        help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/fineweb/fineweb.txt",
                        help="Output text file path")
    args = parser.parse_args()

    if not os.path.exists(args.parquet):
        print(f"Error: {args.parquet} not found.")
        return

    print(f"Reading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    
    print(f"Found {len(df):,} rows.")
    
    # FineWeb-Edu uses 'text' column
    if 'text' not in df.columns:
        print(f"Error: 'text' column not found in parquet. Columns: {df.columns}")
        return

    print(f"Writing to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, text in enumerate(df['text']):
            text = str(text).strip()
            if text:
                f.write(text + '\n')
            if (i+1) % 100000 == 0:
                print(f"  Processed {i+1:,} rows...")

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone! Wrote {file_size_mb:.1f} MB to {args.output}")

if __name__ == "__main__":
    main()
