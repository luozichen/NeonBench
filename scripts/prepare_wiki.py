"""Prepare WikiText-103 parquet files into a single .txt file for training."""
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert WikiText-103 parquet to text")
    parser.add_argument("--data_dir", type=str, default="data/wiki103",
                        help="Directory containing parquet files")
    parser.add_argument("--output", type=str, default="data/wiki103/wiki103.txt",
                        help="Output text file path")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "validation", "all"],
                        help="Which split to use")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        print("pandas not found. Install with: pip3 install pandas pyarrow")
        return

    parquet_files = []
    for f in sorted(os.listdir(args.data_dir)):
        if not f.endswith('.parquet'):
            continue
        if args.split != "all" and not f.startswith(args.split):
            continue
        parquet_files.append(os.path.join(args.data_dir, f))

    if not parquet_files:
        print(f"No parquet files found for split '{args.split}' in {args.data_dir}")
        return

    print(f"Found {len(parquet_files)} parquet file(s):")
    for f in parquet_files:
        print(f"  {f}")

    all_text = []
    total_rows = 0
    for f in parquet_files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        # WikiText-103 uses a 'text' column
        col = 'text' if 'text' in df.columns else df.columns[0]
        for text in df[col]:
            text = str(text).strip()
            if text and text != '':
                all_text.append(text)

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_text))

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {len(all_text)} paragraphs ({total_rows} rows) to {args.output}")
    print(f"File size: {file_size_mb:.1f} MB")

if __name__ == "__main__":
    main()
