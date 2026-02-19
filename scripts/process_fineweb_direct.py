"""Process FineWeb-Edu parquet directly to Tokenizer/Bin (Zero-Disk-Text).
1. Loads parquet into RAM.
2. Trains 16k tokenizer from memory.
3. Pre-tokenizes to binary file.
Avoids creating massive intermediate .txt files.
"""
import os
import argparse
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def get_training_corpus(text_list, batch_size=1000):
    for i in range(0, len(text_list), batch_size):
        yield text_list[i : i + batch_size]

def main():
    parser = argparse.ArgumentParser(description="Process FineWeb directly to bin")
    parser.add_argument("--parquet", type=str, default="data/fineweb/000_00000.parquet", help="Input parquet")
    parser.add_argument("--tokenizer_save", type=str, default="tokenizers/fineweb_tok6.json", help="Tokenizer output")
    parser.add_argument("--bin_save", type=str, default="data/fineweb/fineweb_tok6.bin", help="Binary output")
    parser.add_argument("--vocab_size", type=int, default=16384, help="Vocab size")
    args = parser.parse_args()

    # 1. Load Parquet
    print(f"Loading {args.parquet}...")
    if not os.path.exists(args.parquet):
        print("Error: Parquet file not found.")
        return
    
    df = pd.read_parquet(args.parquet)
    if 'text' not in df.columns:
        print(f"Error: 'text' column missing. Columns: {df.columns}")
        return
    
    # Extract text list
    texts = [str(t).strip() for t in df['text'] if str(t).strip()]
    print(f"Loaded {len(texts):,} documents.")

    # 2. Train Tokenizer
    print("Training Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    tokenizer.train_from_iterator(get_training_corpus(texts), trainer=trainer)
    
    os.makedirs(os.path.dirname(args.tokenizer_save), exist_ok=True)
    tokenizer.save(args.tokenizer_save)
    print(f"Tokenizer saved to {args.tokenizer_save}")

    # 3. Pre-tokenize to Binary
    print("Pre-tokenizing to binary...")
    all_ids = []
    batch_size = 1000
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer.encode_batch(batch)
        for enc in encoded:
            all_ids.extend(enc.ids)
            
        if (i + batch_size) % 50000 == 0:
            print(f"  Processed {i + batch_size:,}/{total:,} docs...")

    print(f"Total tokens: {len(all_ids):,}")
    
    # Save as uint16
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(args.bin_save)
    
    file_size_mb = os.path.getsize(args.bin_save) / (1024 * 1024)
    print(f"Saved {file_size_mb:.1f} MB to {args.bin_save}")

if __name__ == "__main__":
    main()
