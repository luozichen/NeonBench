"""
Build a word-level tokenizer with POS (Part-of-Speech) metadata.

This tokenizer:
1. Uses spaCy to tokenize text into words
2. Tracks the most common POS tag for each word
3. Saves vocabulary with POS information for warm embedding initialization
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import spacy


def build_warm_tokenizer(data_path: str, vocab_size: int, save_path: str, min_freq: int = 2):
    """
    Build a word-level tokenizer with POS metadata.
    
    Args:
        data_path: Path to training text file
        vocab_size: Maximum vocabulary size
        save_path: Where to save the tokenizer JSON
        min_freq: Minimum word frequency to include
    """
    print(f"Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Only need tagger
    nlp.max_length = 10_000_000  # Allow large texts
    
    print(f"Reading {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Tokenizing with spaCy (this may take a moment)...")
    # Process in chunks to avoid memory issues
    chunk_size = 100_000
    word_freq = Counter()
    word_pos = defaultdict(Counter)  # word -> {POS: count}
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        doc = nlp(chunk)
        for token in doc:
            if token.is_alpha and len(token.text) > 0:  # Only alphabetic tokens
                word = token.text.lower()
                word_freq[word] += 1
                word_pos[word][token.pos_] += 1
    
    print(f"Total unique words: {len(word_freq)}")
    
    # Filter by frequency and limit vocab size
    # Reserve first 5 slots for special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    available_slots = vocab_size - len(special_tokens)
    
    # Get most common words that meet frequency threshold
    filtered_words = [(w, c) for w, c in word_freq.most_common() if c >= min_freq]
    top_words = filtered_words[:available_slots]
    
    print(f"Vocabulary after filtering (min_freq={min_freq}): {len(top_words)} words")
    
    # Build vocabulary
    vocab = {}
    
    # Add special tokens first
    for i, token in enumerate(special_tokens):
        vocab[token] = {
            "id": i,
            "pos": "SPECIAL",
            "freq": 0
        }
    
    # Add regular words
    for idx, (word, freq) in enumerate(top_words):
        # Get most common POS for this word
        pos_counts = word_pos[word]
        most_common_pos = pos_counts.most_common(1)[0][0] if pos_counts else "X"
        
        vocab[word] = {
            "id": idx + len(special_tokens),
            "pos": most_common_pos,
            "freq": freq
        }
    
    # Create the tokenizer config
    tokenizer_config = {
        "version": "1.0",
        "type": "word_level_pos",
        "vocab_size": len(vocab),
        "special_tokens": special_tokens,
        "vocab": vocab,
        # Reverse mapping for decoding
        "id_to_token": {v["id"]: k for k, v in vocab.items()}
    }
    
    # Count POS distribution
    pos_dist = Counter(v["pos"] for v in vocab.values() if v["pos"] != "SPECIAL")
    print(f"\nPOS Distribution:")
    for pos, count in pos_dist.most_common():
        print(f"  {pos}: {count}")
    
    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"\nSuccess! Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {len(vocab)}")


class WarmTokenizer:
    """Simple word-level tokenizer that loads from saved JSON."""
    
    def __init__(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.vocab = config["vocab"]
        self.id_to_token = {int(k): v for k, v in config["id_to_token"].items()}
        self.unk_id = self.vocab["[UNK]"]["id"]
        self.pad_id = self.vocab["[PAD]"]["id"]
    
    def encode(self, text: str) -> list:
        """Encode text to token IDs."""
        words = text.lower().split()
        ids = []
        for word in words:
            # Strip punctuation from word edges
            word_clean = word.strip(".,!?;:'\"()-")
            if word_clean in self.vocab:
                ids.append(self.vocab[word_clean]["id"])
            else:
                ids.append(self.unk_id)
        return ids
    
    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(i, "[UNK]") for i in ids]
        return " ".join(tokens)
    
    def get_pos(self, token: str) -> str:
        """Get POS tag for a token."""
        if token in self.vocab:
            return self.vocab[token]["pos"]
        return "X"  # Unknown
    
    def __len__(self):
        return len(self.vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build word-level tokenizer with POS metadata")
    parser.add_argument("--data", type=str, required=True, help="Path to training data text file")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Maximum vocabulary size")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum word frequency")
    parser.add_argument("--save_path", type=str, default="tokenizers/hp_warm.json", help="Where to save tokenizer")
    
    args = parser.parse_args()
    build_warm_tokenizer(args.data, args.vocab_size, args.save_path, args.min_freq)
