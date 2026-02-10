"""
Interactive inference/chat script for Neon models.

Usage:
    python3 scripts/inference.py --model neon001 --tokenizer tokenizers/hp_tok1.json --checkpoint checkpoints/neon001_tok1_best.pth

Enter a prompt and the model will generate the next 500 tokens.
"""

import argparse
import json
import os
import sys
import torch

sys.path.append(os.getcwd())


def load_tokenizer(tokenizer_path):
    """Load tokenizer and detect type."""
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)
    
    if tok_data.get('type') == 'word_level_pos':
        from scripts.build_warm_tokenizer import WarmTokenizer
        return WarmTokenizer(tokenizer_path), "warm"
    else:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tokenizer_path), "bpe"


def encode(tokenizer, text, tokenizer_type):
    """Encode text to token IDs."""
    if tokenizer_type == "bpe":
        return tokenizer.encode(text).ids
    else:
        return tokenizer.encode(text)


def decode(tokenizer, ids, tokenizer_type):
    """Decode token IDs to text."""
    if tokenizer_type == "bpe":
        return tokenizer.decode(ids)
    else:
        return tokenizer.decode(ids)


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50, device='cpu'):
    """
    Generate tokens autoregressively.
    
    Args:
        model: The language model
        idx: Starting token indices [1, seq_len]
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (1.0 = unchanged, <1.0 = more focused, >1.0 = more random)
        top_k: Only sample from top-k most likely tokens
        device: Device to run on
    """
    model.eval()
    block_size = model.config['block_size']
    
    for _ in range(max_new_tokens):
        # Crop to block size if needed
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Get predictions
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature  # Last position only
        
        # Top-k filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append
        idx = torch.cat([idx, idx_next], dim=1)
    
    return idx


def main():
    parser = argparse.ArgumentParser(description="Interactive inference with Neon models")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., neon001)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect if not set)")
    
    args = parser.parse_args()
    
    # Device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer, tokenizer_type = load_tokenizer(args.tokenizer)
    print(f"Tokenizer type: {tokenizer_type}")
    
    # Determine vocab size
    if tokenizer_type == "bpe":
        vocab_size = tokenizer.get_vocab_size()
    else:
        vocab_size = len(tokenizer)
    
    # Config (must match training)
    config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'n_layers': 4,
        'n_head': 4,
        'd_ff': 512,
        'block_size': 256,
    }
    if args.model == "neon004":
        config['d_ff_wide'] = 1024
    
    # Load model
    print(f"Loading model {args.model}...")
    try:
        module = __import__(f"models.{args.model}", fromlist=[args.model.capitalize()])
        ModelClass = getattr(module, args.model.capitalize())
    except (ImportError, AttributeError) as e:
        print(f"Error loading model: {e}")
        return
    
    model = ModelClass(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | Tokenizer: {tokenizer_type} | Vocab: {vocab_size}")
    print(f"Max tokens: {args.max_tokens} | Temperature: {args.temperature} | Top-k: {args.top_k}")
    print(f"{'='*60}")
    print("\nEnter a prompt (or 'quit' to exit):\n")
    
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt.strip():
            continue
        
        # Encode prompt
        input_ids = encode(tokenizer, prompt, tokenizer_type)
        if len(input_ids) == 0:
            print("(Empty encoding, try different text)")
            continue
        
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        print("\nGenerating...\n")
        print("-" * 40)
        
        # Generate
        output_ids = generate(
            model, idx, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        # Decode and print
        generated_ids = output_ids[0].tolist()
        text = decode(tokenizer, generated_ids, tokenizer_type)
        print(text)
        print("-" * 40)
        print(f"\n[Generated {len(generated_ids) - len(input_ids)} new tokens]\n")


if __name__ == "__main__":
    main()
