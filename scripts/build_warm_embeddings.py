"""
Build warm-initialized embeddings using POS prototypes and optional GloVe vectors.

This script:
1. Loads a warm tokenizer with POS metadata
2. Creates embedding vectors initialized based on POS type
3. Optionally incorporates GloVe vectors for known words
4. Saves the embedding matrix as a .pt file
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path


# POS tag groupings for prototype initialization
POS_GROUPS = {
    # Content words (open class)
    "NOUN": {"prototype_id": 0, "description": "Nouns"},
    "VERB": {"prototype_id": 1, "description": "Verbs"},
    "ADJ": {"prototype_id": 2, "description": "Adjectives"},
    "ADV": {"prototype_id": 3, "description": "Adverbs"},
    "PROPN": {"prototype_id": 0, "description": "Proper nouns, same as NOUN"},
    
    # Function words (closed class)
    "DET": {"prototype_id": 4, "description": "Determiners"},
    "PRON": {"prototype_id": 5, "description": "Pronouns"},
    "ADP": {"prototype_id": 6, "description": "Adpositions (prepositions)"},
    "CONJ": {"prototype_id": 7, "description": "Conjunctions"},
    "CCONJ": {"prototype_id": 7, "description": "Coordinating conjunctions"},
    "SCONJ": {"prototype_id": 7, "description": "Subordinating conjunctions"},
    "AUX": {"prototype_id": 1, "description": "Auxiliaries, same as VERB"},
    
    # Other
    "NUM": {"prototype_id": 8, "description": "Numerals"},
    "PART": {"prototype_id": 9, "description": "Particles"},
    "INTJ": {"prototype_id": 10, "description": "Interjections"},
    "PUNCT": {"prototype_id": 11, "description": "Punctuation"},
    "SYM": {"prototype_id": 11, "description": "Symbols"},
    "X": {"prototype_id": 12, "description": "Other/Unknown"},
    "SPECIAL": {"prototype_id": 13, "description": "Special tokens"},
}


def load_glove(glove_path: str, dim: int = 100) -> dict:
    """Load GloVe vectors from file."""
    print(f"Loading GloVe from {glove_path}...")
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            if len(vec) == dim:
                embeddings[word] = vec
    print(f"Loaded {len(embeddings)} GloVe vectors")
    return embeddings


def create_pos_prototypes(d_model: int, n_prototypes: int = 14, seed: int = 42) -> torch.Tensor:
    """
    Create orthogonal prototype vectors for each POS category.
    Uses QR decomposition to ensure prototypes are orthogonal.
    """
    torch.manual_seed(seed)
    
    # Create more vectors than we need, then orthogonalize
    random_matrix = torch.randn(d_model, max(n_prototypes, d_model))
    q, _ = torch.linalg.qr(random_matrix)
    
    # Take first n_prototypes and scale
    prototypes = q[:, :n_prototypes].T * 0.5  # Shape: [n_prototypes, d_model]
    
    return prototypes


def project_glove_to_dim(glove_vec: np.ndarray, d_model: int, projection_matrix: np.ndarray) -> np.ndarray:
    """Project GloVe vector to target dimension."""
    return glove_vec @ projection_matrix


def build_warm_embeddings(
    tokenizer_path: str,
    d_model: int = 256,
    glove_path: str = None,
    save_path: str = "tokenizers/hp_warm_emb.pt",
    seed: int = 42
):
    """
    Build warm-initialized embedding matrix.
    
    Args:
        tokenizer_path: Path to warm tokenizer JSON
        d_model: Embedding dimension
        glove_path: Optional path to GloVe file
        save_path: Where to save the embedding .pt file
        seed: Random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    
    vocab = tokenizer_config["vocab"]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create POS prototypes
    print("Creating POS prototype vectors...")
    prototypes = create_pos_prototypes(d_model)
    
    # Load GloVe if provided
    glove_embeddings = None
    projection_matrix = None
    if glove_path and os.path.exists(glove_path):
        glove_dim = 100  # Assuming GloVe 100d
        glove_embeddings = load_glove(glove_path, dim=glove_dim)
        # Create random projection matrix from glove_dim to d_model
        projection_matrix = np.random.randn(glove_dim, d_model) / np.sqrt(glove_dim)
    
    # Build embedding matrix
    embeddings = torch.zeros(vocab_size, d_model)
    
    glove_hits = 0
    pos_fallback = 0
    
    for token, info in vocab.items():
        token_id = info["id"]
        pos = info["pos"]
        
        # Get prototype ID for this POS
        pos_info = POS_GROUPS.get(pos, POS_GROUPS["X"])
        prototype_id = pos_info["prototype_id"]
        
        # Start with POS prototype
        base_embedding = prototypes[prototype_id].clone()
        
        # Try to incorporate GloVe if available
        if glove_embeddings is not None and token.lower() in glove_embeddings:
            glove_vec = glove_embeddings[token.lower()]
            projected = project_glove_to_dim(glove_vec, d_model, projection_matrix)
            # Blend: 70% GloVe, 30% POS prototype
            base_embedding = 0.3 * base_embedding + 0.7 * torch.from_numpy(projected).float()
            glove_hits += 1
        else:
            # Add small noise to differentiate tokens with same POS
            noise = torch.randn(d_model) * 0.1
            base_embedding = base_embedding + noise
            pos_fallback += 1
        
        embeddings[token_id] = base_embedding
    
    # Normalize embeddings
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
    embeddings = embeddings * np.sqrt(d_model)  # Scale to typical init magnitude
    
    # Save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(embeddings, save_path)
    
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  GloVe coverage: {glove_hits}/{vocab_size} ({100*glove_hits/vocab_size:.1f}%)")
    print(f"  POS-only fallback: {pos_fallback}/{vocab_size} ({100*pos_fallback/vocab_size:.1f}%)")
    print(f"  Mean norm: {embeddings.norm(dim=1).mean():.3f}")
    print(f"\nSuccess! Embeddings saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build warm-initialized embeddings")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to warm tokenizer JSON")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--glove", type=str, default=None, help="Optional path to GloVe file (e.g., glove.6B.100d.txt)")
    parser.add_argument("--save_path", type=str, default="tokenizers/hp_warm_emb.pt", help="Where to save embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    build_warm_embeddings(
        tokenizer_path=args.tokenizer,
        d_model=args.d_model,
        glove_path=args.glove,
        save_path=args.save_path,
        seed=args.seed
    )
