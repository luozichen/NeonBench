import argparse
import json
import os
import sys
import torch
from tokenizers import Tokenizer
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

def get_config(model_name):
    config = {
        'vocab_size': 1024,
        'd_model': 256,
        'n_layers': 4,
        'n_head': 4,
        'd_ff': 512,
        'block_size': 256,
        'learning_rate': 1e-3,
        'max_iters': 10000,
        'eval_interval': 500,
        'batch_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if model_name == "neon004":
        config['d_ff_wide'] = 1024

    if model_name == "neon006":
        config['d_latent'] = 128  # MLA latent dimension

    # --- 10M scale models (neon011-014) ---
    if model_name == "neon011":  # Narrow & Deep
        config.update({'d_model': 384, 'n_layers': 8, 'n_head': 6, 'd_ff': 768})
    elif model_name == "neon012":  # Wide & Medium
        config.update({'d_model': 512, 'n_layers': 6, 'n_head': 8, 'd_ff': 1024})
    elif model_name == "neon013":  # Balanced
        config.update({'d_model': 320, 'n_layers': 8, 'n_head': 8, 'd_ff': 640})
    elif model_name == "neon014":  # MLP-Heavy (4× expansion)
        config.update({'d_model': 384, 'n_layers': 6, 'n_head': 6, 'd_ff': 1536})

    # --- 8-layer deep intent models (neon023-024) ---
    if model_name == "neon023":  # 8-layer neon016, no layerdrop
        config.update({'n_layers': 8, 'layerdrop': 0.0})
    elif model_name == "neon024":  # 8-layer neon016, layerdrop=0.1
        config.update({'n_layers': 8, 'layerdrop': 0.1})

    # neon025: Post-Norm neon016 (uses default 4-layer config, no overrides needed)

    # --- 3M fair comparison models (neon026-030): d_ff scaled up ---
    if model_name == "neon026":    # neon005 baseline scaled
        config.update({'d_ff': 598})
    elif model_name == "neon027":  # neon010 Gated SDPA scaled
        config.update({'d_ff': 592})
    elif model_name == "neon028":  # neon006 MLA scaled
        config.update({'d_ff': 640})
    elif model_name == "neon029":  # neon001 GPT-2 scaled
        config.update({'d_ff': 891})
    elif model_name == "neon030":  # neon002 RMSNorm scaled
        config.update({'d_ff': 896})
    # --- Calculated Intent models (neon031-040): use default config ---
    # All have 3*d_model c_attn (no learned I), intent computed from Q/K/V

    # --- Gated Calculated Intent models (neon041-050): use default config ---
    # Same formulas as neon031-040 + tiny W_g linear gate

    # neon051: Linear combination intent (w_q Q + w_k K + w_v V + b)
    # neon052: Matrix combination intent (Q W_q + K W_k + V W_v + b)
    # neon053: IntentAttention with SiLU gating (instead of Sigmoid)
    # neon054: IntentAttention with SiLU (based on neon046 formula)
    # neon055: neon046 scaled up (d_ff=592)
    # neon056-060: Failed Calculated Intent Experiments
    # neon061: Wide MLP ("Stable Winner"), 16x expansion
    # neon062: MLP-Free (Attention only, 2x layers)
    # neon063: Att-MLP (Attention inside MLP slot)
    # neon064: Hadamard Head Merge (8 heads -> 4 merged)
    # neon065: Big Single Head (Head dim = 2 * d_model)
    
    all_models = [f"neon{i:03d}" for i in range(1, 80)]
    if model_name in all_models:
        if model_name in ['neon055', 'neon056', 'neon057', 'neon059', 'neon060']:
            config['d_ff'] = 592
        
        # Frankenstein Configs
        if model_name == "neon061":
            config['d_ff'] = 2736 # ~10x d_model (4x standard SwiGLU)
        elif model_name == "neon062":
            config['n_layers'] = 8 # Double layers because no MLP
        elif model_name == "neon064":
            config['n_head'] = 8
        elif model_name == "neon065":
            config['n_head'] = 1 # Single head
        elif model_name == "neon066":
            config['n_head'] = 1
            # config['d_ff'] = 512 # Default is used, no override needed.
        elif model_name == "neon066":
            config['n_head'] = 1
            # config['d_ff'] = 512 # Default is used, no override needed.
        elif model_name == "neon067":
            config['n_head'] = 2
        elif model_name == "neon068":
            config['n_head'] = 8
        elif model_name == "neon069":
            config['n_head'] = 16
        elif model_name == "neon070":
            config['d_ff'] = 576 # Increased from 512 to balance cost of Hydra Gate (-49k +32k vs -131k)
        elif model_name == "neon071":
            config['d_ff'] = 640 # Wide Hydra: ~3.3M Params
        elif model_name == "neon072":
            config['d_ff'] = 512 # Gated-Residual: Has both gates, keep d_ff standard. Prams ~3.3M? No.
            # Linear Gate (131k) + Hydra (82k). Total ~213k gate cost. 
            # Standard is 131k. Delta +82k per layer. +320k total.
            # 512 is fine. It will be slightly larger.
        elif model_name == "neon073":
            config['d_ff'] = 576 # Multi-Head Hydra. Attn cost similar (16*4 = 64). Same scaling.
        elif model_name == "neon074":
            config['d_ff'] = 576 # Swish Hydra. Same cost.
        elif model_name == "neon075":
            config['d_ff'] = 576 # Negative Hydra. Same cost.
        elif model_name == "neon076":
            config['d_model'] = 240 # Reduce model dim
            config['block_size'] = 256
            config['d_ff'] = 480 # Target 3.15M
            config['n_head'] = 4 
        elif model_name == "neon077":
            config['d_ff'] = 368 # Target 3.15M
        elif model_name == "neon078":
            config['n_layers'] = 4 
            config['d_ff'] = 500 # Adjusted to match 3.15M params.
        elif model_name == "neon079":
            config['n_layers'] = 4 
            # Qwen3Next Layers are heavy.
            # In_proj_qkvz (4*d^2) + In_proj_ba (2*d*h) + Out_proj (d^2) + Conv (3*d) + Linears in Attn.
            # Standard Attn: 4*d^2.
            # DeltaNet: ~5*d^2.
            # So similar to neon078 logic.
            # Let's start with d_ff = 480.
            config['d_ff'] = 480 

        return config
            
        return config
    else:
        raise ValueError(f"Unknown model: {model_name}")

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size, tokenizer_type="bpe"):
        import numpy as np

        if data_path.endswith('.bin'):
            # Load pre-tokenized binary file (from scripts/pretokenize.py)
            arr = np.fromfile(data_path, dtype=np.uint16)
            self.data = torch.from_numpy(arr.astype(np.int64))
            print(f"Loaded {len(self.data):,} pre-tokenized tokens from {data_path}")
        else:
            # Tokenize text in chunks
            CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB per chunk
            all_ids = []
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    if tokenizer_type == "bpe":
                        encoded = tokenizer.encode(chunk)
                        all_ids.extend(encoded.ids)
                    else:
                        ids = tokenizer.encode(chunk)
                        all_ids.extend(ids)
            self.data = torch.tensor(all_ids, dtype=torch.long)
            print(f"Tokenized {len(self.data):,} tokens from {data_path}")
        
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) tokens
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def estimate_loss(model, dataloader, device, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for i, (X, Y) in enumerate(dataloader):
            if i >= eval_iters: break
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses[i] = loss.item()
    model.train()
    return losses.mean()

def main():
    parser = argparse.ArgumentParser(description="Train Neon models")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., neon001)")
    parser.add_argument("--data", type=str, required=True, help="Path to data file (e.g., data/hp/hp1.txt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/hp_tok1.json", help="Path to tokenizer json")
    parser.add_argument("--tok_name", type=str, default="tok1", help="Tokenizer name for logging (tok1=BPE, tok2=warm)")
    parser.add_argument("--warm_embeddings", type=str, default=None, help="Path to warm embeddings .pt file")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    
    args = parser.parse_args()
    
    # 1. Setup Config & Device
    config = get_config(args.model)
    device = config['device']
    # 2. Load Tokenizer (detect type from file content)
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found at {args.tokenizer}.")
        return
    
    with open(args.tokenizer, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)
    
    if tok_data.get('type') == 'word_level_pos':
        tokenizer_type = "warm"
        from scripts.build_warm_tokenizer import WarmTokenizer
        tokenizer = WarmTokenizer(args.tokenizer)
        config['vocab_size'] = len(tokenizer)
        print(f"Loaded warm tokenizer with vocab size: {config['vocab_size']}")
    else:
        tokenizer_type = "bpe"
        tokenizer = Tokenizer.from_file(args.tokenizer)
        config['vocab_size'] = tokenizer.get_vocab_size()
        print(f"Loaded BPE tokenizer with vocab size: {config['vocab_size']}")

    # 3. Setup Logging & Print Config
    print(f"--- Training {args.model} ---")
    print(f"Device: {device}")
    print(f"Config: {config}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    run_name = f"{args.model}_{args.tok_name}_{data_name}"
    log_file_path = os.path.join(args.log_dir, f"{run_name}_log.txt")
    print(f"Run name: {run_name}")
    print(f"Logging to: {log_file_path}")
    
    with open(log_file_path, "w") as f:
        f.write(f"Run: {run_name}\n")
        f.write(f"Training {args.model} on {args.data} with tokenizer {args.tok_name}\n")
        f.write(f"Config: {config}\n")
    
    # 4. Load warm embeddings if provided
    warm_embeddings = None
    if args.warm_embeddings:
        if not os.path.exists(args.warm_embeddings):
            print(f"Error: Warm embeddings not found at {args.warm_embeddings}")
            return
        warm_embeddings = torch.load(args.warm_embeddings, weights_only=True)
        print(f"Loaded warm embeddings: {warm_embeddings.shape}")
    
    # 4. Import Model
    try:
        module = __import__(f"models.{args.model}", fromlist=[args.model.capitalize()])
        ModelClass = getattr(module, args.model.capitalize())
    except ImportError:
        print(f"Error: Could not import models/{args.model}.py")
        return
    except AttributeError:
        print(f"Error: Could not find class {args.model.capitalize()} in models/{args.model}.py")
        return

    # 5. Initialize Model (with optional warm embeddings)
    model = ModelClass(config, warm_embeddings=warm_embeddings)
    model.to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 6. Data Loader
    dataset = TextDataset(args.data, tokenizer, config['block_size'], tokenizer_type=tokenizer_type)
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0) # Set num_workers > 0 on server if needed
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # 7. Training Loop
    best_val_loss = float('inf')
    iter_num = 0
    start_time = time.time()
    
    # Infinite loop wrapper for DataLoader to match max_iters logic
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(config['max_iters']), desc="Training")
    
    for iter_num in pbar:
        # Get Batch
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            
        X, Y = X.to(device), Y.to(device)
        
        # Forward
        logits, loss = model(X, Y)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if iter_num % config['eval_interval'] == 0:
            val_loss = estimate_loss(model, val_loader, device)
            log_msg = f"Step {iter_num}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}"
            tqdm.write(log_msg)
            
            # Write to log file
            with open(log_file_path, "a") as f:
                f.write(log_msg + "\n")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.out_dir, f"{run_name}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                tqdm.write(f"--> Saved best model to {ckpt_path}")
        
        # Update progress bar description
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    print("Training Complete.")

if __name__ == "__main__":
    main()