"""
Neon Benchmarking Script: FP32 vs FP16 vs CoreH
===============================================
This script measures the performance of neon213 under different precision and correction modes.
Methodology:
1. Load a sequence from the validation set.
2. Split into Prompt (first 128) and Target (last 128).
3. Compute CoreH steering vector on the Prompt.
4. Measure Loss/Accuracy on the Target predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import time
from tokenizers import Tokenizer
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())
from models.neon213 import Neon213

# ================= CONFIGURATION =================
# EDIT THESE PATHS BEFORE RUNNING ON SERVER
CKPT_FP32 = "checkpoints/neon213_growth/neon213_final.pth"
CKPT_FP16 = "checkpoints/neon213_growth/neon213_final_fp16.pth"
DATA_PATH = "data/fineweb/fineweb_tok6.bin"
TOK_PATH = "tokenizers/fineweb_tok6.json"

MODEL_CONFIG = {
    'd_model': 384,
    'n_head': 6,
    'n_layers': 8,
    'd_ff': 1536,
    'block_size': 256,
    'vocab_size': 16384, # Updated on load
    'conv_k': 9,
    'mlp_k': 9,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
# =================================================

def calculate_steering_vector(model, ids, device):
    """
    Calibrate CoreH on the prompt tokens.
    Uses the delta between prediction and ground-truth embeddings.
    """
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    
    hidden_registry = []
    def calibration_hook(module, inp, out):
        hidden_registry.append(out.detach())
    
    h = model.ln_f.register_forward_hook(calibration_hook)
    try:
        with torch.no_grad():
            model(idx)
    finally:
        h.remove()
    
    if not hidden_registry: return None
    
    # [T, D]
    h_prompt = hidden_registry[0][0]
    next_token_ids = torch.tensor(ids[1:], device=device)
    target_embs = model.token_emb(next_token_ids)
    
    # Normalize targets to same space as h_prompt (Pre-Head)
    target_embs_norm = model.ln_f(target_embs)
    
    # Errors for tokens [0:T-1] predicting [1:T]
    preds = h_prompt[:-1]
    errors = target_embs_norm - preds
    
    # Quadratic weighting: end of prompt counts more (i^2)
    weights = torch.arange(1, len(errors) + 1, device=device).float().pow(2)
    weights = weights / weights.sum()
    
    # Weighted average [1, D]
    steering = (errors * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
    return steering

@torch.no_grad()
def evaluate_mode(model, sample_tensor, mode="base", steering_vector=None):
    """
    Evaluates loss and accuracy on the latter half of the sample.
    sample_tensor: [1, 256]
    """
    model.eval()
    B, T = sample_tensor.shape
    prompt_len = T // 2 # 128
    
    # Setup steering hook if in coreh mode
    h_steer = None
    if mode == "coreh" and steering_vector is not None:
        def steering_hook(module, inp, out):
            # Apply nudge to only the 'generated' (eval) tokens
            # out is [1, T, D]. We nudge indices prompt_len and onwards.
            out[:, prompt_len:, :] += steering_vector
            return out
        h_steer = model.ln_f.register_forward_hook(steering_hook)
    
    try:
        X = sample_tensor[:, :-1].to(MODEL_CONFIG['device']) # [1, 255]
        Y = sample_tensor[:, 1:].to(MODEL_CONFIG['device'])  # [1, 255]
        
        logits, _ = model(X)
        
        # Cross-entropy objective for tokens [prompt_len : 255]
        # These correspond to predicting targets Y[prompt_len-1 : 254]
        # (Y[prompt_len-1] is the target for X[prompt_len-1])
        eval_logits = logits[0, prompt_len-1:, :] # [eval_tokens, V]
        eval_targets = Y[0, prompt_len-1:]         # [eval_tokens]
        
        loss = F.cross_entropy(eval_logits, eval_targets)
        
        preds = torch.argmax(eval_logits, dim=-1)
        acc = (preds == eval_targets).float().mean()
        
        return loss.item(), acc.item()
    finally:
        if h_steer: h_steer.remove()

def run_bench():
    # Path Checks
    for p in [CKPT_FP32, CKPT_FP16, DATA_PATH, TOK_PATH]:
        if not os.path.exists(p):
            print(f"CRITICAL: Path not found -> {p}")
            print("Please edit the paths in bench_coreh.py before running.")
            return

    tokenizer = Tokenizer.from_file(TOK_PATH)
    MODEL_CONFIG['vocab_size'] = tokenizer.get_vocab_size()
    device = MODEL_CONFIG['device']
    
    print(f"--- NeonBench: Precision & Steering Test ---")
    print(f"Block Size: {MODEL_CONFIG['block_size']} (Prompt: 128, Eval: 128)")
    print(f"Device: {device}")

    # Load Data (using memory map)
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    val_start = int(0.95 * len(data))
    val_data = data[val_start:]
    
    num_samples = 200 # Statistically significant sample size
    
    results = {
        "FP32 (Base)": {"loss": [], "acc": []},
        "FP16 (Half)": {"loss": [], "acc": []},
        "FP16+CoreH": {"loss": [], "acc": []}
    }
    
    def evaluate_checkpoint(ckpt_path, label, is_fp16=False):
        print(f"\nEvaluating {label}...")
        model = Neon213(MODEL_CONFIG).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        
        for i in tqdm(range(num_samples)):
            offset = i * MODEL_CONFIG['block_size']
            if offset + MODEL_CONFIG['block_size'] > len(val_data): break
            
            sample = val_data[offset : offset + MODEL_CONFIG['block_size']]
            sample_tensor = torch.from_numpy(sample.astype(np.int64)).unsqueeze(0)
            
            # 1. Base Evaluation
            loss, acc = evaluate_mode(model, sample_tensor, mode="base")
            results[label]["loss"].append(loss)
            results[label]["acc"].append(acc)
            
            # 2. CoreH Evaluation (only if label is FP16+CoreH placeholder)
            if label == "FP16 (Half)":
                prompt_ids = sample[:MODEL_CONFIG['block_size']//2].tolist()
                s = calculate_steering_vector(model, prompt_ids, device)
                loss_c, acc_c = evaluate_mode(model, sample_tensor, mode="coreh", steering_vector=s)
                results["FP16+CoreH"]["loss"].append(loss_c)
                results["FP16+CoreH"]["acc"].append(acc_c)
        
        del model
        torch.cuda.empty_cache()

    # Execution
    evaluate_checkpoint(CKPT_FP32, "FP32 (Base)")
    evaluate_checkpoint(CKPT_FP16, "FP16 (Half)")

    # Final Summary
    print("\n" + "═"*60)
    print(f"{'Configuration':<16} │ {'Loss':<10} │ {'PPL':<10} │ {'Top-1 Acc':<10}")
    print("─"*60)
    for mode in results.keys():
        if not results[mode]["loss"]: continue
        avg_loss = np.mean(results[mode]["loss"])
        avg_acc = np.mean(results[mode]["acc"])
        ppl = np.exp(avg_loss)
        print(f"{mode:<16} │ {avg_loss:.4f}     │ {ppl:.4f}     │ {avg_acc:.2%}")
    print("═"*60)
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_bench()
