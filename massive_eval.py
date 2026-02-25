import os
import torch
from tokenizers import Tokenizer
import sys
sys.path.append('.')

from train_parity import TurboSampler, get_config

def massive_eval(model_name, data_path, tok_path, eval_cycles=500):
    config = get_config(model_name)
    device = config['device']
    tokenizer = Tokenizer.from_file(tok_path)
    config['vocab_size'] = tokenizer.get_vocab_size()

    # Import
    module = __import__(f"models.{model_name}", fromlist=[model_name.capitalize()])
    ModelClass = getattr(module, model_name.capitalize())
    model = ModelClass(config).to(device)
    
    # Load
    ckpt_path = f"checkpoints/{model_name}_parity_final.pth"
    if not os.path.exists(ckpt_path):
        print(f"Skipping {model_name} - No checkpoint found.")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    sampler = TurboSampler(data_path, config['block_size'], config['batch_size'], device)
    
    losses = []
    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for i in range(eval_cycles):
            for strm in [False, True]:
                vx, vy = sampler.get_batch('val')
                _, loss = model(vx, vy, is_odd_stream=strm)
                losses.append(loss.item())
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{eval_cycles}] Running Avg: {sum(losses)/len(losses):.5f}")
                
    final_loss = sum(losses) / len(losses)
    print(f"==> {model_name} FINAL MASSIVE EVAL LOSS: {final_loss:.5f} (over {len(losses)} batches)")
    return final_loss

if __name__ == "__main__":
    results = {}
    data = "data/tinyshakespeare/tinyshakespeare.bin"
    tok = "tokenizers/tinyshakespeare"
    
    for i in range(243, 253): # Phase 8 & 9
        results[f"neon{i}"] = massive_eval(f"neon{i}", data, tok, eval_cycles=1000)
        
    print("\n--- FINAL RANKING (1000 BATCHES) ---")
    for k, v in results.items():
        print(f"{k}: {v:.5f}")
