import torch
import torch.nn as nn
import argparse
import os

def quantize_to_fp16(input_path, output_path):
    print(f"Loading weights from {input_path}...")
    state = torch.load(input_path, map_location='cpu')
    
    # Handle both full checkpoints and naked state_dicts
    if 'model' in state:
        weights = state['model']
        is_full_ckpt = True
    else:
        weights = state
        is_full_ckpt = False
        
    print(f"Converting weights to FP16...")
    for k, v in weights.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            weights[k] = v.half()
            
    if is_full_ckpt:
        state['model'] = weights
    else:
        state = weights
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(state, output_path)
    
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Success!")
    print(f"  Original Size: {original_size:.2f} MB")
    print(f"  Quantized Size: {new_size:.2f} MB")
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP16 Quantization Utility")
    parser.add_argument("input", help="Path to the input .pth file")
    parser.add_argument("output", help="Path to save the FP16 .pth file")
    
    args = parser.parse_args()
    quantize_to_fp16(args.input, args.output)
