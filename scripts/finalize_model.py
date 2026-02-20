import torch
import os
import argparse

def finalize(checkpoint_path, out_dir, model_name):
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract only state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    os.makedirs(out_dir, exist_ok=True)
    
    # Save FP32 version
    fp32_path = os.path.join(out_dir, f"{model_name}_final.pth")
    print(f"Saving FP32 weights to {fp32_path}...")
    torch.save(state_dict, fp32_path)
    
    # Save FP16 version
    fp16_path = os.path.join(out_dir, f"{model_name}_final_fp16.pth")
    print(f"Generating and saving FP16 weights to {fp16_path}...")
    
    fp16_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            fp16_state_dict[k] = v.half()
        else:
            fp16_state_dict[k] = v
            
    torch.save(fp16_state_dict, fp16_path)
    
    print(f"Successfully finalized {model_name}:")
    print(f"  FP32: {os.path.getsize(fp32_path)/(1024*1024):.2f} MB")
    print(f"  FP16: {os.path.getsize(fp16_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Finalization Utility")
    parser.add_argument("checkpoint", help="Path to input checkpoint")
    parser.add_argument("--dir", default="checkpoints/neon213_turbo", help="Output directory")
    parser.add_argument("--name", default="neon213_k21", help="Base name for output files")
    
    args = parser.parse_args()
    finalize(args.checkpoint, args.dir, args.name)
