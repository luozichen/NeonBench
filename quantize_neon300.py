"""Quantize Neon300 checkpoint to int8 (weight-only, per-tensor symmetric).
Reduces ~168MB float32 checkpoint to ~42MB int8 + scales.

Usage:
    python quantize_neon300.py
    python quantize_neon300.py --input checkpoints/neon300/neon300_final.pth --output checkpoints/neon300/neon300_final_int8.pth
"""
import argparse
import torch
import os


def quantize_tensor_int8(tensor: torch.Tensor):
    """Symmetric per-tensor int8 quantization. Returns (int8_data, scale)."""
    if tensor.numel() == 0:
        return tensor.to(torch.int8), torch.tensor(1.0)
    amax = tensor.abs().max().float()
    scale = amax / 127.0
    if scale == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), torch.tensor(1.0)
    quantized = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale


def dequantize_tensor_int8(quantized: torch.Tensor, scale: torch.Tensor):
    """Reconstruct float32 tensor from int8 + scale."""
    return quantized.float() * scale


def quantize_checkpoint(input_path: str, output_path: str):
    print(f"Loading checkpoint: {input_path}")
    state_dict = torch.load(input_path, map_location='cpu', weights_only=True)

    input_size = os.path.getsize(input_path)
    print(f"Original size: {input_size / 1e6:.1f} MB")
    print(f"Keys: {len(state_dict)}")

    quantized_state = {}
    stats = {'quantized': 0, 'kept_fp': 0}

    for key, tensor in state_dict.items():
        # Only quantize 2D+ weight tensors (linear layers)
        # Keep 1D params (norms, biases) and small tensors in original precision
        if tensor.ndim >= 2 and tensor.numel() > 256:
            q_data, scale = quantize_tensor_int8(tensor)
            quantized_state[key + '.__qdata__'] = q_data
            quantized_state[key + '.__qscale__'] = scale
            stats['quantized'] += 1

            # Verify reconstruction error
            recon = dequantize_tensor_int8(q_data, scale)
            err = (tensor.float() - recon).abs().mean().item()
        else:
            quantized_state[key] = tensor
            stats['kept_fp'] += 1

    # Save with metadata
    save_payload = {
        'quantized_state': quantized_state,
        'quant_method': 'symmetric_int8_per_tensor',
        'original_keys': list(state_dict.keys()),
    }

    torch.save(save_payload, output_path)
    output_size = os.path.getsize(output_path)

    print(f"\n--- Quantization Complete ---")
    print(f"Quantized layers: {stats['quantized']}")
    print(f"Kept in float:    {stats['kept_fp']}")
    print(f"Output size:      {output_size / 1e6:.1f} MB")
    print(f"Compression:      {input_size / output_size:.1f}x ({100 * (1 - output_size / input_size):.1f}% smaller)")
    print(f"Saved to: {output_path}")


def load_quantized_checkpoint(path: str):
    """Load a quantized checkpoint back into a float32 state_dict."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    quantized_state = payload['quantized_state']
    original_keys = payload['original_keys']

    state_dict = {}
    for key in original_keys:
        qdata_key = key + '.__qdata__'
        qscale_key = key + '.__qscale__'
        if qdata_key in quantized_state:
            state_dict[key] = dequantize_tensor_int8(
                quantized_state[qdata_key],
                quantized_state[qscale_key]
            )
        else:
            state_dict[key] = quantized_state[key]

    return state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Neon300 checkpoint to int8")
    parser.add_argument("--input", type=str, default="checkpoints/neon300/neon300_final.pth")
    parser.add_argument("--output", type=str, default="checkpoints/neon300/neon300_final_int8.pth")
    args = parser.parse_args()

    quantize_checkpoint(args.input, args.output)
