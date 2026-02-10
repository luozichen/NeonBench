import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_config(model_name):
    config = {
        'vocab_size': 1024,
        'd_model': 256,
        'n_layers': 4,
        'n_head': 4,
        'd_ff': 512,
        'block_size': 256
    }
    if model_name == "neon004":
        config['d_ff_wide'] = 1024
    if model_name == "neon006":
        config['d_latent'] = 128
    if model_name == "neon011":
        config.update({'d_model': 384, 'n_layers': 8, 'n_head': 6, 'd_ff': 768})
    elif model_name == "neon012":
        config.update({'d_model': 512, 'n_layers': 6, 'n_head': 8, 'd_ff': 1024})
    elif model_name == "neon013":
        config.update({'d_model': 320, 'n_layers': 8, 'n_head': 8, 'd_ff': 640})
    elif model_name == "neon014":
        config.update({'d_model': 384, 'n_layers': 6, 'n_head': 6, 'd_ff': 1536})
    return config

def main():
    models_to_count = [f"neon{i:03d}" for i in range(1, 15)]

    print(f"{'Model':<10} | {'Parameters':>12}")
    print("-" * 25)

    for model_name in models_to_count:
        try:
            config = get_config(model_name)
            module = __import__(f"models.{model_name}", fromlist=[model_name.capitalize()])
            ModelClass = getattr(module, model_name.capitalize())
            model = ModelClass(config)
            params = count_parameters(model)
            print(f"{model_name:<10} | {params:>12,}")
        except Exception as e:
            print(f"{model_name:<10} | Error: {e}")

if __name__ == "__main__":
    main()