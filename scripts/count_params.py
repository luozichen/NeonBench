import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
from train import get_config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    models_to_count = [f"neon{i:03d}" for i in range(1, 66)]

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