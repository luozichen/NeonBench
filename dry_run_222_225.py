import subprocess
import sys

models = ["neon222", "neon223", "neon224", "neon225"]
data_path = "data/wiki103_tok5.npy" # Assuming this path from context

for m in models:
    print(f"\n--- Testing {m} ---")
    # Run for 5 iterations just to check for crashes
    cmd = [
        "python3", "train.py",
        "--model", m,
        "--data", data_path,
        "--max_iters", "5",
        "--batch_size", "8",
        "--eval_interval", "10"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"SUCCESS: {m} started and finished 5 steps.")
        else:
            print(f"FAILURE: {m} failed with return code {result.returncode}")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {m} took too long (likely hung).")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR running {m}: {e}")
        sys.exit(1)

print("\nALL MODELS VERIFIED FUNCTIONAL")
