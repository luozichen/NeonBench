"""Convergence threshold analysis: steps to reach val loss thresholds."""
import os
import re
import argparse

def parse_log(log_path):
    steps, val_losses = [], []
    if not os.path.exists(log_path):
        return steps, val_losses
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r"Step (\d+):.*Val Loss ([\d.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
    return steps, val_losses

def steps_to_threshold(steps, losses, threshold):
    for s, l in zip(steps, losses):
        if l < threshold:
            return s
    return None

def main():
    parser = argparse.ArgumentParser(description="Convergence threshold analysis")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--data_name", type=str, default="hp0")
    parser.add_argument("--tok", type=str, default="tok1")
    parser.add_argument("--models", type=str, default="neon005,neon016",
                        help="Comma-separated model names")
    parser.add_argument("--thresholds", type=str, default="3.0,2.5,2.0,1.5",
                        help="Comma-separated val loss thresholds")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # Header
    th_headers = [f"VL<{t}" for t in thresholds]
    header = f"{'Model':<12} | " + " | ".join(f"{h:>8}" for h in th_headers) + f" | {'Final':>8}"
    print(header)
    print("-" * len(header))

    for model in models:
        log_path = os.path.join(args.log_dir, f"{model}_{args.tok}_{args.data_name}_log.txt")
        steps, losses = parse_log(log_path)
        if not steps:
            print(f"{model:<12} | " + " | ".join(f"{'—':>8}" for _ in thresholds) + f" | {'—':>8}")
            continue

        row = f"{model:<12} | "
        for th in thresholds:
            s = steps_to_threshold(steps, losses, th)
            row += f"{s if s else '—':>8} | "
        row += f"{losses[-1]:>8.4f}"
        print(row)

if __name__ == "__main__":
    main()
