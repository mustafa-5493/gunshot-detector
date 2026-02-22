"""
Inference latency benchmark — CUDA (GTX 1050) vs Apple MPS (M4)
Run this script separately on each machine after training is complete.

On Windows/CUDA machine:
    python benchmark/latency_test.py --device cuda

On Mac M4:
    python benchmark/latency_test.py --device mps
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHECKPOINT_DIR
from src.model import GunshotCNN

NUM_WARMUP_RUNS = 50
NUM_BENCH_RUNS  = 500
BATCH_SIZES     = [1, 8, 32, 64]


def sync_device(device_str):
    """Synchronize device to get accurate wall-clock timing."""
    if device_str == "cuda":
        torch.cuda.synchronize()
    elif device_str == "mps":
        torch.mps.synchronize()
    # cpu needs no sync


def benchmark_device(device_str):
    device = torch.device(device_str)

    checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint), f"Checkpoint not found: {checkpoint}"

    model = GunshotCNN()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"\n{'='*55}")
    print(f" Device : {device_str.upper()}")
    print(f" Model  : GunshotCNN ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"{'='*55}")
    print(f"{'Batch Size':<15} {'Avg Latency (ms)':<20} {'Throughput (samples/s)'}")
    print(f"{'-'*55}")

    results = {}

    with torch.no_grad():
        for batch_size in BATCH_SIZES:
            dummy = torch.randn(batch_size, 1, 64, 173).to(device)

            # Warmup — let the device JIT compile and cache
            for _ in range(NUM_WARMUP_RUNS):
                _ = model(dummy)
            sync_device(device_str)

            # Benchmark
            latencies = []
            for _ in range(NUM_BENCH_RUNS):
                sync_device(device_str)
                start = time.perf_counter()
                _ = model(dummy)
                sync_device(device_str)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

            avg_ms     = np.mean(latencies)
            std_ms     = np.std(latencies)
            throughput = (batch_size / avg_ms) * 1000  # samples/sec

            results[batch_size] = {
                "avg_ms"    : avg_ms,
                "std_ms"    : std_ms,
                "throughput": throughput,
            }

            print(f"{batch_size:<15} {avg_ms:.3f} ± {std_ms:.3f} ms"
                  f"      {throughput:.0f} samples/s")

    print(f"\nSingle-sample inference (batch=1): "
          f"{results[1]['avg_ms']:.3f} ms avg on {device_str.upper()}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/benchmark_{device_str}.txt"
    with open(out_path, "w") as f:
        f.write(f"Device: {device_str}\n")
        for bs, r in results.items():
            f.write(f"batch={bs} | avg={r['avg_ms']:.3f}ms | "
                    f"std={r['std_ms']:.3f}ms | "
                    f"throughput={r['throughput']:.0f} samples/s\n")
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to benchmark"
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available on this machine.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available on this machine.")

    benchmark_device(args.device)