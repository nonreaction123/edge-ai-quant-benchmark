#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

def plot(csv_path, out_png):
    df = pd.read_csv(csv_path)
    # Example: compare latency_p50_ms by quant_type for each model_name
    pivot = df.pivot_table(index="model_name", columns="quant_type", values="latency_p50_ms")
    ax = pivot.plot(kind="bar", figsize=(10,5))
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Latency comparison by quantization type")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png)
    print("Saved plot:", out_png)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: plot_benchmarks.py <csv_path> <out_png>")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_png = sys.argv[2]
    plot(csv_path, out_png)
