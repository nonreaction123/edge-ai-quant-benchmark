"""Plot benchmark CSV into a latency/throughput figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to reports/benchmark_results.csv")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    labels = df["quant_type"].tolist()
    x = range(len(labels))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar([i - 0.18 for i in x], df["latency_p50_ms"], width=0.36, label="Latency p50 (ms)")
    ax1.set_ylabel("Latency p50 (ms)")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=15)

    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], df["throughput_fps"], width=0.36, label="Throughput (FPS)", alpha=0.6)
    ax2.set_ylabel("Throughput (FPS)")

    ax1.set_title("Edge AI quantization benchmark")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
