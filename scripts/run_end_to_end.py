"""Run the full P3 Edge-AI quantization benchmark pipeline.

Usage:
    python scripts/run_end_to_end.py --config configs/edge_experiment.yaml

This produces:
    - outputs/models/*_fp32.npz
    - outputs/models/*_dynamic_float16.npz
    - outputs/models/*_int8.npz
    - reports/benchmark_results.csv
    - reports/figures/latency_plot.png
    - assets/demo.gif
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import yaml

# Allow running from project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from edge_quant.core import (  # noqa: E402
    benchmark_model,
    export_models,
    make_synthetic_dataset,
    train_linear_classifier,
    write_benchmark_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/edge_experiment.yaml")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    bench_cfg = cfg["benchmark"]
    out_cfg = cfg["outputs"]

    print("[1/5] Generating synthetic edge dataset...")
    dataset = make_synthetic_dataset(
        seed=seed,
        n_train=int(data_cfg["n_train"]),
        n_test=int(data_cfg["n_test"]),
        n_features=int(data_cfg["n_features"]),
        n_classes=int(data_cfg["n_classes"]),
    )

    print("[2/5] Training tiny model...")
    model = train_linear_classifier(
        dataset=dataset,
        n_classes=int(data_cfg["n_classes"]),
        epochs=int(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        seed=seed,
    )
    print(f"    Test accuracy before quantization: {model.accuracy:.4f}")

    print("[3/5] Exporting FP32, float16/dynamic, and int8 variants...")
    model_dir = PROJECT_ROOT / out_cfg["model_dir"]
    model_paths = export_models(model, model_dir, model_cfg["name"])

    print("[4/5] Benchmarking model variants...")
    device = platform.machine() or platform.platform()
    results = []
    for path in model_paths:
        result = benchmark_model(
            model_path=path,
            x_sample=dataset.x_test,
            y_sample=dataset.y_test,
            iterations=int(bench_cfg["iterations"]),
            warmup_runs=int(bench_cfg["warmup_runs"]),
            device=device,
        )
        results.append(result)
        print(
            f"    {result['quant_type']:<16} "
            f"size={result['size_mb']:.4f} MB, "
            f"p50={result['latency_p50_ms']:.5f} ms, "
            f"fps={result['throughput_fps']:.1f}, "
            f"acc={result['accuracy']:.4f}"
        )

    csv_path = PROJECT_ROOT / out_cfg["report_csv"]
    write_benchmark_csv(results, csv_path)
    print(f"    Wrote {csv_path}")

    print("[5/5] Creating plot and GIF demo...")
    subprocess.check_call([
        sys.executable,
        str(PROJECT_ROOT / "src/visualization/plot_benchmarks.py"),
        "--csv",
        str(csv_path),
        "--output",
        str(PROJECT_ROOT / out_cfg["figure_path"]),
    ])
    subprocess.check_call([
        sys.executable,
        str(PROJECT_ROOT / "src/visualization/make_demo_gif.py"),
        "--output",
        str(PROJECT_ROOT / out_cfg["demo_gif"]),
    ])

    print("Done. Key artifacts:")
    print(f"  - {csv_path}")
    print(f"  - {PROJECT_ROOT / out_cfg['figure_path']}")
    print(f"  - {PROJECT_ROOT / out_cfg['demo_gif']}")


if __name__ == "__main__":
    main()
