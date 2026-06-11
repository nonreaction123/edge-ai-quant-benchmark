# Edge AI Quantization Benchmark (P3)

This project implements a runnable P3 portfolio project: **quantization + deployment-style benchmarking for edge AI**.

The direction chosen here is a **lightweight, reproducible NumPy baseline** instead of a TensorFlow-heavy pipeline. This avoids the common `pip install tensorflow` issue on Windows/RPi/Jetson and still produces the required artifacts:

- FP32 model
- Dynamic float16 model
- INT8 quantized model
- Latency / FPS / memory / model-size logging
- CSV result table
- Latency plot
- Demo GIF

The same structure can later be extended to TensorFlow Lite or TensorRT when the target hardware is ready.

## Why this direction?

For a portfolio project, the first priority is to have an end-to-end pipeline that runs reliably and produces measurable results. After that, the inference backend can be swapped from NumPy to TFLite or TensorRT.

## Project structure

```text
configs/
  edge_experiment.yaml
src/
  edge_quant/
    core.py
  visualization/
    plot_benchmarks.py
    make_demo_gif.py
scripts/
  run_end_to_end.py
reports/
  benchmark_results.csv
  figures/latency_plot.png
assets/
  demo.gif
outputs/
  models/
```

## Run

```bash
python -m venv .venv
.\.venv\Scripts\activate    # Windows
# source .venv/bin/activate    # Linux/macOS

pip install -r requirements.txt
python scripts/run_end_to_end.py --config configs/edge_experiment.yaml
```

Or:

```bash
make install
make run
```

## Results

The pipeline writes:

```text
reports/benchmark_results.csv
reports/figures/latency_plot.png
assets/demo.gif
```

CSV columns:

```text
model_name, quant_type, file_path, size_mb, latency_p50_ms,
latency_p95_ms, throughput_fps, ram_usage_mb, accuracy, device, date, notes
```

## Hardware deployment notes

- Raspberry Pi: this NumPy baseline runs directly; for production inference, replace the backend with `tflite-runtime`.
- Jetson: use this repository as the benchmark harness; replace `infer()` with TensorRT or `jetson-inference`.
- Always write device name and benchmark date into `reports/benchmark_results.csv`.
