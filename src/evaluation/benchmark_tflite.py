#!/usr/bin/env python3
"""
Simple TFLite benchmarking script (pure Python).
Measures latency distribution and throughput, logs to reports/benchmark_results.csv.
Works with TF or tflite-runtime interpreters.
"""
import argparse
import time
import os
import numpy as np
import pandas as pd
import psutil
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

def load_interpreter(tflite_path, num_threads=1):
    try:
        return tflite.Interpreter(model_path=tflite_path, num_threads=num_threads)
    except Exception:
        return tflite.Interpreter(model_path=tflite_path)

def random_input(shape, dtype=np.float32):
    if dtype == np.uint8:
        return (np.random.rand(*shape) * 255).astype(np.uint8)
    return np.random.rand(*shape).astype(dtype)

def measure(tflite_path, runs=200, warmup=20, num_threads=1):
    interp = load_interpreter(tflite_path, num_threads=num_threads)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    input_shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']

    # warmup
    for _ in range(warmup):
        inp = random_input(input_shape, dtype)
        interp.set_tensor(input_details[0]['index'], inp)
        interp.invoke()

    latencies = []
    proc = psutil.Process()
    mem_samples = []
    start_all = time.time()
    for _ in range(runs):
        t0 = time.time()
        inp = random_input(input_shape, dtype)
        interp.set_tensor(input_details[0]['index'], inp)
        interp.invoke()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000.0)
        mem_samples.append(proc.memory_info().rss / 1024.0 / 1024.0)
    duration = time.time() - start_all
    fps = runs / duration
    return {
        "p50": float(np.percentile(latencies, 50)),
        "p95": float(np.percentile(latencies, 95)),
        "fps": float(fps),
        "mem_mb": float(np.max(mem_samples))
    }

def append_csv(row, csv_path="reports/benchmark_results.csv"):
    df = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--quant-type", required=True)
    p.add_argument("--device", default="local")
    p.add_argument("--runs", type=int, default=200)
    args = p.parse_args()

    stats = measure(args.model, runs=args.runs)
    size_mb = os.path.getsize(args.model) / (1024*1024)
    row = {
        "model_name": os.path.basename(args.model),
        "quant_type": args.quant_type,
        "file_path": args.model,
        "size_mb": round(size_mb, 3),
        "latency_p50_ms": stats["p50"],
        "latency_p95_ms": stats["p95"],
        "throughput_fps": round(stats["fps"], 2),
        "ram_usage_mb": round(stats["mem_mb"], 2),
        "device": args.device,
        "date": pd.Timestamp.now().isoformat(),
        "notes": ""
    }
    append_csv(row)
    print("Appended:", row)
