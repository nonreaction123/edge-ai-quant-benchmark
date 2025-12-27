# Edge AI Quantization Benchmark (P3)

Mục tiêu
- Thực hiện pipeline benchmark cho quantization của mô hình Edge AI (TensorFlow / TFLite).
- Export 3 phiên bản: FP32, Dynamic range quantization, Full integer quantization.
- Đo và so sánh: latency (ms), throughput (FPS), model size (MB), RAM usage trên thiết bị Edge (Raspberry Pi, Jetson).

Requirement highlights & tools khuyến nghị
- TensorFlow Lite post-training quantization  
  https://www.tensorflow.org/model_optimization/guide/quantization/post_training
- benchmark_model / LiteRT performance measurement  
  https://ai.google.dev/edge/litert/models/measurement
- Nếu có Jetson: dùng jetson-inference (TensorRT) để demo realtime  
  https://github.com/dusty-nv/jetson-inference

Kịch bản “vừa sức nhưng rất ăn điểm”
1. Train model nhỏ (MobileNetV2/Small CNN) cho task từ P2 (NEU hoặc 1 class trong MVTec)
2. Export 3 bản:
   - FP32
   - Dynamic range quantization
   - Full integer quantization (nếu kịp)
3. Benchmark trên thiết bị:
   - latency (ms), throughput (FPS), model size (MB), RAM usage
4. Ghi kết quả vào `reports/benchmark_results.csv` + plot so sánh

Deliverables bắt buộc
- `reports/benchmark_results.csv`
- `reports/figures/latency_plot.png`
- `assets/demo.gif` (inference realtime webcam/ảnh)
- README có bảng so sánh 3 phiên bản model

Cấu trúc đề xuất (mở rộng)
- configs/
  - quant_experiment.yaml
- src/
  - models/
    - train.py (placeholder)
    - export_tflite.py
  - evaluation/
    - benchmark_tflite.py
  - visualization/
    - plot_benchmarks.py
- reports/
  - benchmark_results.csv
  - figures/latency_plot.png
- assets/
  - demo.gif

Quick start (local dev)
1. Tạo môi trường
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .\\.venv\\Scripts\\activate    # Windows
   pip install -r requirements.txt
   ```
   Ghi chú: trên RPi/Jetson cài `tflite-runtime` hoặc TensorRT/torch tùy device.

2. Huấn luyện model (ví dụ)
   ```bash
   python src/models/train.py --config configs/quant_experiment.yaml
   ```

3. Export sang TFLite (FP32, dynamic, full-int)
   ```bash
   python src/models/export_tflite.py --input outputs/<model>/saved_model --output-dir outputs/tflite/
   ```

4. Đo benchmark (local hoặc remote device)
   - Dùng `benchmark_model` (Google) hoặc `src/evaluation/benchmark_tflite.py`.
   - Kết quả được append vào `reports/benchmark_results.csv`.

5. Vẽ plot
   ```bash
   python src/visualization/plot_benchmarks.py reports/benchmark_results.csv reports/figures/latency_plot.png
   ```

RPi / Jetson notes
- Raspberry Pi: cài `tflite-runtime` + opencv, dùng `benchmark_model` binary (prebuilt).
- Jetson: có thể convert model sang TensorRT và demo realtime bằng `jetson-inference`. Ghi rõ device info vào CSV khi benchmark.

CSV schema (reports/benchmark_results.csv)
- model_name, quant_type, file_path, size_mb, latency_p50_ms, latency_p95_ms, throughput_fps, ram_usage_mb, device, date, notes

Gợi ý CI / reproducibility
- Thêm script `scripts/run_small_benchmark.sh` để chạy full pipeline trên sample data.
- Thêm GitHub Action để chạy smoke test (train tiny model + export FP32 + dynamic quant + run small benchmark) nếu muốn.
