#!/usr/bin/env python3
"""
Export a Keras/TensorFlow model to TFLite:
- FP32
- Dynamic range quantization
- Full integer quantization (requires representative_dataset)
"""
import argparse
import tensorflow as tf
import os
import numpy as np

def representative_dataset_gen():
    # TODO: replace with real data loader (use a small subset)
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

def convert(model_path, out_dir):
    model = tf.keras.models.load_model(model_path)
    os.makedirs(out_dir, exist_ok=True)

    # FP32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = os.path.join(out_dir, "model_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)

    # Dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()
    dyn_path = os.path.join(out_dir, "model_dynamic.tflite")
    with open(dyn_path, "wb") as f:
        f.write(tflite_dynamic)

    # Full integer quantization (weights + activations)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8
        tflite_int8 = converter.convert()
        int8_path = os.path.join(out_dir, "model_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
    except Exception as e:
        print("Full integer quantization failed:", e)
        int8_path = None

    print("Saved:", fp32_path, dyn_path, int8_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Keras model (.h5) or saved_model dir")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    convert(args.input, args.output_dir)
