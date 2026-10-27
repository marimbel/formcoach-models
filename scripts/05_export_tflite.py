#!/usr/bin/env python3
"""Export Model A to TensorFlow Lite for on-device inference and check it matches Keras.

    python scripts/05_export_tflite.py --run runs/finetune --data data/processed [--fp16]

Writes RUN/form_classifier.tflite and RUN/labels.json. Copy both into
formcoach-app/mobile/assets/models/ and assets/config/ respectively.
Input: float32 [1,224,224,3] RGB in 0..255.  Output: float32 [1,N] softmax over all classes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.labels import CONFIG_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, help="crops.csv folder for an equivalence check")
    ap.add_argument("--fp16", action="store_true", help="float16 weights (~half size, tiny accuracy change)")
    args = ap.parse_args()

    import keras
    import tensorflow as tf
    model = keras.models.load_model(args.run / "model.keras")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.fp16:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    tfl = conv.convert()
    out = args.run / "form_classifier.tflite"
    out.write_bytes(tfl)
    shutil.copy(CONFIG_PATH, args.run / "labels.json")
    print(f"Wrote {out} ({len(tfl) / 1e6:.1f} MB)")

    interp = tf.lite.Interpreter(model_content=tfl)
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    print("input", i_d["shape"], i_d["dtype"], "output", o_d["shape"], o_d["dtype"])
    if args.data:
        from formcoach.data_tf import load_images
        crops = pd.read_csv(args.data / "crops.csv").head(16)
        x = load_images([args.data / p for p in crops.image_path]).numpy()
    else:
        x = np.random.default_rng(0).uniform(0, 255, (4, 224, 224, 3)).astype(np.float32)
    ref = model.predict(x, verbose=0)
    got = []
    for img in x:
        interp.set_tensor(i_d["index"], img[None].astype(np.float32))
        interp.invoke()
        got.append(interp.get_tensor(o_d["index"])[0])
    got = np.array(got)
    agree = float((ref.argmax(1) == got.argmax(1)).mean())
    print(f"max |keras - tflite| = {np.abs(ref - got).max():.4g}; argmax agreement = {agree:.0%}")


if __name__ == "__main__":
    main()
