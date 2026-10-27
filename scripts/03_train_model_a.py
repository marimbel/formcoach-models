#!/usr/bin/env python3
"""Train Model A: MobileNetV3-Large form-error classifier on bottom-of-rep body crops.

Two runs are needed for the report (the course requires comparing against the pretrained model):

  # 1) Pretrained baseline: frozen ImageNet backbone + new linear head ("linear probe")
  python scripts/03_train_model_a.py --data data/processed --mode probe    --out runs/probe
  # 2) Fine-tuned: probe warm-up, then unfreeze the top of the backbone at a low learning rate
  python scripts/03_train_model_a.py --data data/processed --mode finetune --out runs/finetune

Run on a Colab GPU (see notebooks/colab_model_a.ipynb). Output: model.keras, history.json, run_config.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.labels import class_names  # noqa: E402


def build_model(n_classes: int, weights: str | None, dropout: float):
    import keras
    base = keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights=weights,
                                               pooling="avg", include_preprocessing=True)
    base.trainable = False
    inp = keras.Input((224, 224, 3), name="image_0_255")
    x = base(inp, training=False)  # keeps BatchNorm in inference mode even after unfreezing
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax", name="form_probs")(x)
    return keras.Model(inp, out, name="formcoach_model_a"), base


def class_weights(train_df, names):
    counts = train_df.class_name.value_counts()
    present = [c for c in names if counts.get(c, 0) > 0]
    total = sum(counts[c] for c in present)
    return {i: (total / (len(present) * counts[c]) if counts.get(c, 0) else 0.0) for i, c in enumerate(names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with crops.csv (after 02_make_splits)")
    ap.add_argument("--mode", choices=["probe", "finetune"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs-head", type=int, default=15)
    ap.add_argument("--epochs-ft", type=int, default=20)
    ap.add_argument("--unfreeze", type=int, default=60, help="number of backbone layers to unfreeze (from the top)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-ft", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (smoke tests only)")
    ap.add_argument("--seed", type=int, default=663)
    args = ap.parse_args()

    import keras
    import tensorflow as tf
    from formcoach.data_tf import make_dataset

    keras.utils.set_random_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    names = class_names()
    c2i = {c: i for i, c in enumerate(names)}
    df = pd.read_csv(args.data / "crops.csv")
    if "split" not in df:
        sys.exit("crops.csv has no split column; run 02_make_splits.py first")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    print(f"train {len(tr)} crops / val {len(va)} crops / {len(names)} classes")
    ds_tr = make_dataset(tr, args.data, c2i, True, args.batch, args.seed)
    ds_va = make_dataset(va, args.data, c2i, False, args.batch)
    cw = class_weights(tr, names)

    model, base = build_model(len(names), None if args.weights == "none" else args.weights, args.dropout)
    ckpt = str(args.out / "model.keras")
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)]
    metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(optimizer=keras.optimizers.Adam(args.lr_head), loss="sparse_categorical_crossentropy", metrics=metrics)
    h1 = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs_head, class_weight=cw, callbacks=cbs, verbose=2)
    history = {k: list(map(float, v)) for k, v in h1.history.items()}
    stage_boundary = len(history["loss"])

    if args.mode == "finetune":
        base.trainable = True
        for layer in base.layers[:-args.unfreeze]:
            layer.trainable = False
        for layer in base.layers:  # BatchNorm stays frozen: small batches would wreck its statistics
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=keras.optimizers.Adam(args.lr_ft), loss="sparse_categorical_crossentropy", metrics=metrics)
        h2 = model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs_ft, class_weight=cw, callbacks=cbs, verbose=2)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(map(float, v))

    model.save(ckpt)
    (args.out / "history.json").write_text(json.dumps({"history": history, "stage_boundary": stage_boundary}, indent=1))
    (args.out / "run_config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
                                                          "class_names": names,
                                                          "n_train": len(tr), "n_val": len(va),
                                                          "trainable_params": int(np.sum([np.prod(w.shape) for w in model.trainable_weights])),
                                                          "tf_version": tf.__version__}, indent=1))
    print(f"Saved {ckpt}")


if __name__ == "__main__":
    main()
