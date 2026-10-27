#!/usr/bin/env python3
"""Evaluate one or more Model A runs on the held-out TEST participants and build report figures.

    python scripts/04_eval_model_a.py --data data/processed --runs runs/probe runs/finetune \
        --export-preds runs/finetune

The user picks the exercise in the app, so predictions are restricted ("masked") to that exercise's
classes and renormalised -- exactly what the app does. Metrics are reported per rep: the probabilities
of the crops around the rep's bottom frame are averaged, again mirroring the app.

Writes to each run dir: metrics.json, report.txt, confusion.png, curves.png, success.jpg, failure.jpg,
pred_test.csv; and results_model_a.md (comparison table) next to the first run.
--export-preds RUN writes RUN/preds_all.csv (every split) for the Gemini dataset builder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.labels import class_names, exercise_class_indices, exercises, split_class  # noqa: E402


def masked(probs: np.ndarray, exercise_per_row: list[str]) -> np.ndarray:
    out = np.zeros_like(probs)
    for ex in set(exercise_per_row):
        rows = [i for i, e in enumerate(exercise_per_row) if e == ex]
        cols = exercise_class_indices(ex)
        sub = probs[np.ix_(rows, cols)]
        out[np.ix_(rows, cols)] = sub / np.clip(sub.sum(1, keepdims=True), 1e-9, None)
    return out


def predict(model, df, root, batch=64):
    from formcoach.data_tf import make_dataset
    names = class_names()
    ds = make_dataset(df, root, {c: i for i, c in enumerate(names)}, False, batch)
    return model.predict(ds, verbose=0)


def rep_level(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    names = class_names()
    p = pd.DataFrame(probs, columns=names)
    p["rep_id"] = df.rep_id.values
    agg = p.groupby("rep_id")[names].mean()
    meta = df.drop_duplicates("rep_id").set_index("rep_id")[["participant", "exercise", "error", "class_name", "split"]]
    out = meta.join(agg)
    out["pred_class"] = out[names].idxmax(axis=1)
    out["pred_prob"] = out[names].max(axis=1)
    return out.reset_index()


def plot_confusions(rep_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    exs = [e for e in exercises() if (rep_df.exercise == e).any()]
    fig, axes = plt.subplots(1, len(exs), figsize=(5 * len(exs), 4.6), squeeze=False)
    for ax, ex in zip(axes[0], exs):
        sub = rep_df[rep_df.exercise == ex]
        labels = [class_names()[i] for i in exercise_class_indices(ex)]
        cm = confusion_matrix(sub.class_name, sub.pred_class, labels=labels)
        ax.imshow(cm, cmap="Blues")
        short = [split_class(l)[1] for l in labels]
        ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
        ax.set_yticks(range(len(short)), short)
        for i in range(len(short)):
            for j in range(len(short)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(ex)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_curves(hist_path: Path, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not hist_path.exists():
        return
    h = json.loads(hist_path.read_text())
    hist, b = h["history"], h.get("stage_boundary")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in ((a1, "loss"), (a2, "accuracy")):
        ax.plot(hist.get(key, []), label=f"train {key}")
        ax.plot(hist.get("val_" + key, []), label=f"val {key}")
        if b and b < len(hist.get(key, [])):
            ax.axvline(b - 0.5, ls="--", c="gray", lw=0.8)
            ax.text(b, ax.get_ylim()[1], " unfreeze", va="top", fontsize=8, color="gray")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def gallery(rows: pd.DataFrame, crops: pd.DataFrame, root: Path, path: Path, n=12):
    import cv2
    tiles = []
    for _, r in rows.head(n).iterrows():
        c = crops[(crops.rep_id == r.rep_id) & (crops.offset == 0)]
        c = c if len(c) else crops[crops.rep_id == r.rep_id]
        img = cv2.imread(str(root / c.iloc[0].image_path))
        img = cv2.resize(img, (224, 224))
        cv2.rectangle(img, (0, 0), (224, 40), (0, 0, 0), -1)
        cv2.putText(img, f"T:{split_class(r.class_name)[1]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img, f"P:{split_class(r.pred_class)[1]} {r.pred_prob:.2f}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(path), grid)


def evaluate(run: Path, crops: pd.DataFrame, root: Path) -> dict:
    import keras
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
    model = keras.models.load_model(run / "model.keras")
    test = crops[crops.split == "test"].reset_index(drop=True)
    probs = masked(predict(model, test, root), list(test.exercise))
    crop_pred = [class_names()[i] for i in probs.argmax(1)]
    reps = rep_level(test, probs)
    reps.to_csv(run / "pred_test.csv", index=False)
    y, yp = reps.class_name, reps.pred_class
    p, r, f, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    m = {"run": run.name, "n_test_reps": len(reps), "n_test_crops": len(test),
         "rep_accuracy": accuracy_score(y, yp), "rep_macro_precision": p, "rep_macro_recall": r, "rep_macro_f1": f,
         "crop_accuracy": accuracy_score(test.class_name, crop_pred),
         "per_exercise_accuracy": {ex: accuracy_score(g.class_name, g.pred_class) for ex, g in reps.groupby("exercise")},
         "per_participant_accuracy": {pp: accuracy_score(g.class_name, g.pred_class) for pp, g in reps.groupby("participant")},
         "binary_error_vs_correct_f1": f1_score(~y.str.endswith("__correct"), ~yp.str.endswith("__correct"), zero_division=0)}
    (run / "metrics.json").write_text(json.dumps(m, indent=1))
    (run / "report.txt").write_text(classification_report(y, yp, zero_division=0))
    plot_confusions(reps, run / "confusion.png")
    plot_curves(run / "history.json", run / "curves.png")
    ok = reps[y == yp].sort_values("pred_prob", ascending=False)
    bad = reps[y != yp].sort_values("pred_prob", ascending=False)  # confident mistakes first
    gallery(ok, crops, root, run / "success.jpg")
    gallery(bad, crops, root, run / "failure.jpg")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--export-preds", type=Path, help="run dir whose model predicts every split for Gemini data")
    args = ap.parse_args()
    crops = pd.read_csv(args.data / "crops.csv")
    results = [evaluate(r, crops, args.data) for r in args.runs]
    lines = ["| run | rep acc | macro P | macro R | macro F1 | error-vs-correct F1 | crop acc |", "|---|---|---|---|---|---|---|"]
    for m in results:
        lines.append(f"| {m['run']} | {m['rep_accuracy']:.3f} | {m['rep_macro_precision']:.3f} | {m['rep_macro_recall']:.3f} | "
                     f"{m['rep_macro_f1']:.3f} | {m['binary_error_vs_correct_f1']:.3f} | {m['crop_accuracy']:.3f} |")
    table = "\n".join(lines)
    (args.runs[0].parent / "results_model_a.md").write_text("# Model A test results (unseen participants)\n\n" + table + "\n")
    print(table)

    if args.export_preds:
        import keras
        model = keras.models.load_model(args.export_preds / "model.keras")
        probs = masked(predict(model, crops, args.data), list(crops.exercise))
        rep_level(crops, probs).to_csv(args.export_preds / "preds_all.csv", index=False)
        print(f"Wrote {args.export_preds / 'preds_all.csv'}")


if __name__ == "__main__":
    main()
