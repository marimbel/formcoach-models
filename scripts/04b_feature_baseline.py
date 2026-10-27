#!/usr/bin/env python3
"""Skeleton-only baseline: logistic regression on the pose features (joint angles, ratios).

    python scripts/04b_feature_baseline.py --data data/processed

Answers a question the report should address: does the image model (Model A) add anything over
angles computed from Model C's landmarks? If this baseline is close to Model A, say so honestly
and discuss which errors need appearance (e.g. heel lift, subtle valgus) vs geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.labels import exercises  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    args = ap.parse_args()
    reps = pd.read_csv(args.data / "reps.csv")
    out = {}
    for ex in exercises():
        sub = reps[reps.exercise == ex]
        if sub.empty:
            continue
        feats = pd.DataFrame([json.loads(f) for f in sub.features]).drop(columns=["view"], errors="ignore")
        X = feats.apply(pd.to_numeric, errors="coerce").to_numpy(float)
        y = sub.error.to_numpy()
        tr, te = (sub.split == "train").to_numpy(), (sub.split == "test").to_numpy()
        if not tr.any() or not te.any():
            continue
        clf = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        out[ex] = {"accuracy": accuracy_score(y[te], yp), "macro_f1": f1_score(y[te], yp, average="macro"),
                   "n_test": int(te.sum())}
        print(f"== {ex} ==\n" + classification_report(y[te], yp, zero_division=0))
    (args.data / "feature_baseline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
