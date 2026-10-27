#!/usr/bin/env python3
"""Assign participants to train/val/test (70/15/15 by sample count) and tag reps.csv + crops.csv.

    python scripts/02_make_splits.py --data data/processed --pin p09=test

Pin at least one participant who is NOT a group member to test, so the test set is a truly
unseen user (the course requires test examples that were never used for training).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.splits import participant_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--pin", action="append", default=[], help="participant=split, repeatable")
    ap.add_argument("--seed", type=int, default=663)
    args = ap.parse_args()

    reps = pd.read_csv(args.data / "reps.csv")
    crops = pd.read_csv(args.data / "crops.csv")
    ov_path = args.data / "rep_overrides.csv"  # columns: rep_id,new_error  (new_error = drop to remove)
    if ov_path.exists():
        ov = pd.read_csv(ov_path).set_index("rep_id").new_error.to_dict()
        drop = {r for r, e in ov.items() if e == "drop"}
        for df in (reps, crops):
            df.drop(df[df.rep_id.isin(drop)].index, inplace=True)
            m = df.rep_id.isin(ov)
            df.loc[m, "error"] = df.loc[m, "rep_id"].map(ov)
            df.loc[m, "class_name"] = df.loc[m, "exercise"] + "__" + df.loc[m, "error"]
        print(f"Applied {len(ov)} overrides ({len(drop)} dropped) from {ov_path.name}")
    pinned = dict(p.split("=", 1) for p in args.pin)
    split = participant_split(reps.groupby("participant").size().to_dict(), seed=args.seed, pinned=pinned)
    reps["split"] = reps.participant.map(split)
    crops["split"] = crops.participant.map(split)
    reps.to_csv(args.data / "reps.csv", index=False)
    crops.to_csv(args.data / "crops.csv", index=False)
    (args.data / "splits.json").write_text(json.dumps(split, indent=2))

    print(pd.crosstab(reps.class_name, reps.split, margins=True).to_string())
    print("\nParticipants:", json.dumps(split, indent=1))
    missing = reps.groupby("split").class_name.nunique()
    n_cls = reps.class_name.nunique()
    for s, n in missing.items():
        if n < n_cls:
            print(f"WARNING: split '{s}' has {n}/{n_cls} classes; record more participants or re-pin.")


if __name__ == "__main__":
    main()
