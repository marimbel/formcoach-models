#!/usr/bin/env python3
"""Build Model B (Gemini) training situations from real reps + Model A predictions.

    python scripts/06_build_gemini_situations.py --data data/processed \
        --preds runs/finetune/preds_all.csv --out data/gemini/situations.csv

Each row = one "rep just finished" situation (the exact evidence the app sends to the backend) plus a
DRAFT coaching reply from formcoach/coaching_kb.py. Reviewers then fill the `final_json` column
(see docs/gemini_labeling_guide.md). Splits follow the participant split, so test-split
situations come only from unseen participants.

Situations are grouped into short pseudo-sessions per participant+exercise so the `history`
field is realistic (repeated errors, corrections). Extra case types are added on purpose:
  low_visibility -> should reposition;  uncertain -> flattened probabilities, should hold;
  conflict       -> Model A label swapped to an error the geometry doesn't support.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.coaching_kb import draft_response  # noqa: E402
from formcoach.labels import errors_for  # noqa: E402


def model_a_block(exercise: str, probs: dict[str, float]) -> dict:
    errs = errors_for(exercise)
    p = {e: round(float(probs.get(e, 0.0)), 3) for e in errs}
    top = max(p, key=p.get)
    return {"top_error": top, "top_prob": p[top], "probs": p}


def simulated_probs(true_err: str, exercise: str, rng: np.random.Generator) -> dict:
    """Used only when real Model A predictions are unavailable (e.g. before Model A is trained)."""
    errs = errors_for(exercise)
    label = true_err if rng.random() > 0.15 else rng.choice([e for e in errs if e != true_err])
    conf = rng.uniform(0.5, 0.95)
    rest = rng.dirichlet(np.ones(len(errs) - 1)) * (1 - conf)
    out, k = {}, 0
    for e in errs:
        if e == label:
            out[e] = conf
        else:
            out[e] = float(rest[k])
            k += 1
    return out


def flatten(probs: dict, rng) -> dict:
    t = rng.uniform(3.0, 6.0)
    v = np.array(list(probs.values())) + 1e-6
    v = v ** (1 / t)
    v /= v.sum()
    return dict(zip(probs.keys(), v.tolist()))


def detected(ma: dict) -> str:
    return ma["top_error"] if ma["top_prob"] >= 0.5 else "uncertain"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--preds", type=Path, help="preds_all.csv from 04_eval_model_a.py --export-preds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--session-len", type=int, default=5)
    ap.add_argument("--p-lowvis", type=float, default=0.08)
    ap.add_argument("--p-uncertain", type=float, default=0.08)
    ap.add_argument("--p-conflict", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=663)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    reps = pd.read_csv(args.data / "reps.csv")
    crops = pd.read_csv(args.data / "crops.csv")
    bottom_img = crops[crops.offset == 0].drop_duplicates("rep_id").set_index("rep_id").image_path.to_dict()
    preds = None
    if args.preds:
        preds = pd.read_csv(args.preds).set_index("rep_id")
    else:
        print("WARNING: no --preds; simulating Model A probabilities. Rebuild once Model A is trained.")

    rows = []
    for (pid, ex), g in reps.groupby(["participant", "exercise"]):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        # put same-error reps next to each other sometimes, so repeats/corrections occur
        if rng.random() < 0.6:
            g = g.sort_values("error", key=lambda s: s.map({e: rng.random() for e in s.unique()}))
        recs = g.to_dict("records")
        for s0 in range(0, len(recs), args.session_len):
            history: list[str] = []
            for i, r in enumerate(recs[s0:s0 + args.session_len]):
                if preds is not None and r["rep_id"] in preds.index:
                    pr = preds.loc[r["rep_id"]]
                    probs = {e: float(pr.get(f"{ex}__{e}", 0.0)) for e in errors_for(ex)}
                    source = "model_a"
                else:
                    probs = simulated_probs(r["error"], ex, rng)
                    source = "simulated"
                feats = json.loads(r["features"])
                view = feats.pop("view", "unknown")
                case = "natural"
                vis, req_vis = float(r["visibility"]), bool(r["required_visible"])
                u = rng.random()
                if u < args.p_lowvis:
                    case, vis, req_vis = "low_visibility", float(rng.uniform(0.25, 0.55)), False
                elif u < args.p_lowvis + args.p_uncertain:
                    case, probs = "uncertain", flatten(probs, rng)
                elif u < args.p_lowvis + args.p_uncertain + args.p_conflict:
                    others = [e for e in errors_for(ex) if e not in ("correct", r["error"])]
                    if others:
                        fake = prng.choice(others)
                        conf = float(rng.uniform(0.55, 0.9))
                        probs = {e: (conf if e == fake else (1 - conf) / (len(errors_for(ex)) - 1)) for e in errors_for(ex)}
                        case = "conflict"
                ma = model_a_block(ex, probs)
                situation = {"exercise": ex, "rep_index": i + 1, "phase": "bottom", "view": view,
                             "pose_quality": {"visibility": round(vis, 2), "required_visible": req_vis},
                             "model_a": ma, "pose_features": feats, "history": history[-3:]}
                draft = draft_response(situation, prng)
                rows.append({"example_id": f"{r['rep_id']}_s{s0 // args.session_len}_{case}", "split": r["split"],
                             "participant": pid, "exercise": ex, "true_error": r["error"], "case_type": case,
                             "model_a_source": source, "rep_id": r["rep_id"],
                             "image_path": bottom_img.get(r["rep_id"], ""),
                             "situation_json": json.dumps(situation), "draft_json": json.dumps(draft),
                             "final_json": "", "reviewer": "", "notes": ""})
                if case != "low_visibility":
                    history.append(detected(ma))
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["split", "case_type"]).size().unstack(fill_value=0).to_string())
    print(f"\nWrote {len(out)} situations -> {args.out}. Next: review final_json, then 07_situations_to_jsonl.py")


if __name__ == "__main__":
    main()
