#!/usr/bin/env python3
"""Reviewed situations.csv -> Vertex AI Gemini tuning JSONL (+ our own test file).

    python scripts/07_situations_to_jsonl.py --situations data/gemini/situations_reviewed.csv \
        --out data/gemini --image-gcs-prefix gs://YOUR_BUCKET/formcoach/crops

Writes:
  train.jsonl, val.jsonl   Vertex supervised-tuning format (systemInstruction + contents)
  test_eval.jsonl          held-out situations + reference reply + local image path, for 09_eval_gemini.py

Upload the crops folder to the same GCS prefix first:
  gcloud storage cp -r data/processed/crops gs://YOUR_BUCKET/formcoach/
Without --image-gcs-prefix the examples are text-only (pose evidence only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.prompting import SYSTEM_INSTRUCTION, parse_response, render_target, render_user_prompt  # noqa: E402


def gcs_uri(prefix: str, image_path: str) -> str:
    # image_path is like "crops/<video>/rep00_o+0.jpg"; prefix points at the folder that contains "crops/"
    rel = image_path.split("crops/", 1)[-1]
    return prefix.rstrip("/") + "/" + rel


def record(situation: dict, target: dict, image_uri: str | None) -> dict:
    parts = []
    if image_uri:
        parts.append({"fileData": {"mimeType": "image/jpeg", "fileUri": image_uri}})
    parts.append({"text": render_user_prompt(situation)})
    return {"systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": parts},
                         {"role": "model", "parts": [{"text": render_target(target)}]}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--situations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-gcs-prefix", help="gs:// prefix that contains the uploaded crops")
    ap.add_argument("--allow-unreviewed-test", action="store_true",
                    help="use drafts as test references (NOT recommended: test labels should be human-checked)")
    args = ap.parse_args()

    df = pd.read_csv(args.situations, keep_default_na=False)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {}
    files = {s: open(args.out / f"{s}.jsonl", "w") for s in ("train", "val")}
    test_f = open(args.out / "test_eval.jsonl", "w")
    for _, r in df.iterrows():
        reviewed = bool(str(r.final_json).strip())
        target_text = r.final_json if reviewed else r.draft_json
        target, problems = parse_response(target_text)
        if target is None or problems:
            print(f"SKIP {r.example_id}: target invalid {problems}")
            continue
        s = r.split
        st = stats.setdefault(s, {"reviewed": 0, "draft": 0})
        st["reviewed" if reviewed else "draft"] += 1
        situation = json.loads(r.situation_json)
        uri = gcs_uri(args.image_gcs_prefix, r.image_path) if args.image_gcs_prefix and r.image_path else None
        if s in files:
            files[s].write(json.dumps(record(situation, target, uri)) + "\n")
        elif s == "test":
            if not reviewed and not args.allow_unreviewed_test:
                continue
            test_f.write(json.dumps({"example_id": r.example_id, "case_type": r.case_type, "true_error": r.true_error,
                                     "situation": situation, "reference": target, "image_path": r.image_path,
                                     "reviewed": reviewed}) + "\n")
    for f in (*files.values(), test_f):
        f.close()
    print(json.dumps(stats, indent=1))
    t = stats.get("train", {})
    if t and t.get("draft", 0) > t.get("reviewed", 0):
        print("NOTE: most training targets are unedited drafts; fine-tuning will mostly copy the template style.")
    if stats.get("test", {}).get("reviewed", 0) == 0 and not args.allow_unreviewed_test:
        print("WARNING: test_eval.jsonl is empty because no test rows are reviewed yet.")


if __name__ == "__main__":
    main()
