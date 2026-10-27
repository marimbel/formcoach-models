#!/usr/bin/env python3
"""Base Gemini vs fine-tuned Gemini on the held-out test situations (unseen participants).

    python scripts/09_eval_gemini.py --project YOUR_PROJECT --location us-central1 \
        --test data/gemini/test_eval.jsonl --data data/processed \
        --base gemini-2.5-flash --tuned projects/.../endpoints/...   --out runs/gemini/eval

Both models get the identical system instruction, evidence and image, in the same order as the
training examples. Automatic metrics (vs. the human-reviewed reference): valid JSON, schema
compliance, action accuracy (coach/praise/reposition/hold), error accuracy, cue length, number
grounding (are quoted measurements real?) and latency, overall and per case type.
Also writes rating_sheet.csv: blind A/B pairs for group members to score clarity and correctness 1-5.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.gemini_eval import score, summarize  # noqa: E402
from formcoach.prompting import SYSTEM_INSTRUCTION, parse_response, render_user_prompt  # noqa: E402


def call(client, types, model: str, situation: dict, image: bytes | None, json_mode: bool, thinking_budget: int | None):
    parts = []
    if image:
        parts.append(types.Part.from_bytes(data=image, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=render_user_prompt(situation)))
    cfg = dict(system_instruction=SYSTEM_INSTRUCTION, temperature=0.2, max_output_tokens=512)
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    for attempt in range(4):
        try:
            t0 = time.perf_counter()
            resp = client.models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)],
                                                  config=types.GenerateContentConfig(**cfg))
            return resp.text or "", (time.perf_counter() - t0) * 1000
        except Exception as e:  # rate limits etc.
            wait = 2 ** attempt * 5
            print(f"  {model[:40]}: {type(e).__name__} {str(e)[:120]}; retry in {wait}s")
            time.sleep(wait)
    return "", None


def plot(summary: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["json_valid", "schema_ok", "action_match", "error_match", "cue_len_ok", "number_grounding_rate"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.38
    for k, (name, m) in enumerate(summary.items()):
        vals = [m["overall"].get(x) or 0 for x in keys]
        ax.bar([i + k * w for i in range(len(keys))], vals, w, label=name)
    ax.set_xticks([i + w / 2 for i in range(len(keys))], [k.replace("_", "\n") for k in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model B: base vs fine-tuned Gemini (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--data", type=Path, help="processed data root, to attach the crop images")
    ap.add_argument("--base", default="gemini-2.5-flash")
    ap.add_argument("--tuned", required=True, help="tuned endpoint from runs/gemini/tuning_job.json")
    ap.add_argument("--base-json-mode", action=argparse.BooleanOptionalAction, default=True,
                    help="give the base model JSON mode (a stronger, fairer baseline)")
    ap.add_argument("--thinking-budget", type=int, default=0, help="0 = no thinking (lower latency); -1 = omit setting")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("runs/gemini/eval"))
    args = ap.parse_args()

    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    tb = None if args.thinking_budget < 0 else args.thinking_budget
    tests = [json.loads(l) for l in args.test.read_text().splitlines() if l.strip()][: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    models = {"base": (args.base, args.base_json_mode), "tuned": (args.tuned, False)}
    rows = {k: [] for k in models}
    raw = open(args.out / "responses.jsonl", "w")
    for i, ex in enumerate(tests):
        img = None
        if args.data and ex.get("image_path"):
            p = args.data / ex["image_path"]
            img = p.read_bytes() if p.exists() else None
        for name, (model, jm) in models.items():
            text, ms = call(client, types, model, ex["situation"], img, jm, tb)
            obj, problems = parse_response(text)
            s = score(ex["reference"], obj, problems, ex["situation"])
            s.update(latency_ms=ms, case_type=ex["case_type"], example_id=ex["example_id"])
            rows[name].append(s)
            raw.write(json.dumps({"model": name, "example_id": ex["example_id"], "text": text, "problems": problems,
                                  **s}) + "\n")
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(tests)}")
    raw.close()

    summary = {}
    for name, rs in rows.items():
        by_case = {}
        for c in sorted({r["case_type"] for r in rs}):
            by_case[c] = summarize([r for r in rs if r["case_type"] == c])
        summary[name] = {"overall": summarize(rs), "by_case_type": by_case}
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=1))
    plot(summary, args.out / "base_vs_tuned.png")

    # Blind A/B rating sheet
    resp = {}
    for l in (args.out / "responses.jsonl").read_text().splitlines():
        d = json.loads(l)
        resp.setdefault(d["example_id"], {})[d["model"]] = d["text"]
    rng = random.Random(663)
    with open(args.out / "rating_sheet.csv", "w", newline="") as f, open(args.out / "rating_key.csv", "w", newline="") as k:
        w, kw = csv.writer(f), csv.writer(k)
        w.writerow(["example_id", "evidence", "reply_A", "reply_B", "A_correct_1to5", "A_clear_1to5",
                    "B_correct_1to5", "B_clear_1to5", "preferred_A_or_B", "rater"])
        kw.writerow(["example_id", "A", "B"])
        for ex in tests:
            pair = ["base", "tuned"]
            rng.shuffle(pair)
            r = resp.get(ex["example_id"], {})
            w.writerow([ex["example_id"], render_user_prompt(ex["situation"]), r.get(pair[0], ""), r.get(pair[1], ""),
                        "", "", "", "", "", ""])
            kw.writerow([ex["example_id"], *pair])
    print(json.dumps({k: v["overall"] for k, v in summary.items()}, indent=1))
    print(f"Outputs in {args.out}. Give rating_sheet.csv to raters; keep rating_key.csv hidden until scoring.")


if __name__ == "__main__":
    main()
