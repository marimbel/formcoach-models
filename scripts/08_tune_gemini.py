#!/usr/bin/env python3
"""Launch a Vertex AI supervised fine-tuning job for Model B and wait for it.

    gcloud storage cp data/gemini/train.jsonl data/gemini/val.jsonl gs://YOUR_BUCKET/formcoach/gemini/
    python scripts/08_tune_gemini.py --project YOUR_PROJECT --location us-central1 \
        --train gs://YOUR_BUCKET/formcoach/gemini/train.jsonl --val gs://YOUR_BUCKET/formcoach/gemini/val.jsonl

Check the current list of tunable models in the Vertex AI docs before choosing --base-model; Google
retires model versions often. Auth: `gcloud auth application-default login` (or run in Colab).
Writes runs/gemini/tuning_job.json with the tuned endpoint to put in the backend's TUNED_MODEL env var.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="us-central1")
    ap.add_argument("--base-model", default="gemini-2.5-flash")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr-multiplier", type=float, default=1.0)
    ap.add_argument("--adapter-size", type=int, choices=[1, 2, 4, 8, 16], default=4)
    ap.add_argument("--name", default="formcoach-coach-v1")
    ap.add_argument("--out", type=Path, default=Path("runs/gemini/tuning_job.json"))
    args = ap.parse_args()

    from google import genai
    from google.genai import types

    sizes = {1: "ADAPTER_SIZE_ONE", 2: "ADAPTER_SIZE_TWO", 4: "ADAPTER_SIZE_FOUR", 8: "ADAPTER_SIZE_EIGHT",
             16: "ADAPTER_SIZE_SIXTEEN"}
    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    cfg = types.CreateTuningJobConfig(
        tuned_model_display_name=args.name, epoch_count=args.epochs,
        learning_rate_multiplier=args.lr_multiplier, adapter_size=types.AdapterSize(sizes[args.adapter_size]),
        validation_dataset=types.TuningValidationDataset(gcs_uri=args.val) if args.val else None)
    job = client.tunings.tune(base_model=args.base_model, training_dataset=types.TuningDataset(gcs_uri=args.train),
                              config=cfg)
    print("Started", job.name)
    done = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while str(job.state.name if hasattr(job.state, "name") else job.state) not in done:
        time.sleep(60)
        job = client.tunings.get(name=job.name)
        print(time.strftime("%H:%M:%S"), job.state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    info = {"job": job.name, "state": str(job.state), "base_model": args.base_model,
            "tuned_model": getattr(job.tuned_model, "model", None) if job.tuned_model else None,
            "endpoint": getattr(job.tuned_model, "endpoint", None) if job.tuned_model else None,
            "epochs": args.epochs, "adapter_size": args.adapter_size, "lr_multiplier": args.lr_multiplier}
    args.out.write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print("Training/validation loss curves: Vertex AI console > Tuning > this job (export them for the report).")


if __name__ == "__main__":
    main()
