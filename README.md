# FormCoach — model development (CS 663 Project 2, Group 1)

Repository 2 of 2: data preparation, **Model A** training/evaluation, **Model B** (Gemini) fine-tuning
and evaluation. The mobile app and the Cloud Run backend are in **formcoach-app** (repository 1).

| Model | What | Where it runs | Customization | Code here |
|---|---|---|---|---|
| A | MobileNetV3-Large form-error classifier on bottom-of-rep body crops | on device (TFLite) | fine-tuned (vs. frozen-backbone baseline) | `03`–`05` |
| B | Gemini 3.5 Flash coach: evidence → structured coaching JSON | Vertex AI (via backend) | supervised fine-tuning (vs. base Gemini 3.5 Flash) | `06`–`09` |
| C | BlazePose landmarks (ML Kit on device; MediaPipe here for data prep) | on device | pretrained | `01` |
| D | Gemini 3.1 Flash Image, correct-form illustration | Vertex AI (via backend) | pretrained | in app repo |

## Documentation

The full models and data documentation (model choices, data sources, data cards, test-set separation, workflow)
is in [`docs/wiki/`](docs/wiki/Home.md) and on the [project wiki](https://github.com/marimbel/formcoach-models/wiki).
**Start here: [Training guide](docs/wiki/Training-Guide.md)**, step by step from data to a working app.
To update the wiki after editing `docs/wiki/`, run `bash scripts/publish_wiki.sh` from your own computer.

## Data sources

| Source | What | License | Used for |
|---|---|---|---|
| **Our recordings** (`p01`, `p02`, …) | squat, lunge, push-up; all 13 classes; 2 views | ours (consent notes) | train, val, **test (only source)** |
| Zenodo 17558630 squat images (`zsq`) | side-view squats: Good / Bad Back / Bad Heel | CC BY 4.0 | train, val |
| Mendeley kgbb3yn47p v2 videos (`mdl`) | bodyweight squats and push-ups, good/bad form, 3 views, 26 subjects; bad clips hand-labelled by us | CC BY 4.0 | train, val |
| REHAB24-6, Zenodo 13305826 (`reh`) | lunges, push-ups (and squats): every rep marked correct/incorrect, 2 cameras, 10 adults; incorrect reps hand-labelled by us | CC BY-NC 4.0 (non-commercial) | train, val |

Exercises: **squat, lunge, push-up**. Push-up replaced the proposal's chest press because chest press has no
public form data, while push-ups have two public video sets and need no equipment to record.

`02_make_splits.py` keeps every public-data participant out of the test set. `10_data_report.py` writes the
counts per split / class / source for the report. Step-by-step: `notebooks/colab_model_a.ipynb` (GPU) and
`notebooks/colab_gemini.ipynb`.

## Pipeline

```
public downloads ──00──▶ raw videos / image manifest
raw videos ──01──▶ landmarks + reps.csv + crops/ ──02──▶ participant splits
                                            │
                       ┌────────────────────┴───────────────────┐
                 03 train Model A (probe + finetune)       04b skeleton-only baseline
                 04 evaluate on unseen participants
                 05 export TFLite ──▶ app
                       │ preds_all.csv
                 06 build Gemini situations (+ drafts) ──▶ human review ──07──▶ JSONL
                 08 tune Gemini on Vertex AI ──▶ 09 base vs tuned eval
                 10 data report (counts for the documentation)
```

## Pilot run before your own recordings

You can train Model A on the public squat data alone to test the pipeline (see the "Pilot run" cells in
`notebooks/colab_model_a.ipynb`): import Zenodo/Mendeley, run `01b` (and `01` if `raw/` has clips), run
`02_make_splits.py` with no `--pin`, train with `03`, and evaluate with `--eval-split val`:

    python scripts/04_eval_model_a.py --data data/processed --runs runs/probe runs/finetune --eval-split val
    python scripts/04b_feature_baseline.py --data data/processed --eval-split val

These are validation numbers on squats only, never the final results. Once your recordings exist, re-run from `01`
(it rewrites `reps.csv`), split with pinned test participants, and evaluate on `test`.

## Setup

```bash
pip install -r requirements.txt
python -m pytest -q tests              # unit tests: geometry, reps, splits, prompt contract
```
Training needs a GPU: use `notebooks/colab_model_a.ipynb`. Gemini steps need a GCP project with Vertex AI
enabled and `gcloud auth application-default login`.

## Step by step

1. **Record** following `docs/data_collection_protocol.md`; name clips `p03_squat_knee_valgus_t1.mp4`.
2. **Extract** reps and crops; then **open every QA plot** in `data/processed/qa/` and delete/re-record bad clips.
   ```bash
   python scripts/01_extract_reps.py --videos data/raw --out data/processed
   python scripts/02_make_splits.py --data data/processed --pin p09=test   # pin a non-member as unseen test user
   ```
3. **Model A** (both runs are needed: the course requires the pretrained comparison):
   ```bash
   python scripts/03_train_model_a.py --data data/processed --mode probe    --out runs/probe
   python scripts/03_train_model_a.py --data data/processed --mode finetune --out runs/finetune
   python scripts/04_eval_model_a.py  --data data/processed --runs runs/probe runs/finetune --export-preds runs/finetune
   python scripts/04b_feature_baseline.py --data data/processed
   python scripts/05_export_tflite.py --run runs/finetune --data data/processed --fp16
   ```
   Copy `runs/finetune/form_classifier.tflite` → `formcoach-app/mobile/assets/models/`.
4. **Model B data**: build situations, then reviewers fill `final_json` (`docs/gemini_labeling_guide.md`).
   ```bash
   python scripts/06_build_gemini_situations.py --data data/processed --preds runs/finetune/preds_all.csv --out data/gemini/situations.csv
   # ... review in Google Sheets, export as data/gemini/situations_reviewed.csv ...
   gcloud storage cp -r data/processed/crops gs://BUCKET/formcoach/
   python scripts/07_situations_to_jsonl.py --situations data/gemini/situations_reviewed.csv --out data/gemini --image-gcs-prefix gs://BUCKET/formcoach/crops
   gcloud storage cp data/gemini/train.jsonl data/gemini/val.jsonl gs://BUCKET/formcoach/gemini/
   ```
5. **Tune + evaluate Gemini**:
   ```bash
   python scripts/08_tune_gemini.py --project PROJECT --train gs://BUCKET/formcoach/gemini/train.jsonl --val gs://BUCKET/formcoach/gemini/val.jsonl
   python scripts/09_eval_gemini.py --project PROJECT --test data/gemini/test_eval.jsonl --data data/processed --tuned <endpoint from runs/gemini/tuning_job.json>
   ```
   Put the endpoint in the backend's `TUNED_MODEL` env var. Whenever `formcoach/prompting.py` changes, copy it to
   the backend and run `python scripts/check_prompting_sync.py ../formcoach-app`.

## Design notes worth putting in the report

- **Why crops at the bottom of each rep?** Most errors (depth, valgus, lean, flare) are defined at the bottom.
  Model C's landmarks find that frame (`formcoach/reps.py`), so Model A sees one informative crop instead of
  every frame. The app averages 3 crops around the bottom, and so does evaluation.
- **Why participant-level splits?** Frames from the same person look alike; mixing them inflates test accuracy.
- **Masking:** the user picks the exercise, so Model A's 13 outputs are restricted to that exercise's classes.
- **Shared contracts:** `configs/labels.json` (classes + rep thresholds) and `formcoach/prompting.py` (Gemini
  format) are used by both repos. Geometry is mirrored in `mobile/lib/vision/geometry.dart`.
- **Smoke test without real data:** `python tests/make_synthetic_dataset.py /tmp/fc && python scripts/01_extract_reps.py --videos /tmp/fc/raw --out /tmp/fc/processed --reuse-landmarks`
  then steps 02–07 with `--weights none` (stick figures only, for checking that the code runs).
