# Workflow

There are 15 days to the Dec 7 submission (as of Nov 22, 2026), so recording has to start now. Each step names the notebook or script that does it.

| # | Step | Due | How |
| --- | --- | --- | --- |
| 1 | Record own data | Nov 27 | 8–10 people, all 13 classes, front + side. Name files `p03_squat_heel_lift_t1.mp4`. See [[Data Collection Protocol]] |
| 2 | Get public data | Nov 25 | `colab_model_a.ipynb` downloads Zenodo and checks its md5. Mendeley needs a manual "Download All", then upload the zip to Drive |
| 3 | Label Mendeley and REHAB24-6 incorrect clips | Nov 28 | Fill `review.csv` from the contact sheets, then `apply-review` ([[Data Sources]]) |
| 4 | Extract and split | Nov 28 | `01_extract_reps` → `01b_extract_images` → `02_make_splits`, pinning the chosen test people |
| 5 | Train and test Model A | Nov 30 | `03` (probe, then fine-tune), `04`, `04b`, `05` to TFLite; copy the `.tflite` into the app |
| 6 | Build and review Gemini examples | Dec 1 | `colab_gemini.ipynb` runs `06`; split reviewing across the three of us; `07` + upload to Cloud Storage |
| 7 | Tune Model B | Dec 2 | `08_tune_gemini.py`; the job takes a few hours. Put the endpoint in the backend's settings |
| 8 | Evaluate B, C, D | Dec 4 | `09_eval_gemini.py`, blind rating sheet, pose and image checks ([[Models C and D]]) |
| 9 | Data report and write-up | Dec 6 | `10_data_report.py`, paste the tables, record the demo |
| 10 | Submit | Dec 7 | |

## Script order

```
00_import_public_data → 01_extract_reps → 01b_extract_images → 02_make_splits
→ 03_train_model_a → 04_eval_model_a / 04b_feature_baseline → 05_export_tflite
→ 06_build_gemini_situations → human review → 07_situations_to_jsonl
→ 08_tune_gemini → 09_eval_gemini → 10_data_report
```

Run `01b` right after `01` and before `02`: `02` adds a split column and `01b` refuses to append after that.
