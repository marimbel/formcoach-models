# Model A data card

One example is one body crop at the bottom of a rep. Each rep gives up to 3 crops, and every metric is reported per rep. The target is 8–10 participants × about 80 reps, roughly 700 own reps before public data.

| Item | Answer |
| --- | --- |
| Training data | Own reps from train participants, plus the Zenodo, Mendeley and REHAB24-6 examples placed in train |
| Validation data | Own reps from val participants, plus about 15% of the public groups; used for early stopping and picking thresholds |
| Test data | Own reps from 1–2 participants who appear nowhere else; never used for training or tuning |
| Where it came from | Our recordings (see [[Data Sources]]) and the public sets (Zenodo, Mendeley, REHAB24-6) |
| How it was collected | Phone on a tripod at hip height, 2–3 m away, front and side views. Each person does about 8 reps per class, acting out each error after a demo. See [[Data Collection Protocol]] |
| Number of examples | Filled from `docs/DATA_REPORT.md`, made by `10_data_report.py` from the files actually trained on |
| Format | 224×224 RGB JPEG crops (saved at 256×256); `reps.csv` (one row per rep, with joint-angle features as JSON); `crops.csv` (one row per crop) |
| Labels | 13 classes, `exercise__error`: squat correct / knee_valgus / forward_lean / insufficient_depth / heel_lift; lunge correct / knee_alignment / forward_lean / insufficient_depth; push_up correct / hip_sag / elbow_flare / partial_rom. The label comes from the file name of the take, set when it was recorded |
| Preprocessing | MediaPipe pose → One Euro smoothing → rep segmentation on the main joint angle → 3 frames around the deepest point → square crop with 15% margin, black padding → resize to 256, random crop to 224. Training adds zoom 0.85–1.0, horizontal flip and brightness/contrast jitter. Reps where the needed joints are under 0.5 visibility are dropped |
| Split | By participant, never by rep: own people 70/15/15 train/val/test; public groups 85/15 train/val only |
| Limitations | Acted errors are cleaner than real ones. Few participants, so per-class counts are small. Zenodo images are side view only. Public lunge and push-up data is labeled only correct/incorrect, so we add the error type by hand |
| Potential biases | Participants are mostly college-age students of a narrow range of body types; few older or heavier people. Indoor lighting and plain backgrounds. Public data skews squat-heavy, so squat may score better than the other two exercises |

Report per-class precision, recall and F1, macro-F1 and a confusion matrix on test, for A and both baselines.

## Counts (fill in after running `10_data_report.py`)

| Split | Participants | Reps (own) | Reps (public) | Crops |
| --- | --- | --- | --- | --- |
| Train | | | | |
| Val | | | | |
| Test | | | 0 | |
