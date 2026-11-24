# Data sources

Our own recordings are the core dataset and the only test data. Public datasets add training and validation examples for all three exercises: squat, lunge and push-up. Push-up replaced the proposal's chest press because chest press has no public form data.

| Source | What it has | License | Used for | Prefix |
| --- | --- | --- | --- | --- |
| **Own recordings** (Marianna, Brent, Nico + 5–7 volunteers) | Phone video of squats, lunges and push-ups; every one of the 13 classes performed on purpose, front and side views | Consent form; group use only | Train, val **and test** (A and B) | `p01`, `p02`, … |
| [Zenodo 17558630 Squat Dataset](https://zenodo.org/records/17558630) (Teng 2025) | Side-view squat images in 3 folders: Good, Bad Back, Bad Heel (824 MB zip) | CC BY 4.0 | Train/val, mapped to squat correct / forward_lean / heel_lift | `zsq` |
| [Mendeley Multi-View Raw Video Dataset of Seven Fitness Exercises](https://data.mendeley.com/datasets/kgbb3yn47p/2) (Mashetty et al. 2026) | 26 subjects, 3 views, good/bad form labels; bodyweight squats and push-ups used | CC BY 4.0 | Train/val; good clips as correct, bad clips hand-labeled with an error type | `mdl` |
| [REHAB24-6](https://zenodo.org/records/13305826) (Zenodo 13305826, 2024) | Lunges, push-ups and squats; every rep marked correct or incorrect, 2 cameras, 10 adults | CC BY-NC 4.0 (non-commercial) | Train/val; correct runs as correct, incorrect runs hand-labeled with an error type | `reh` |
| [Fitness-AQA](https://github.com/ParitoshParmar/Fitness-AQA) (Parmar et al., ECCV 2022), optional | Back squat knees inward / forward / shallow, from online videos | Non-commercial, request access | Optional extra squat train data | `faq` |
| [Zenodo 18859797](https://zenodo.org/records/18859797) (Dai 2026), optional | MediaPipe keypoints for standard, shallow and good-morning squats, 22 subjects; no images | CC BY 4.0 | Optional extra rows for the skeleton baseline only | `pub` |

Not used, and why: EC3D (squat and lunge errors) has only 4 subjects, 3D poses and no public download. QEVD and FLEX are large but don't label our error types, or aren't released. Roboflow SquatsForm overlaps the Zenodo images.

## Importing them

```bash
# Zenodo images (one folder per class) -> manifest -> crops + features
python scripts/00_import_public_data.py images data/public/zenodo_squat --prefix zsq \
    --map "good=squat:correct" --map "bad back=squat:forward_lean" --map "bad heel=squat:heel_lift" \
    --out data/public/zenodo_squat_manifest.csv
python scripts/01b_extract_images.py --manifest data/public/zenodo_squat_manifest.csv --prefix zsq --out data/processed

# Mendeley videos (manual "Download All" first)
python scripts/00_import_public_data.py inspect data/public/mendeley
python scripts/00_import_public_data.py videos data/public/mendeley --exercise squat \
    --keyword squat --good good --bad bad --prefix mdl --out data/raw --review data/review/mendeley
# fill the `error` column of data/review/mendeley/review.csv, then:
python scripts/00_import_public_data.py apply-review data/review/mendeley/review.csv --out data/raw

# Mendeley push-ups: same command, own review folder
python scripts/00_import_public_data.py videos data/public/mendeley --exercise push_up \
    --keyword push --good good --bad bad --prefix mdl --out data/raw --review data/review/mendeley_push_up

# REHAB24-6 lunges (exercise 5) and push-ups (exercise 3)
python scripts/00_import_public_data.py rehab data/public/rehab24/videos \
    --segmentation data/public/rehab24/Segmentation.csv --map 5=lunge --map 3=push_up \
    --prefix reh --out data/raw --review data/review/rehab
# label data/review/rehab/<exercise>/review.csv, then apply-review on each
```

The Mendeley bad clips say "bad" but not which error. The `videos` command makes a 6-frame contact sheet per clip. A group member labels each with an error type for its exercise (squat: knee_valgus, forward_lean, insufficient_depth, heel_lift; lunge: knee_alignment, forward_lean, insufficient_depth; push-up: hip_sag, elbow_flare, partial_rom) or skip. REHAB24-6 incorrect reps go through the same review. `apply-review` copies the labeled clips in.
