# FormCoach training guide: start to finish

This guide takes you from an empty Google account to a working Android app with your own trained models. Follow
the parts in order. Each step says what to click or run, how long it takes, and what you should see before moving on.

**Contents**

1. [What you are building](#1-what-you-are-building)
2. [Accounts and tools](#2-accounts-and-tools)
3. [Connect to Google Colab](#3-connect-to-google-colab)
4. [Set up Google Cloud](#4-set-up-google-cloud)
5. [The datasets and where to get them](#5-the-datasets-and-where-to-get-them)
6. [The models and their architectures](#6-the-models-and-their-architectures)
7. [Prepare the data](#7-prepare-the-data)
8. [Train Model A](#8-train-model-a)
9. [Fine-tune Model B](#9-fine-tune-model-b)
10. [Put the models into the app](#10-put-the-models-into-the-app)
11. [Numbers for the report](#11-numbers-for-the-report)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What you are building

```
 phone (Flutter app)                                         Google Cloud
 ┌──────────────────────────────────────────────────┐        ┌───────────────────────────────────────┐
 │ camera → C: ML Kit Pose (33 landmarks, every frame)│        │ Cloud Run backend (FastAPI)            │
 │        → rep counter + joint angles                │  per   │   → B: fine-tuned Gemini 3.5 Flash     │
 │        → 3 body crops at the bottom of each rep    │  rep   │        (Vertex AI endpoint)            │
 │        → A: MobileNetV3 form classifier (TFLite) ──┼──────▶ │   → D: Gemini 3.1 Flash Image          │
 │ ◀── spoken cue, alert card, Visual Trainer image ──┼─────── │        (when you tap "See correct form")│
 └──────────────────────────────────────────────────┘        └───────────────────────────────────────┘
```

| Model | What it is | Trained by you? | Where you train it | Output file / resource |
|---|---|---|---|---|
| **A** | MobileNetV3-Large image classifier, 13 form classes | **Yes** (fine-tuned) | Google Colab, GPU | `form_classifier.tflite` (goes in the app) |
| **B** | Gemini 3.5 Flash coach (evidence → JSON cue) | **Yes** (supervised fine-tuning) | Vertex AI, launched from Colab | a Vertex AI endpoint ID (goes in the backend) |
| **C** | ML Kit Pose / MediaPipe BlazePose | No (pretrained) | – | already in the app |
| **D** | Gemini 3.1 Flash Image | No (pretrained) | – | called by the backend |

**Order matters:** Model B learns from Model A's predictions, so you train A first.

**Time and cost, roughly**

| Stage | Your time | Machine time | Cost |
|---|---|---|---|
| Setup (this guide, parts 2–4) | 1 hour | – | $0 |
| Public data download + labeling | 3–5 hours (split three ways) | 45 min | $0 |
| Recording 8–10 people | 40 min per person | – | $0 |
| Model A | 1 hour | 1–2 hours on a Colab GPU | $0 (free Colab) |
| Model B review + tuning | 3 hours review (split three ways) | 30–90 min | about $18 |
| Backend + app hookup | 1 hour | 5 min | cents per day of use |

---

## 2. Accounts and tools

| You need | Why | Get it |
|---|---|---|
| A Google account | Drive, Colab and Google Cloud all use it | one per person; the person who owns the Cloud project shares it |
| Google Drive (15 GB free is enough) | Colab saves data and models here so nothing is lost when it disconnects | https://drive.google.com |
| Google Colab | Free GPU for training Model A, and a place to run every script | https://colab.research.google.com |
| Google Cloud project with billing | Vertex AI (Model B tuning, Model D), Cloud Storage, Cloud Run (backend) | https://console.cloud.google.com |
| Google Cloud CLI (`gcloud`) on your Mac | deploy the backend, manage the bucket | https://cloud.google.com/sdk/docs/install |
| Android Studio + Flutter | build and run the app | `formcoach-app/docs/platform_setup.md` |
| The two repos | the code | https://github.com/marimbel/formcoach-models and https://github.com/marimbel/formcoach-app |

Install `gcloud` on a Mac: download the macOS package from the install page above (or `brew install --cask google-cloud-sdk`
if you use Homebrew), then in Terminal:

```bash
gcloud init                                # sign in and pick (or create) your project
gcloud auth application-default login     # lets Python scripts on your Mac call Vertex AI
```

---

## 3. Connect to Google Colab

Colab is a Jupyter notebook in your browser running on a Google machine. Both notebooks are in the repo.

### 3.1 Open the notebook

1. Go to https://colab.research.google.com.
2. **File → Open notebook → GitHub** tab.
3. Type `marimbel/formcoach-models`, press Enter, and pick **`notebooks/colab_model_a.ipynb`**.
   (Later, the same way: `notebooks/colab_gemini.ipynb`.)
4. **File → Save a copy in Drive**, so your edits (like your participant IDs) are kept.

### 3.2 Turn on the GPU (Model A notebook only)

1. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
2. Check it in a new cell: `!nvidia-smi` should print a table with "Tesla T4".

The free tier gives a GPU when one is available and disconnects after a period of inactivity or a few hours of
use. That's fine: everything is saved to Drive, so you just reconnect and continue from the cell you were on.
The Gemini notebook doesn't need a GPU.

### 3.3 Connect Google Drive

Run the first cell. It runs `drive.mount('/content/drive')`: click **Connect to Google Drive**, pick your
account, and allow access. It creates this folder layout:

```
MyDrive/formcoach/
  raw/         your videos + public clips, named p03_squat_heel_lift_t1.mp4
  public/      downloaded public datasets
  review/      picture sheets + review.csv for labeling public clips
  processed/   reps.csv, crops.csv, crops/ (made by the pipeline)
  runs/        trained models, metrics, plots
  gemini/      Model B training data
```

The second cell clones the code into the Colab machine, installs MediaPipe and runs the unit tests (should end in
"passed").

### 3.4 Connect Colab to Google Cloud (Gemini notebook)

The first cell of `colab_gemini.ipynb` runs `auth.authenticate_user()`: a popup asks you to sign in with the Google
account that has access to the Cloud project. Then edit these two lines and run the cell:

```python
PROJECT = 'your-gcp-project'      # the project ID, e.g. formcoach-cs663 (not the display name)
BUCKET = 'gs://your-bucket'       # the bucket from section 4.5
```

Find the project ID on the console's home page under **Project info**.

---

## 4. Set up Google Cloud

One person does this and adds the others. Console links are given; the `gcloud` line next to each does the same.

### 4.1 Create the project

Console → project picker (top bar) → **New project** → name `formcoach-cs663` → **Create**.
`gcloud projects create formcoach-cs663`

### 4.2 Turn on billing and a budget

1. **Billing** (https://console.cloud.google.com/billing) → link a billing account to the project. New accounts
   are often offered trial credits; check what your account shows, and ask your instructor whether the course has
   education credits.
2. **Billing → Budgets & alerts → Create budget**: $50, alerts at 50%, 90% and 100%.

### 4.3 Enable the APIs

Console → **APIs & Services → Enable APIs and services**, then enable each of these. Or all at once:

```bash
gcloud services enable aiplatform.googleapis.com storage.googleapis.com run.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com --project formcoach-cs663
```

| API | Used for |
|---|---|
| Vertex AI API (`aiplatform`) | Model B tuning and the tuned endpoint, base Gemini, Model D |
| Cloud Storage | the training files Vertex AI reads |
| Cloud Run, Cloud Build, Artifact Registry | building and running the backend |

### 4.4 Give your teammates access

Console → **IAM & Admin → IAM → Grant access**, add each teammate's Google account with these roles:

| Role | Lets them |
|---|---|
| Vertex AI User | run tuning jobs and call Gemini |
| Storage Object Admin | upload training files to the bucket |
| Cloud Run Admin + Service Account User | deploy the backend (only whoever deploys) |

### 4.5 Create the bucket

Console → **Cloud Storage → Buckets → Create**: a globally unique name like `formcoach-cs663-data`, location
type **Region**, **us-central1** (same region as the tuning job), standard class, uniform access. Keep public access
prevention on.

```bash
gcloud storage buckets create gs://formcoach-cs663-data --location us-central1 --project formcoach-cs663
```

### 4.6 Check the Gemini models

Vertex AI changes which models can be tuned. Before step 9, check the supervised tuning page for the current list
and regions (https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning) and the
price (https://cloud.google.com/vertex-ai/generative-ai/pricing). The code defaults to `gemini-3.5-flash`
because Gemini 2.5 Flash was shut down on Vertex AI on Oct 20, 2026.

---

## 5. The datasets and where to get them

Public data goes into **training and validation only**. The test set is always your own recordings of people the
models never saw. The code enforces this (section 7.3).

| Dataset | URL | What's in it | License | Download | Used for | ID prefix |
|---|---|---|---|---|---|---|
| **Your own recordings** | – | all 13 classes of squat, lunge, push-up, front + side | your consent forms | you record them | train, val **and test** | `p01`, `p02`… |
| **Zenodo Squat Dataset** (Teng 2025) | https://zenodo.org/records/17558630 | side-view squat images in folders Good / Bad Back / Bad Heel | CC BY 4.0 | automatic in the notebook (`Dataset.zip`, 824 MB, md5 checked) | squat `correct`, `forward_lean`, `heel_lift` | `zsq` |
| **Mendeley Multi-View Raw Video Dataset of Seven Fitness Exercises** (Mashetty et al. 2026) | https://data.mendeley.com/datasets/kgbb3yn47p/2 | 26 subjects, front/side/diagonal videos of 7 exercises, labeled good or bad | CC BY 4.0 | **manual**: open the page → **Download All** → upload the zip to `MyDrive/formcoach/public/` | squats and push-ups; bad clips labeled by you | `mdl` |
| **REHAB24-6** (2024) | https://zenodo.org/records/13305826 | 6 exercises incl. lunges (exercise 5), push-ups (3), squats (6); 2 cameras; every rep marked correct/incorrect in `Segmentation.csv`; 10 adults | CC BY-NC 4.0 (non-commercial; cite the SISAP 2024 paper) | automatic in the notebook (`videos.zip` 2.7 GB + `Segmentation.csv`) | lunges and push-ups; incorrect reps labeled by you | `reh` |
| Fitness-AQA (optional) | https://github.com/ParitoshParmar/Fitness-AQA | back squat knees-inward / knees-forward / shallow, overhead press, barbell row | non-commercial, by request | follow the repo's access instructions | not wired into the pipeline; skip unless you have spare time | `faq` |
| MediaPipe Pose Landmarker model | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task | the BlazePose model used to extract landmarks offline | Google's model card terms | automatic (downloaded by `01_extract_reps.py`) | data preparation | – |

**Why some clips need labeling by you:** Mendeley and REHAB24-6 say a rep was good or bad, not which error. The
import makes a 6-frame picture sheet per bad clip and a `review.csv`; you write the error name for each. The error
names are in section 6.2.

**Recording your own data** (the full protocol is [[Data Collection Protocol]]):

- 8–10 people: the 3 of you plus friends and family, at least **2 who are not group members** (they become the test set).
- Per person, for each exercise: one clip per class, 5–8 reps per clip, one take front-45° and one from the side.
- Squats and lunges: phone at hip height, 2–3 m away, whole body in frame. Push-ups: phone on the floor about 2 m
  away, turned about 45° from the side (elbow flare only shows from the front).
- Act each error clearly but realistically; one person watches and confirms it's visible.
- File names: `<participant>_<exercise>_<error>_t<take>.mp4`, e.g. `p03_push_up_hip_sag_t1.mp4`. Upload to
  `MyDrive/formcoach/raw/`.

---

## 6. The models and their architectures

### 6.1 Model C: BlazePose (pretrained, not trained by you)

- **Architecture:** Google's BlazePose, a two-stage network: a fast person detector, then a landmark regressor that
  outputs **33 body landmarks** (x, y, visibility) per frame.
- **In the app:** ML Kit Pose Detection, `base` model in stream mode, on every camera frame.
- **In training:** MediaPipe Pose Landmarker `full` (float16), same 33-landmark layout, so the features computed in
  Python match the app's Dart code exactly (checked by a parity test).
- **What it feeds:** rep counting (main joint angle: knee for squat/lunge, elbow for push-up), joint-angle features,
  and the body box used to crop images for Model A.

### 6.2 Model A: MobileNetV3-Large form classifier (fine-tuned by you)

**Task:** given an image of the person at the bottom of a rep, say which form error is visible.

**Classes (13):**

| Exercise | Classes |
|---|---|
| Squat | `correct`, `knee_valgus`, `forward_lean`, `insufficient_depth`, `heel_lift` |
| Lunge | `correct`, `knee_alignment`, `forward_lean`, `insufficient_depth` |
| Push-up | `correct`, `hip_sag`, `elbow_flare`, `partial_rom` |

**Architecture** (`scripts/03_train_model_a.py`):

```
input  224 × 224 × 3 RGB, pixel values 0–255
  → MobileNetV3-Large backbone, ImageNet weights, built-in preprocessing (rescaling)   188 layers, 2,996,352 params
  → global average pooling                                                           960 features
  → Dropout 0.3
  → Dense 13, softmax                                                                12,493 params
output 13 class probabilities                                           total 3,008,845 params
```

MobileNetV3-Large is a mobile CNN built from inverted-residual blocks with depthwise convolutions, squeeze-and-excite
attention and hard-swish activations. It's small and fast enough to run on a phone in a few milliseconds.

**Training, in two stages:**

| | Stage 1: "probe" (the baseline) | Stage 2: "finetune" (the model you ship) |
|---|---|---|
| What trains | only the Dense head (12,493 params) | head + the top 60 backbone layers (2,480,605 params) |
| Optimizer | Adam, learning rate 1e-3 | Adam, learning rate 1e-4 |
| Epochs | up to 15 | up to 20 more |
| Stops | early stopping on validation loss, patience 5, best weights kept | same |
| BatchNorm | frozen | still frozen (small batches would spoil its statistics) |

Shared settings: batch 32, sparse categorical cross-entropy, **class weights** so rare classes count as much as
common ones, seed 663.

**Data augmentation** (training only): random zoom 85–100% of the crop, horizontal flip, brightness ±30,
contrast 0.75–1.25, saturation 0.7–1.3.

**How one example is made:** the pipeline finds each rep's deepest frame, takes 3 frames around it, crops a square
around the body with a 15% margin (black padding where it leaves the frame), and saves it at 256 px. Training uses
random 224 px crops; evaluation and the app resize the whole square to 224.

**At prediction time (app and evaluation):** the 3 crops of a rep are classified, probabilities are averaged, and
only the selected exercise's classes are kept and renormalized (a squat can't be labeled `hip_sag`).

**Export:** TensorFlow Lite with float16 weights (`--fp16`), about 6 MB; the export checks that TFLite and Keras
give the same answers.

**Baselines you compare against:**

1. the **probe** (frozen ImageNet backbone): shows what fine-tuning adds;
2. **logistic regression on joint angles** from Model C (`04b_feature_baseline.py`): shows what the images add
   beyond pose geometry.

### 6.3 Model B: fine-tuned Gemini 3.5 Flash coach

**Task:** turn the evidence about one finished rep into one coaching decision, as JSON.

**Base model:** `gemini-3.5-flash` on Vertex AI, multimodal (text + image in, text out).

**Fine-tuning method:** Vertex AI **supervised fine-tuning**, which trains small **adapter** weights on top of the
frozen Gemini model (parameter-efficient tuning; Gemini's own weights don't change). Settings
(`scripts/08_tune_gemini.py`): 3 epochs, adapter size 4, learning-rate multiplier 1.0.

**One training example** (JSONL, one per line):

```json
{"systemInstruction": {"role": "system", "parts": [{"text": "You are FormCoach, a strength coach giving real-time feedback inside a phone app. ..."}]},
 "contents": [
   {"role": "user", "parts": [
     {"fileData": {"mimeType": "image/jpeg", "fileUri": "gs://formcoach-cs663-data/formcoach/crops/p03/....jpg"}},
     {"text": "EVIDENCE\n{\"exercise\": \"squat\", \"features\": {\"knee_angle_left\": 88.0, \"knee_to_ankle_width\": 0.71, \"trunk_lean_deg\": 28.4}, \"history\": [\"correct\", \"knee_valgus\"], \"model_a\": {\"top_error\": \"knee_valgus\", \"top_prob\": 0.81, ...}, \"pose_quality\": {\"required_visible\": true, \"visibility\": 0.93}, \"rep_index\": 6, \"view\": \"oblique\"}"}]},
   {"role": "model", "parts": [{"text": "{\"action\": \"coach\", \"error\": \"knee_valgus\", \"cue\": \"Push your knees out over your toes.\", \"explanation\": \"Your knees were at 71% of ankle width at the bottom.\", \"severity\": \"medium\", \"visual_prompt\": \"...\"}"}]}]}
```

- **Input:** the bottom-of-rep crop + Model A's probabilities, joint angles, visibility, camera view and the last 3 reps.
- **Output:** `action` (`coach` / `praise` / `reposition` / `hold`), `error`, `cue` (spoken, ≤ 12 words),
  `explanation`, `severity`, `visual_prompt` (for Model D).
- **Case types** so the model learns judgment, not just templates: `natural` (real reps), `uncertain` (low
  confidence → should `hold`), `conflict` (Model A's label disagrees with the angles → shouldn't repeat it),
  `low_visibility` (→ `reposition`).
- **Baseline:** base `gemini-3.5-flash` with the same system instruction, JSON mode on.

### 6.4 Model D: Gemini 3.1 Flash Image (pretrained)

Given Model B's `visual_prompt` (text only, no user photo), it draws the corrected position for the Visual Trainer
screen. The backend caches images and falls back to stored ones if generation fails. About $0.067 per image, so it
runs only when the user taps **See correct form**.

---

## 7. Prepare the data

All of this is in `colab_model_a.ipynb` (GPU runtime, Drive connected).

### 7.1 Public data (sections 2a–2c of the notebook)

| Notebook section | What it does | Time |
|---|---|---|
| **2a** Zenodo | downloads, checks md5, unzips, maps folders → `squat:correct`, `squat:forward_lean`, `squat:heel_lift` | 5 min |
| **2b** Mendeley squats | finds squat clips; good ones go to `raw/`, bad ones to `review/mendeley/` with picture sheets | 10 min |
| **2b** Mendeley push-ups | same for push-ups → `review/mendeley_push_up/` | 5 min |
| **2c** REHAB24-6 | downloads videos + `Segmentation.csv`, cuts each video into runs of correct and incorrect reps; correct → `raw/`, incorrect → `review/rehab/lunge/` and `review/rehab/push_up/` | 20 min |

**Label the review sheets** (split between the three of you):

1. Open `review.csv` in Google Sheets (right-click in Drive → Open with → Google Sheets).
2. For each row, open the `sheet` picture and type the error name in the `error` column (names from section 6.2),
   or `skip` if you can't tell.
3. **File → Download → CSV**, and replace the `review.csv` in the same Drive folder.
4. Run the matching `apply-review` cell. It prints how many clips it copied into `raw/`.

### 7.2 Extract reps and crops (section 3)

```
python scripts/01_extract_reps.py --videos $D/raw --out $D/processed        # videos (yours + Mendeley + REHAB)
python scripts/01b_extract_images.py --manifest ... --prefix zsq --out ...   # Zenodo images
```

Always run `01` then `01b`, then the split. `01` rewrites `reps.csv`, so re-running it starts clean.

**Check the QA plots** (next cell): each video's plot should show one dip per rep with its bottom marked. If a clip
labeled with an error has reps that look correct, list them in `processed/rep_overrides.csv` (`rep_id,new_error`,
or `drop`). Report how many you relabeled.

### 7.3 Split by participant (section 4)

```
python scripts/02_make_splits.py --data $D/processed --pin p09=test --pin p10=test
```

Use your two non-member IDs in the pins. The rules the script enforces:

- each person is entirely in one split, never spread across two;
- your own people: 70% train, 15% validation, 15% test;
- public groups (`zsq`, `mdl`, `reh`): 85% train, 15% validation, **never test**; the script stops if any public
  rep lands in test.

**Check:** the printed table, and no "WARNING: split ... has n/13 classes". If a class is missing from a split,
record more of it or pin someone else.

**Before your recordings exist:** run the **Pilot split** cell instead (no pins) and use the pilot evaluation in 8.3.

---

## 8. Train Model A

### 8.1 Train (section 5 of the notebook)

```
python scripts/03_train_model_a.py --data $D/processed --mode probe    --out $D/runs/probe
python scripts/03_train_model_a.py --data $D/processed --mode finetune --out $D/runs/finetune
```

Probe takes about 10 minutes on a T4, finetune 20–40. Each run prints loss and accuracy per epoch and saves
`model.keras`, `history.json` and `run_config.json` in its folder.

**Check:** validation accuracy rises, training stops early on its own, and finetune ends with lower validation
loss than probe.

### 8.2 Evaluate on the unseen test people (section 6)

```
python scripts/04_eval_model_a.py --data $D/processed --runs $D/runs/probe $D/runs/finetune --export-preds $D/runs/finetune
python scripts/04b_feature_baseline.py --data $D/processed
```

Outputs: `runs/results_model_a.md` (accuracy, macro precision/recall/F1, error-vs-correct F1 for each run),
`confusion.png`, `curves.png`, `success.jpg` and `failure.jpg` galleries, per-exercise and per-participant
accuracy, and `preds_all.csv` (Model A's prediction for every rep, which Model B needs).

**What good looks like:** finetune clearly above probe and above the joint-angle baseline in macro-F1. Mixing up
similar errors is normal; many `correct` reps called errors means you need more data or cleaner labels.

### 8.3 Pilot evaluation (only before your recordings exist)

The **Pilot evaluation** cell runs the same scripts with `--eval-split val`. These are validation numbers on
public data: useful for checking the pipeline, never your final results.

### 8.4 Export to TensorFlow Lite (section 7)

```
python scripts/05_export_tflite.py --run $D/runs/finetune --data $D/processed --fp16
python scripts/10_data_report.py --data $D/processed --out $D/DATA_REPORT.md
```

You get `runs/finetune/form_classifier.tflite` and `DATA_REPORT.md` (counts per split, class and source, plus the
separation check for your report).

---

## 9. Fine-tune Model B

Open `notebooks/colab_gemini.ipynb` (no GPU), connect Drive, sign in to Google Cloud and set `PROJECT` / `BUCKET`
(section 3.4).

| Step | Notebook section | What happens | Time |
|---|---|---|---|
| 1 | Build situations | `06_build_gemini_situations.py` turns every rep + Model A's prediction into an example with a draft reply, adds the `uncertain`, `conflict` and `low_visibility` cases, and keeps each person's split | 2 min |
| 2 | **Human review** | open `gemini/situations.csv` in Google Sheets; copy `draft_json` into `final_json` and rewrite it as natural coaching ([[Gemini Labeling Guide]]); review every test row and every non-natural case; save as `situations_reviewed.csv` | 2–3 hours, split three ways |
| 3 | Export JSONL | uploads the crops to `gs://…/formcoach/crops`, writes `train.jsonl`, `val.jsonl`, `test_eval.jsonl` and uploads the first two | 5 min |
| 4 | Fine-tune | starts the Vertex AI job on `gemini-3.5-flash` (3 epochs, adapter size 4), prints its state every minute, and saves the result in `runs/gemini/tuning_job.json` | 30–90 min |
| 5 | Evaluate | base vs tuned on `test_eval.jsonl` | 15 min |

**Watch the job in the console:** in https://console.cloud.google.com open **Vertex AI → Tuning**; it
shows the job's progress and the training/validation loss curves; export them for the report.

**When it finishes,** copy the **endpoint** from the output (`projects/…/locations/us-central1/endpoints/…`).
The backend needs it.

**Evaluation outputs** (`runs/gemini/eval/`): `metrics.json` with valid JSON rate, schema compliance, action
accuracy, error accuracy, cue length, whether quoted numbers match the evidence, and latency, overall and per case
type; `base_vs_tuned.png`; and `rating_sheet.csv`, a blind A/B sheet each of you scores 1–5 for correctness and
clarity without looking at `rating_key.csv`.

---

## 10. Put the models into the app

### 10.1 Model A: one file

1. In Drive, download `formcoach/runs/finetune/form_classifier.tflite`.
2. Copy it into the app:
   ```bash
   cp ~/Downloads/form_classifier.tflite ~/Desktop/FORM/formcoach-app/mobile/assets/models/
   ```
3. The app's `mobile/assets/config/labels.json` must list the same 13 classes in the same order as
   `configs/labels.json` here (it does unless you changed the classes; if you did, copy it over).
4. In Android Studio: run `flutter test`, then press **Run**.

**What changes:** the Home screen's "Form model not installed yet" card disappears. After each rep the live screen
shows the alert card with the detected error and turns the problem joints red. Without a backend, the cue text
comes from the app's built-in rules.

### 10.2 Model B and D: deploy the backend

From Terminal on your Mac:

```bash
cd ~/Desktop/FORM/formcoach-app/backend
PROJECT=formcoach-cs663; REGION=us-central1
gcloud iam service-accounts create formcoach-api --project $PROJECT
gcloud projects add-iam-policy-binding $PROJECT \
  --member serviceAccount:formcoach-api@$PROJECT.iam.gserviceaccount.com --role roles/aiplatform.user
gcloud run deploy formcoach-api --source . --region $REGION --project $PROJECT \
  --service-account formcoach-api@$PROJECT.iam.gserviceaccount.com --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$REGION,DEFAULT_COACH=tuned,TUNED_MODEL=<endpoint from step 9>,API_KEY=<a long random string>
```

It prints a URL like `https://formcoach-api-xxxx.run.app`. Open `<URL>/healthz` in a browser to check it's up.
Keep the `API_KEY` secret: it's what stops strangers from using your Vertex AI quota.

### 10.3 Connect the app to the backend

In Android Studio: **Run → Edit Configurations → main.dart → Additional run args**:

```
--dart-define=BACKEND_URL=https://formcoach-api-xxxx.run.app --dart-define=API_KEY=<the same key>
```

**What changes:** cues now come from your fine-tuned Gemini. **Settings → Coach model** switches between
**Tuned**, **Base** and **Rules**, so you can demo the difference live. **See correct form** opens the Visual
Trainer with a Model D illustration.

### 10.4 End-to-end check

- [ ] Home screen has no "Form model not installed" card
- [ ] A squat set counts reps correctly
- [ ] A deliberate error shows the alert card and speaks the cue
- [ ] **See correct form** shows an illustration
- [ ] Ending the set shows the Session Summary with a form score
- [ ] The session appears on **Progress**
- [ ] Settings → Coach model → Base gives different cues than Tuned

---

## 11. Numbers for the report

| Course requirement | Where it comes from |
|---|---|
| Data sources, collection, counts, format, labels, preprocessing, split | this guide + `DATA_REPORT.md` + [[Model A Data Card]] / [[Model B Data Card]] |
| Test examples never used for training | the separation check in `DATA_REPORT.md` |
| Model A vs baselines | `runs/results_model_a.md`, `confusion.png`, `curves.png`, galleries |
| Model B base vs tuned | `runs/gemini/eval/metrics.json`, `base_vs_tuned.png`, rating sheet, Vertex loss curves |
| Model C reliability | rep-count accuracy and reposition rate from session logs (`scripts/analyze_sessions.py` in the app repo) |
| Model D quality | your ratings of 20–30 generated images |
| Limitations and biases | the data cards |

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `!nvidia-smi` fails | Runtime → Change runtime type → T4 GPU. If none is available, try later or train on CPU (much slower). |
| Colab disconnected | Reconnect, re-run the two setup cells, continue where you were; data is on Drive. |
| `01b` says reps.csv "has columns … expected …" | You split before adding images. Re-run from `01_extract_reps.py`. |
| "split … has n/13 classes" | Record more of the missing class, or pin a different person to test. |
| Mendeley import finds 0 clips | Run the `inspect` cell and change `--keyword` / `--good` / `--bad` to the words in the file names. |
| REHAB import: "no video found for … ids" | Run `inspect` on the videos folder; file names must contain the ID (`PM_021`) and the camera (`Camera17`). |
| Tuning job: permission denied on `gs://` | Give the Vertex AI service agent (`service-<project number>@gcp-sa-aiplatform.iam.gserviceaccount.com`) the Storage Object Viewer role on the bucket. |
| Tuning job: model not found / not tunable | Check the supervised tuning page (section 4.6) and pass `--base-model` with a listed model. |
| `gcloud run deploy` asks to enable APIs | Answer yes, or run the command in 4.3. |
| App: "model has N outputs but labels.json has M classes" | The `.tflite` and `labels.json` come from different runs: copy both from the same one. |
| App cues never change | `BACKEND_URL` isn't set or the phone can't reach it: open `<URL>/healthz` in the phone's browser. |
| Costs climbing | Check **Billing → Reports**; the tuned endpoint costs per request, so turn off demo phones when done. |
