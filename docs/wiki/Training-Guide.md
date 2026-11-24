# Training guide: from zero to a working app

This page walks through training the two models FormCoach needs and putting them into the Android app, in order.
Each step says what to run, how long it takes, and what you should see before moving on.

| Model | Trained? | Where it ends up |
|---|---|---|
| **A** MobileNetV3 form classifier | Yes, by you, on Colab (GPU) | `form_classifier.tflite` inside the app |
| **B** Gemini 3.5 Flash coach | Yes, fine-tuned on Vertex AI | a Vertex AI endpoint the backend calls |
| **C** ML Kit Pose | No, pretrained | already in the app |
| **D** Gemini 3.1 Flash Image | No, pretrained | called by the backend |

Order matters: Model B is trained on Model A's predictions, so A comes first.

---

## Part 1. One-time setup (about 30 minutes)

1. **Google Drive.** Create the folder `MyDrive/formcoach/`. The notebooks keep everything there (`raw/`,
   `public/`, `processed/`, `runs/`, `gemini/`), so you can stop and come back.
2. **Colab.** Open `notebooks/colab_model_a.ipynb` from GitHub in Colab (File → Open notebook → GitHub →
   `marimbel/formcoach-models`). Runtime → Change runtime type → **T4 GPU**.
3. **Google Cloud** (needed for Model B, the backend and Model D; one person sets it up and shares it):
   1. Create a project with billing on, and set a **budget alert** (e.g. $50).
   2. Enable the APIs: `gcloud services enable aiplatform.googleapis.com run.googleapis.com cloudbuild.googleapis.com storage.googleapis.com`
   3. Create a bucket in `us-central1`: `gcloud storage buckets create gs://formcoach-<yourname> --location us-central1`
4. **Locally (Mac):** clone both repos, install Flutter + Android Studio (see the app repo's `docs/platform_setup.md`).

---

## Part 2. Training data

### 2a. Public data (today, no recording needed)

In `colab_model_a.ipynb`, run in order:

| Cell | What it does | Time |
|---|---|---|
| Setup (Drive mount, clone) | Mounts Drive, gets the code, installs packages | 2 min |
| **2a** Zenodo | Downloads 824 MB of squat images and checks the md5 | 5 min |
| **2b** Mendeley | Needs a manual **Download All** from the dataset page, then upload the zip to `MyDrive/formcoach/public/`. Imports squats, then push-ups | 15 min |
| **2c** REHAB24-6 | Downloads lunge and push-up videos (about 3.5 GB) and cuts them into clips | 20 min |

**Label the incorrect public clips.** Mendeley and REHAB24-6 say a rep was done badly but not which error. Each
import makes a picture sheet per clip (`review/.../*.jpg`) and a `review.csv`:

1. Open `review.csv` in Google Sheets.
2. Look at each picture and type the error in the `error` column:
   - squat: `knee_valgus`, `forward_lean`, `insufficient_depth`, `heel_lift`
   - lunge: `knee_alignment`, `forward_lean`, `insufficient_depth`
   - push-up: `hip_sag`, `elbow_flare`, `partial_rom`
   - or `skip` if you can't tell
3. Download as CSV back to the same place, then run the `apply-review` cell.

Split the sheets between the three of you; about 1–2 hours each.

### 2b. Your own recordings (required: the only test data)

Follow [[Data Collection Protocol]]. The short version:

- 8–10 people (the 3 of you plus friends), at least 2 who are **not** group members (they become the test set).
- Per person: every class of all 3 exercises, one clip each, 5–8 reps per clip, front-45° and side views.
- Phone on a tripod at hip height, 2–3 m away (push-ups: phone on the floor, about 45° from the side).
- Name every file `p03_squat_heel_lift_t1.mp4` (participant, exercise, error, take) and upload to `MyDrive/formcoach/raw/`.

Budget about 40 minutes per person.

> **Before you have recordings,** you can run Parts 3.1–3.4 as a **pilot** on public data only: use the "Pilot
> split" and "Pilot evaluation" cells. It tests the pipeline and gives early numbers. Re-run from 3.1 once your
> recordings are in.

---

## Part 3. Train Model A (Colab, about 1–2 hours)

Still in `colab_model_a.ipynb`.

### 3.1 Extract reps and crops (section 3)

Runs pose detection on every video, finds each rep, and saves 3 body crops at the bottom of each rep.

**Check before moving on:** open the QA plots cell. Each plot should show one dip per rep with the bottom marked
at the deepest point. If a clip shows the wrong number of reps, note it; if a rep labelled with an error looks
correct, add it to `processed/rep_overrides.csv` (`rep_id,new_error`).

### 3.2 Split by participant (section 4)

Edit the pins to your two non-member participants, e.g. `--pin p09=test --pin p10=test`, and run.

**Check:** the printed table shows public data (`mdl`, `zsq`, `reh`) only in train and val, and the script ends
without "BUG". Every split should have every class; if not, it prints a warning: record more or re-pin.

### 3.3 Train (section 5)

Two runs:

1. **probe**: the ImageNet backbone frozen, only the head trained (the baseline). About 10 minutes.
2. **finetune**: then the top layers unfrozen. About 20–40 minutes.

**Check:** the validation accuracy rises and the run stops early on its own. Finetune should beat probe.

### 3.4 Evaluate (section 6)

Scores both runs on the **unseen test participants** and runs the joint-angle baseline. You get
`results_model_a.md`, a confusion matrix, learning curves, and galleries of correct and wrong predictions.

**What good looks like:** macro-F1 clearly above the probe and above the joint-angle baseline. Look at the
failure gallery: mistakes between visually similar errors (e.g. forward lean vs depth) are normal; mistakes on
`correct` reps mean more data is needed.

### 3.5 Export (section 7)

Writes `runs/finetune/form_classifier.tflite` and checks it gives the same answers as the Keras model. Also
writes `DATA_REPORT.md` with the counts for your report.

---

## Part 4. Train Model B (Colab + Vertex AI, about half a day including review)

Open `notebooks/colab_gemini.ipynb` (no GPU needed). Edit `PROJECT` and `BUCKET` in the first cell.

| Step | What to do | Time |
|---|---|---|
| **1. Build situations** | Turns every real rep plus Model A's prediction into a "rep just finished" example with a draft coaching reply | 2 min |
| **2. Human review** | Open `gemini/situations.csv` in Google Sheets. For each row copy `draft_json` into `final_json` and **edit it** into natural coaching ([[Gemini Labeling Guide]]). Review every test row and every `conflict`, `uncertain` and `low_visibility` row. Save as `situations_reviewed.csv` | 2–3 hours, split three ways |
| **3. Export JSONL** | Uploads the crops and writes `train.jsonl`, `val.jsonl`, `test_eval.jsonl` | 5 min |
| **4. Fine-tune** | Starts the Vertex AI tuning job on `gemini-3.5-flash` and waits. Cost about $18 for 1,000 examples × 3 epochs | 30–90 min |
| **5. Evaluate** | Base vs tuned on the test situations: valid JSON, right action, right error, cue length, latency. Writes a blind A/B rating sheet | 15 min |

**Check:** step 4 prints the tuned **endpoint** (`projects/.../endpoints/...`) and saves it in
`runs/gemini/tuning_job.json`. Copy it; the backend needs it.

Each of you fills in `rating_sheet.csv` without looking at `rating_key.csv`, then score it.

---

## Part 5. Put the models into the app

### 5.1 Model A: copy one file into the app

```bash
cp ~/Downloads/form_classifier.tflite ~/Desktop/FORM/formcoach-app/mobile/assets/models/
```

(Download it from `MyDrive/formcoach/runs/finetune/` first.) `labels.json` in the app is already the same as
`configs/labels.json`; if you changed classes, copy that too into `mobile/assets/config/`.

Then in Android Studio: run `flutter test`, press **Run**.

**What changes in the app:** the Home screen's "Form model not installed yet" card disappears, and during a set the
live screen now shows an alert card after each rep with the detected error, plus red joints on the skeleton.
Without a backend the cue comes from the offline rules.

### 5.2 Model B: deploy the backend with the tuned endpoint

From your Mac (after `gcloud auth login`), in `formcoach-app/backend`:

```bash
PROJECT=your-project; REGION=us-central1
gcloud iam service-accounts create formcoach-api --project $PROJECT
gcloud projects add-iam-policy-binding $PROJECT \
  --member serviceAccount:formcoach-api@$PROJECT.iam.gserviceaccount.com --role roles/aiplatform.user
gcloud run deploy formcoach-api --source . --region $REGION --project $PROJECT \
  --service-account formcoach-api@$PROJECT.iam.gserviceaccount.com --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$REGION,DEFAULT_COACH=tuned,TUNED_MODEL=<endpoint from step 4>,API_KEY=<a long random string>
```

It prints a URL like `https://formcoach-api-xxxx.run.app`. Check it: open `<URL>/healthz` in a browser.

Connect the app to it: in Android Studio, **Run → Edit Configurations → main.dart → Additional run args**:

```
--dart-define=BACKEND_URL=https://formcoach-api-xxxx.run.app --dart-define=API_KEY=<the same key>
```

**What changes in the app:** cues now come from the fine-tuned Gemini. In **Settings → Coach model** you can
switch between **Tuned**, **Base** and **Rules** to demo the difference live. The Session Summary's "Performance
details" shows the coach's response time and how many replies fell back to the offline rules.

### 5.3 Model D: nothing to train

The backend calls Gemini 3.1 Flash Image when someone taps **See correct form**. The Visual Trainer screen shows the
illustration; if none can be made, it says so and the key points still show.

### 5.4 End-to-end check

- [ ] Home screen has no "Form model not installed" card
- [ ] A squat set counts reps correctly
- [ ] Doing a deliberate error shows the alert card and speaks the cue
- [ ] **See correct form** opens the Visual Trainer with an illustration
- [ ] Ending the set shows the Session Summary with a form score
- [ ] The session appears on **Progress**
- [ ] Settings → Coach model → Base gives different (usually worse) cues than Tuned

---

## Part 6. Numbers for the report

| What | Where it comes from |
|---|---|
| Data counts per split, class and source | `DATA_REPORT.md` (`10_data_report.py`) |
| Model A vs probe vs joint-angle baseline | `runs/results_model_a.md`, `runs/finetune/confusion.png` |
| Model B base vs tuned | `runs/gemini/eval/metrics.json`, `base_vs_tuned.png`, the rating sheet |
| Model C reliability | rep-count accuracy and reposition rate from session logs (`scripts/analyze_sessions.py` in the app repo) |
| Model D | your ratings of 20–30 generated images |

## Troubleshooting

| Problem | Fix |
|---|---|
| `01b` says reps.csv "has columns … expected …" | You ran `02` before `01b`. Re-run from `01_extract_reps.py`. |
| A split is missing a class | Record more of that class, or pin a different participant to test. |
| Colab disconnects during training | Re-run from the setup cells; everything is on Drive. |
| Tuning job fails on the model name | Check Vertex AI's list of tunable models and pass `--base-model`. |
| App says "model has N outputs but labels.json has M classes" | The `.tflite` and `labels.json` don't match: copy both from the same run. |
| Cues never change in the app | `BACKEND_URL` isn't set, or the phone can't reach it: open `<URL>/healthz` on the phone's browser. |
