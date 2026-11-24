# Data collection protocol

## Consent
Every participant signs a short consent note: videos are used to train and test a class project model, crops
may be uploaded to the group's private Google Cloud bucket and sent to Gemini, and they can withdraw. Never
publish identifiable footage in the report or video without explicit permission.

## Who
Aim for **8–10 participants**: the 3 group members plus friends and family. Public datasets add squat, lunge and
push-up examples for training (Zenodo, Mendeley, REHAB24-6), but the whole test set comes only from us.
Recruit at least 2 non-members and pin them to the test split (`02_make_splits.py --pin p09=test`), so the
"unseen user" test is real. Record variety: body types, heights, clothing (baggy vs. fitted), skin tones,
hair covering the shoulders, etc.

### How much to record
Per participant: 13 classes × 1 clip × 6 reps ≈ 80 reps, about 30–40 minutes with rests. Add a second camera
angle for the squat classes if there is time. 8 participants ≈ 600+ reps (1,800+ crops) of our own data.

## What to record (per participant)
For each exercise, one clip per class, **5–8 reps per clip**, plus a second take from a different angle:

| Exercise | Classes | Camera |
|---|---|---|
| Squat | correct, knee_valgus, forward_lean, insufficient_depth, heel_lift | take 1 front-45°, take 2 side |
| Lunge | correct, knee_alignment (front knee far past toes), forward_lean, insufficient_depth | take 1 side, take 2 front-45° |
| Push-up | correct, hip_sag (hips drop toward the floor), elbow_flare (elbows straight out to the sides), partial_rom (chest stays high) | phone on the floor; take 1 side, take 2 front-45° (elbow flare only shows from the front) |

Acted errors should be **realistic but clear**. Exaggerated cartoon errors make Model A look great on test and
fail in the real world. Have one person watch and confirm each error is visible before moving on.

## Conditions to vary (also used later for real-world testing)
Indoor/outdoor, bright/dim light, plain/cluttered background, 1.5–3.5 m distance, phone at hip height vs. on the
floor, partial occlusion (chair in front of a leg), another person walking behind.

## Naming
`<participant>_<exercise>_<error>_t<take>.mp4`, e.g. `p07_lunge_forward_lean_t2.mp4`. Lowercase, no spaces.
Participants are `p01`, `p02`, ... (keep a private mapping to real names out of the repo).

## After recording
1. Upload to the shared Drive folder `formcoach/raw/`.
2. Run `01_extract_reps.py` and **look at every QA plot**: the expected number of reps, bottoms at the deepest point.
3. Spot-check crops: if a clip labelled `knee_valgus` has reps that look correct, note them in
   `data/rep_overrides.csv` (rep_id,new_error) and fix them before training. Report how many you relabelled.

## Data card (fill in for the report)
For Model A and Model B separately: source, collection method, number of participants/videos/reps/crops, label
definitions, preprocessing (pose smoothing, rep segmentation, square crop with 15% margin, 256→224), split
counts per class, known limitations (acted errors, few participants, mostly indoor, one phone model) and
potential biases (body types and clothing represented, camera heights, who labelled).
