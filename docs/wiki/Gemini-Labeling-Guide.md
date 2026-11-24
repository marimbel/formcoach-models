# Writing Gemini fine-tuning targets

`06_build_gemini_situations.py` produces `situations.csv` with a **draft** reply per situation. Import it into a
Google Sheet, split rows among the group, and fill `final_json` (copy the draft and edit it). Then export as
`situations_reviewed.csv`.

**Why this matters:** if you train on unedited drafts, the tuned model just learns our templates, and the
base-vs-tuned comparison shows only that it learned the JSON format. Your edits are what teach coaching judgment.

## Priorities
1. **All test-split rows must be reviewed.** They are the references for evaluation. `07` skips unreviewed test rows.
2. Review **every** `conflict`, `uncertain` and `low_visibility` row; the drafts are weakest there.
3. Then as many `natural` train rows as you can. Rewrite at least ~40% in your own coaching words.

## Rules for a good reply
- Look at the crop image (`image_path`) and the evidence together. If the image shows the classifier is wrong,
  write what a careful coach would say given what the **app can know** (e.g. hold, or a low-severity cue).
- `cue`: what the phone says aloud, ≤ 12 words, imperative, one fix only. Vary wording; don't always use the same phrase.
- `explanation`: 1–2 sentences that cite real numbers from the evidence. Never invent a measurement.
- `severity`: low = minor or uncertain; medium = clear error; high = repeated over 3 reps or clearly risky.
- `visual_prompt`: one sentence describing an illustration of the *correct* form for this fix (used by Model D).
  `null` when action is praise/hold/reposition.
- `action`:
  - `reposition` when visibility is low: don't comment on form at all.
  - `hold` when the evidence is weak or contradictory: `cue` must be `""`.
  - `praise` for correct reps, especially right after a corrected error.
  - Never mention pain, injury or diagnosis.

Must stay valid JSON with exactly these keys: `action, error, cue, explanation, severity, visual_prompt`.
`07` rejects invalid rows and prints which ones.
