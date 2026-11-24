# Model B data card

One example is one "rep just finished" situation and the coaching reply a coach should give. Examples are built from the real reps in the Model A dataset, so Model B inherits its participants and its split. Aim for about 1,000 train, 150 val and 150 test examples.

| Item | Answer |
| --- | --- |
| Where it came from | Each rep in `reps.csv` plus Model A's prediction for it (`06_build_gemini_situations.py`) |
| Input (situation) | Model A top classes and probabilities, joint-angle features, visibility, camera view, the last 3 reps, plus the bottom-of-rep crop |
| Target (label) | JSON reply: `action` (coach / praise / reposition / hold), `error`, `cue` (≤ 12 words), `explanation`, `visual_prompt` |
| How targets are made | A rule-based draft from `coaching_kb.py`, then a group member edits it into the `final_json` column, following the [[Gemini Labeling Guide]]. Only reviewed rows are used |
| Case types | natural (real rep, real prediction); uncertain (flattened probabilities → should hold); conflict (A's label swapped to an error the angles don't support → should not repeat it); low_visibility (→ should ask to reposition) |
| Format | Vertex AI supervised-tuning JSONL (`systemInstruction` + `contents`, image as a `gs://` URI) for train and val; `test_eval.jsonl` with the reference reply for our own evaluation |
| Split | Follows the participant split: test situations come only from the unseen test participants |
| Preprocessing | The same prompt text as the live backend (`prompting.py`, checked byte-identical by `check_prompting_sync.py`) |
| Cost | About 1,000 × 600 tokens × 3 epochs ≈ 1.8M tokens ≈ $18 at $0.01 per 1,000 training tokens |
| Limitations | Targets are written by 3 students, not certified trainers. Synthetic case types are made by editing real situations |
| Potential biases | Cue style follows our own wording. It inherits every bias of Model A's data |

## Evaluation

`09_eval_gemini.py` compares base gemini-3.5-flash and the tuned endpoint on `test_eval.jsonl`: valid JSON, schema compliance, action accuracy, error accuracy, cue length, whether quoted numbers match the evidence, and latency, overall and per case type. It also writes a blind A/B `rating_sheet.csv` so each member can rate correctness and clarity 1–5 (keep `rating_key.csv` hidden until scoring).
