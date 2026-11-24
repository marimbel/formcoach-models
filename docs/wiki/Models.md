# Models

Use these four models. A, B and C meet the 3-model minimum. D is the optional extra.

| Model | Course slot | What it does | Input → output | Runs on | Customization | Compared against |
| --- | --- | --- | --- | --- | --- | --- |
| **A: MobileNetV3-Large form classifier** | Student-trained / fine-tuned vision model | Names the form error for each rep | 3 body crops at the bottom of the rep (224×224 RGB) → 13 class probabilities, masked to the chosen exercise and averaged | Phone (TFLite) | ImageNet weights, fine-tuned in 2 stages on our crops (head first, then the top layers) | Frozen-backbone linear probe; logistic regression on joint angles |
| **B: Fine-tuned Gemini 3.5 Flash coach** | Fine-tuned Gemini multimodal model | Turns the evidence into one short coaching cue, as JSON | Model A probabilities, joint angles, visibility, camera view, last 3 reps, plus the crop → `{action, error, cue, explanation, visual_prompt}` | Cloud Run backend → Vertex AI endpoint | Supervised fine-tuning on Vertex AI with human-reviewed JSONL, 3 epochs | Base gemini-3.5-flash with the same prompt and JSON mode |
| **C: ML Kit Pose (BlazePose)** | Pretrained vision / foundation model | Finds 33 body landmarks in every frame; drives rep counting, angles and crops | Camera frame → 33 (x, y, visibility) landmarks | Phone (ML Kit); MediaPipe Pose Landmarker offline for data prep | None (used as-is) | Rule-based cues on its angles alone, versus adding A and B |
| **D: Gemini 3.1 Flash Image** (optional) | Extra generative model | Draws the corrected position on the Visual Trainer screen | Crop + `visual_prompt` from B → an annotated image | Backend → Gemini API | None (prompted) | Static reference figures that ship with the app |

Gemini 2.5 Flash, named in the proposal, was retired on Vertex AI on Oct 20, 2026. Gemini 3.5 Flash replaces it for both tuning and the baseline. Training costs $0.01 per 1,000 tokens: about 1,000 examples × 600 tokens × 3 epochs ≈ $18. Model D costs about $0.067 per image, so the app calls it only when someone taps "Show me".

Why A is needed when C already gives angles: angle thresholds miss errors that depend on camera view and body shape (heel lift, back rounding). A learns them from pixels. Its skeleton-feature baseline measures how much the images add.
