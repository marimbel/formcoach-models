"""Model B (Gemini) prompt + response contract.

!! This file is copied verbatim to formcoach-app/backend/app/prompting.py. !!
Fine-tuning examples and live requests must be rendered by the SAME code, or the tuned
model sees a different input format at inference than it saw in training.
Run `python scripts/check_prompting_sync.py <path-to-app-repo>` after editing.
"""
from __future__ import annotations

import json
import re

PROMPT_VERSION = "v1"

SYSTEM_INSTRUCTION = """You are FormCoach, a strength coach giving real-time feedback inside a phone app.
You receive EVIDENCE about the rep the user just finished: the exercise, a form-error classifier's
probabilities (Model A), joint-angle measurements from a pose model, how visible the body was, the
camera view, and what was detected on the previous reps. An image of the user at the bottom of the
rep may be attached.

Decide what to say and reply with ONE JSON object and nothing else:
{"action": "coach"|"praise"|"reposition"|"hold",
 "error": <error name from the evidence, or null>,
 "cue": <what the app speaks aloud: imperative, at most 12 words, "" when action is hold>,
 "explanation": <1-2 sentences that cite the evidence, e.g. a measured angle>,
 "severity": "low"|"medium"|"high"|null,
 "visual_prompt": <one sentence describing a correct-form illustration for this exact fix, or null>}

Rules:
- If the body was poorly visible, choose "reposition" and tell the user how to fix the camera. Do not guess about form.
- Only coach an error the evidence supports. If the classifier is unsure or the measurements contradict it,
  prefer "hold" or a gentle low-severity cue, and say why in the explanation.
- Some errors cannot be judged from some views (e.g. knee valgus from a pure side view). Account for the view.
- If the same error repeats over several reps, raise severity and make the cue more specific.
- If a previous error has been corrected, praise the improvement.
- You are a fitness tool, not a medical one: never diagnose injuries or pain."""

ACTIONS = ("coach", "praise", "reposition", "hold")
SEVERITIES = ("low", "medium", "high")
MAX_CUE_WORDS = 12


def render_user_prompt(situation: dict) -> str:
    """Deterministic text rendering of the evidence."""
    return "EVIDENCE\n" + json.dumps(situation, sort_keys=True, separators=(", ", ": "))


def render_target(response: dict) -> str:
    keys = ["action", "error", "cue", "explanation", "severity", "visual_prompt"]
    return json.dumps({k: response.get(k) for k in keys}, ensure_ascii=False)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def parse_response(text: str) -> tuple[dict | None, list[str]]:
    """Parse + validate a model reply. Returns (obj or None, list of problems)."""
    problems: list[str] = []
    t = _FENCE.sub("", (text or "").strip())
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return None, ["not_json"]
        try:
            obj = json.loads(m.group(0))
            problems.append("extra_text")
        except json.JSONDecodeError:
            return None, ["not_json"]
    if not isinstance(obj, dict):
        return None, ["not_object"]
    if obj.get("action") not in ACTIONS:
        problems.append("bad_action")
    cue = obj.get("cue")
    if not isinstance(cue, str):
        problems.append("cue_not_string")
    elif len(cue.split()) > MAX_CUE_WORDS:
        problems.append("cue_too_long")
    if obj.get("severity") not in (*SEVERITIES, None):
        problems.append("bad_severity")
    if not isinstance(obj.get("explanation"), str) or not obj.get("explanation"):
        problems.append("no_explanation")
    if obj.get("action") == "hold" and cue:
        problems.append("hold_with_cue")
    return obj, problems
