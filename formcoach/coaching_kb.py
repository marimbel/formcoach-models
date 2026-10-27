"""Rule-based "draft coach" used to pre-fill Gemini fine-tuning targets.

These drafts are a STARTING POINT for human reviewers, not the final labels. If the group
trains on unedited drafts, the tuned Gemini only learns to imitate these templates, and the
base-vs-tuned comparison says little. Reviewers should rewrite cues in natural coaching
language and fix any draft that the evidence does not support (see docs/gemini_labeling_guide.md).
The same short cues are mirrored in the app's offline fallback (mobile/lib/coach/local_cues.dart).
"""
from __future__ import annotations

import math
import random

CUES = {
    ("squat", "knee_valgus"): ["Push your knees out over your toes.",
                               "Drive your knees outward as you stand.",
                               "Keep knees tracking over your middle toes."],
    ("squat", "forward_lean"): ["Keep your chest up as you sit back.",
                                "Brace and keep your torso taller.",
                                "Lift your chest; don't fold forward."],
    ("squat", "insufficient_depth"): ["Sit lower, hips to knee height.",
                                      "Go a little deeper this rep.",
                                      "Lower until your thighs are near parallel."],
    ("squat", "heel_lift"): ["Keep your heels down through the rep.",
                             "Press through your whole foot.",
                             "Shift weight back toward your heels."],
    ("lunge", "knee_alignment"): ["Keep your front knee above your ankle.",
                                  "Take a slightly longer step.",
                                  "Front knee stays over your foot, not past it."],
    ("lunge", "forward_lean"): ["Stay tall through your torso.",
                                "Keep your chest up as you lower.",
                                "Hips straight down, chest proud."],
    ("lunge", "insufficient_depth"): ["Lower your back knee closer to the floor.",
                                      "Drop a bit deeper on the next rep.",
                                      "Bend both knees to about ninety degrees."],
    ("chest_press", "elbow_flare"): ["Tuck your elbows to about 45 degrees.",
                                     "Bring elbows in closer to your sides.",
                                     "Don't flare; elbows angled toward your hips."],
    ("chest_press", "uneven_arms"): ["Press both arms at the same speed.",
                                     "Keep the weights level as you press.",
                                     "Match your left and right arms."],
    ("chest_press", "partial_rom"): ["Lower the weights all the way down.",
                                     "Use your full range of motion.",
                                     "Bring elbows down until upper arms touch."],
}

PRAISE = ["Nice rep, keep that form.", "Good rep.", "Solid, keep it up."]
IMPROVED = ["Better! That fixed it.", "Much better, keep that.", "That's the fix, nice work."]
REPOSITION = {
    "squat": "Step back so your whole body is in frame.",
    "lunge": "Step back so your legs and feet are visible.",
    "chest_press": "Move the phone so both arms are visible.",
}

EVIDENCE_TEXT = {
    "knee_valgus": lambda f: _fmt("Knee width was {:.0%} of ankle width at the bottom", f.get("knee_to_ankle_width")),
    "forward_lean": lambda f: _fmt("Torso leaned {:.0f} degrees from vertical", f.get("trunk_lean_deg")),
    "insufficient_depth": lambda f: _fmt("Knee angle only reached {:.0f} degrees", _min(f, "knee_angle_left", "knee_angle_right")),
    "heel_lift": lambda f: _fmt("Heel rose {:.2f} shank-lengths above the toes", f.get("heel_lift")),
    "knee_alignment": lambda f: _fmt("Front knee travelled {:.2f} shank-lengths past the toes", f.get("knee_past_toe")),
    "elbow_flare": lambda f: _fmt("Elbows opened to about {:.0f} degrees from the torso", _max(f, "shoulder_abduction_left", "shoulder_abduction_right")),
    "uneven_arms": lambda f: _fmt("Left and right elbow angles differed by {:.0f} degrees", f.get("elbow_asymmetry_deg")),
    "partial_rom": lambda f: _fmt("Elbows only bent to {:.0f} degrees", _min(f, "elbow_angle_left", "elbow_angle_right")),
}

# Rough "does the geometry agree?" checks. STARTING thresholds only: calibrate them on your own
# reps (scripts/04b_feature_baseline.py prints per-class feature distributions you can use).
# Each returns True (supports), False (contradicts) or None (not measurable).
def _chk(v, fn):
    return None if not _ok(v) else bool(fn(v))


SUPPORTS = {
    "knee_valgus": lambda f: _chk(f.get("knee_to_ankle_width"), lambda v: v < 0.85),
    "forward_lean": lambda f: _chk(f.get("trunk_lean_deg"), lambda v: v > 40),
    "insufficient_depth": lambda f: _chk(_min(f, "knee_angle_left", "knee_angle_right"), lambda v: v > 100),
    "heel_lift": lambda f: _chk(f.get("heel_lift"), lambda v: v > 0.08),
    "knee_alignment": lambda f: _chk(f.get("knee_past_toe"), lambda v: v > 0.15),
    "elbow_flare": lambda f: _chk(_max(f, "shoulder_abduction_left", "shoulder_abduction_right"), lambda v: v > 70),
    "uneven_arms": lambda f: _chk(f.get("elbow_asymmetry_deg"), lambda v: v > 15),
    "partial_rom": lambda f: _chk(_min(f, "elbow_angle_left", "elbow_angle_right"), lambda v: v > 100),
}

VIEW_BLIND = {("knee_valgus", "side"), ("uneven_arms", "side"), ("forward_lean", "front"), ("knee_alignment", "front")}


def _ok(v):
    return isinstance(v, (int, float)) and not math.isnan(v)


def _fmt(t, v):
    return t.format(v) + "." if _ok(v) else ""


def _min(f, *k):
    vals = [f.get(x) for x in k if _ok(f.get(x))]
    return min(vals) if vals else None


def _max(f, *k):
    vals = [f.get(x) for x in k if _ok(f.get(x))]
    return max(vals) if vals else None


def visual_prompt(exercise: str, error: str) -> str:
    fix = CUES[(exercise, error)][0].rstrip(".").lower()
    view = "front view" if error in ("knee_valgus", "uneven_arms") else "side view"
    name = exercise.replace("_", " ")
    return (f"Clean instructional illustration, {view}, of an adult athlete in plain workout clothes "
            f"at the bottom of a {name} with correct form: {fix}. Neutral background, arrows marking the fix.")


def draft_response(situation: dict, rng: random.Random | None = None) -> dict:
    rng = rng or random.Random(0)
    ex = situation["exercise"]
    q = situation["pose_quality"]
    if not q.get("required_visible", True) or q.get("visibility", 1) < 0.6:
        return dict(action="reposition", error=None, cue=REPOSITION[ex],
                    explanation=f"Key joints were only {q.get('visibility', 0):.0%} visible, so form can't be judged reliably.",
                    severity=None, visual_prompt=None)
    ma = situation["model_a"]
    err, p = ma["top_error"], ma["top_prob"]
    hist = situation.get("history", [])
    view = situation.get("view", "unknown")
    if err == "correct" and p >= 0.5:
        prev_err = next((h for h in reversed(hist) if h not in ("correct", "uncertain")), None)
        if prev_err and hist and hist[-1] == prev_err:
            return dict(action="praise", error=None, cue=rng.choice(IMPROVED),
                        explanation=f"The previous rep showed {prev_err.replace('_', ' ')}; this one was classified correct ({p:.0%}).",
                        severity=None, visual_prompt=None)
        return dict(action="praise", error=None, cue=rng.choice(PRAISE),
                    explanation=f"Rep classified as correct form ({p:.0%}).", severity=None, visual_prompt=None)
    if p < 0.5:
        return dict(action="hold", error=None, cue="",
                    explanation=f"The classifier was unsure (top guess {err.replace('_', ' ')} at {p:.0%}), so no correction yet.",
                    severity=None, visual_prompt=None)
    ev = EVIDENCE_TEXT[err](situation.get("pose_features", {}))
    repeats = sum(1 for h in hist[-3:] if h == err)
    if (err, view) in VIEW_BLIND:
        return dict(action="coach", error=err, cue=rng.choice(CUES[(ex, err)]),
                    explanation=f"Possible {err.replace('_', ' ')} ({p:.0%}), but it is hard to confirm from a {view} view. {ev}".strip(),
                    severity="low", visual_prompt=visual_prompt(ex, err))
    if SUPPORTS[err](situation.get("pose_features", {})) is False:
        if p < 0.75:
            return dict(action="hold", error=None, cue="",
                        explanation=f"Model A suggested {err.replace('_', ' ')} ({p:.0%}) but the joint measurements "
                                    f"don't show it. {ev}".strip(),
                        severity=None, visual_prompt=None)
        return dict(action="coach", error=err, cue=rng.choice(CUES[(ex, err)]),
                    explanation=f"Model A is fairly confident of {err.replace('_', ' ')} ({p:.0%}), though the angles "
                                f"only weakly support it. {ev}".strip(),
                    severity="low", visual_prompt=visual_prompt(ex, err))
    sev = "high" if repeats >= 2 else ("medium" if p >= 0.7 or repeats == 1 else "low")
    cue = rng.choice(CUES[(ex, err)])
    if repeats >= 2:
        cue = "Again: " + cue[0].lower() + cue[1:]
        cue = " ".join(cue.split()[:12])
    return dict(action="coach", error=err, cue=cue,
                explanation=(f"{ev} Model A flagged {err.replace('_', ' ')} at {p:.0%}"
                             + (f", {repeats + 1} reps in a row." if repeats >= 2 else ".")).strip(),
                severity=sev, visual_prompt=visual_prompt(ex, err))
