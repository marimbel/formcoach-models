import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from formcoach import geometry as G
from formcoach.coaching_kb import draft_response
from formcoach.gemini_eval import number_grounding, score
from formcoach.labels import class_names, errors_for, exercise_class_indices, exercises, parse_video_name
from formcoach.prompting import parse_response, render_target, render_user_prompt
from formcoach.reps import OneEuroFilter, interpolate_gaps, segment_reps, smooth_landmarks
from formcoach.splits import participant_split
from synth import squat_frame, squat_sequence


# ---------- labels ----------
def test_class_list_and_indices():
    names = class_names()
    assert len(names) == sum(len(errors_for(e)) for e in exercises()) == 13
    for ex in exercises():
        idx = exercise_class_indices(ex)
        assert all(names[i].startswith(ex + "__") for i in idx)


@pytest.mark.parametrize("name,exp", [
    ("p03_squat_knee_valgus_t2.mp4", ("p03", "squat", "knee_valgus", "t2")),
    ("P01_chest_press_uneven_arms_t1.MOV", ("p01", "chest_press", "uneven_arms", "t1")),
    ("p10_lunge_correct_t12.mp4", ("p10", "lunge", "correct", "t12")),
])
def test_parse_video_name(name, exp):
    m = parse_video_name(name)
    assert (m.participant, m.exercise, m.error, m.take) == exp


@pytest.mark.parametrize("bad", ["squat_knee_valgus.mp4", "p1_squat_bad_t1.mp4", "p1_deadlift_correct_t1.mp4"])
def test_parse_video_name_rejects(bad):
    with pytest.raises(ValueError):
        parse_video_name(bad)


# ---------- geometry ----------
def test_angle_basic():
    assert G.angle(np.array([1.0, 0]), np.array([0.0, 0]), np.array([0.0, 1])) == pytest.approx(90)
    assert G.angle(np.array([1.0, 0]), np.array([0.0, 0]), np.array([-1.0, 0])) == pytest.approx(180)
    assert math.isnan(G.angle(np.zeros(2), np.zeros(2), np.ones(2)))


@pytest.mark.parametrize("k", [80, 120, 170])
def test_synthetic_knee_angle(k):
    lm = squat_frame(k)
    kl, kr = G.knee_angles(lm)
    assert kl == pytest.approx(k, abs=0.5) and kr == pytest.approx(k, abs=0.5)
    assert G.primary_angle(lm, "knee", "mean") == pytest.approx(k, abs=0.5)


def test_primary_angle_ignores_invisible_side():
    lm = squat_frame(90)
    lm[25, 2] = 0.1  # left knee hidden
    lm[26, :2] = lm[26, :2]  # right unchanged
    assert G.primary_angle(lm, "knee", "mean") == pytest.approx(G.knee_angles(lm)[1])
    lm[26, 2] = 0.1
    assert math.isnan(G.primary_angle(lm, "knee", "mean"))


def test_valgus_feature_moves():
    normal = G.frame_features(squat_frame(90), "squat")["knee_to_ankle_width"]
    valgus = G.frame_features(squat_frame(90, valgus=30), "squat")["knee_to_ankle_width"]
    assert valgus < normal


def test_features_json_serialisable():
    for ex in exercises():
        f = G.frame_features(squat_frame(95), ex)
        json.dumps(f)
        assert "view" in f and "trunk_lean_deg" in f


def test_person_bbox_square_and_padded_crop():
    lm = squat_frame(90)
    x0, y0, x1, y1 = G.person_bbox(lm, 720, 1280)
    assert (x1 - x0) == (y1 - y0)
    img = np.full((1280, 720, 3), 200, np.uint8)
    crop = G.crop_square(img, (x0, y0, x1, y1))
    assert crop.shape[0] == crop.shape[1] == x1 - x0
    if x0 < 0:  # box wider than the frame -> padded, not stretched
        assert crop[:, 0].max() == 0
    crop2 = G.crop_square(img, (-50, -50, 50, 50))
    assert crop2[:50, :50].max() == 0 and crop2[60, 60, 0] == 200


# ---------- temporal ----------
def test_segment_reps_counts_and_bottoms():
    frames, true_angles, fps = squat_sequence(n_reps=5, noise=2.0)
    sm = smooth_landmarks(frames, fps)
    sig = np.array([G.primary_angle(f, "knee", "mean") for f in sm])
    reps = segment_reps(sig, fps, top=160, bottom_enter=145, min_rep_seconds=0.6)
    assert len(reps) == 5
    for r in reps:
        assert true_angles[r.bottom] < 90  # bottom found near the true minimum (80)
        assert r.start < r.bottom < r.end


def test_segment_reps_ignores_jitter_at_top():
    sig = 170 + np.random.default_rng(1).normal(0, 3, 300)
    assert segment_reps(sig, 30, 160, 145) == []


def test_partial_rep_not_counted_without_return_to_top():
    sig = np.concatenate([np.full(30, 170), np.linspace(170, 100, 30), np.full(30, 120)])
    assert segment_reps(sig, 30, 160, 145) == []


def test_one_euro_reduces_noise():
    rng = np.random.default_rng(0)
    x = 100 + rng.normal(0, 5, 300)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    y = np.array([f(v, t / 30) for t, v in enumerate(x)])
    assert y[30:].std() < 0.5 * x[30:].std()


def test_interpolate_gaps():
    s = np.array([0.0, np.nan, np.nan, 3.0, np.nan] + [np.nan] * 10 + [5.0])
    out = interpolate_gaps(s, max_gap=3)
    assert out[1] == pytest.approx(1) and out[2] == pytest.approx(2)
    assert np.isnan(out[6])  # long gap left alone


# ---------- splits ----------
def test_participant_split_disjoint_and_pinned():
    counts = {f"p{i:02d}": 30 + i for i in range(10)}
    sp = participant_split(counts, pinned={"p09": "test"})
    assert set(sp) == set(counts)
    assert sp["p09"] == "test"
    assert {"train", "val", "test"} <= set(sp.values())
    share = {s: sum(counts[p] for p in counts if sp[p] == s) / sum(counts.values()) for s in ("train", "val", "test")}
    assert share["train"] > 0.5


# ---------- prompting / draft coach ----------
def _situation(ex, err, p, vis=0.95, req=True, hist=(), view="oblique", feats=None):
    probs = {e: (p if e == err else (1 - p) / (len(errors_for(ex)) - 1)) for e in errors_for(ex)}
    return {"exercise": ex, "rep_index": 3, "phase": "bottom", "view": view,
            "pose_quality": {"visibility": vis, "required_visible": req},
            "model_a": {"top_error": err, "top_prob": p, "probs": probs},
            "pose_features": feats if feats is not None else {}, "history": list(hist)}


def test_drafts_always_valid():
    rng = random.Random(0)
    for _ in range(2000):
        ex = rng.choice(exercises())
        err = rng.choice(errors_for(ex))
        hist = [rng.choice(errors_for(ex) + ["uncertain"]) for _ in range(rng.randint(0, 3))]
        feats = G.frame_features(squat_frame(rng.uniform(70, 150), valgus=rng.uniform(0, 40)), ex)
        feats.pop("view")
        s = _situation(ex, err, rng.uniform(0.2, 0.99), vis=rng.uniform(0.3, 1), req=rng.random() > 0.1,
                       hist=hist, view=rng.choice(["front", "side", "oblique"]), feats=feats)
        d = draft_response(s, rng)
        obj, problems = parse_response(render_target(d))
        assert obj is not None and problems == [], (d, problems)


def test_draft_decisions():
    assert draft_response(_situation("squat", "knee_valgus", 0.9, vis=0.4, req=False))["action"] == "reposition"
    assert draft_response(_situation("squat", "knee_valgus", 0.35))["action"] == "hold"
    r = draft_response(_situation("squat", "correct", 0.9, hist=["knee_valgus"]))
    assert r["action"] == "praise" and "previous" in r["explanation"]
    rep = draft_response(_situation("squat", "forward_lean", 0.8, hist=["forward_lean", "forward_lean"]))
    assert rep["severity"] == "high"
    side = draft_response(_situation("squat", "knee_valgus", 0.9, view="side"))
    assert side["severity"] == "low"
    # Model A says valgus but knees are wide -> contradiction
    contra = draft_response(_situation("squat", "knee_valgus", 0.6, feats={"knee_to_ankle_width": 1.2}))
    assert contra["action"] == "hold"


def test_parse_response_variants():
    good = '{"action":"coach","error":"knee_valgus","cue":"Knees out.","explanation":"x","severity":"low","visual_prompt":null}'
    assert parse_response(good)[1] == []
    assert parse_response("```json\n" + good + "\n```")[1] == []
    assert "extra_text" in parse_response("Sure! " + good)[1]
    assert parse_response("nope")[0] is None
    long = good.replace("Knees out.", " ".join(["word"] * 13))
    assert "cue_too_long" in parse_response(long)[1]


def test_user_prompt_deterministic():
    s = _situation("lunge", "correct", 0.8)
    assert render_user_prompt(s) == render_user_prompt(json.loads(json.dumps(s)))


# ---------- gemini eval ----------
def test_number_grounding():
    s = {"pose_features": {"trunk_lean_deg": 47.3}, "model_a": {"top_prob": 0.84}}
    assert number_grounding("Torso leaned 47 degrees; 84% confident.", s) == (2, 2)
    assert number_grounding("Your knee hit 33 degrees.", s) == (0, 1)


def test_score_handles_bad_json():
    out = score({"action": "coach"}, None, ["not_json"], {})
    assert out["json_valid"] is False and out["action_match"] is False
