"""Joint-angle and form features from 33 BlazePose landmarks.

The mobile app computes the SAME features in Dart (mobile/lib/vision/geometry.dart), so
names, formulas and units must stay in sync. Landmarks are (x, y, visibility) in PIXELS
of the upright image (y grows downward). Using pixels, not normalized coordinates, keeps
angles correct when the image is not square.
"""
from __future__ import annotations

import math

import numpy as np

# BlazePose / ML Kit indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

VIS_THRESH = 0.5


def _p(lm: np.ndarray, i: int) -> np.ndarray:
    return lm[i, :2].astype(float)


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ABC in degrees (0..180)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def visible(lm: np.ndarray, idx: list[int], thresh: float = VIS_THRESH) -> bool:
    return bool(np.all(lm[idx, 2] >= thresh))


def knee_angles(lm):
    return (angle(_p(lm, L_HIP), _p(lm, L_KNEE), _p(lm, L_ANKLE)),
            angle(_p(lm, R_HIP), _p(lm, R_KNEE), _p(lm, R_ANKLE)))


def elbow_angles(lm):
    return (angle(_p(lm, L_SHOULDER), _p(lm, L_ELBOW), _p(lm, L_WRIST)),
            angle(_p(lm, R_SHOULDER), _p(lm, R_ELBOW), _p(lm, R_WRIST)))


def primary_angle(lm: np.ndarray, which: str, combine: str) -> float:
    """Angle that drives rep counting. Sides with low visibility are ignored."""
    if which == "knee":
        vals = knee_angles(lm)
        vis = [min(lm[L_HIP, 2], lm[L_KNEE, 2], lm[L_ANKLE, 2]),
               min(lm[R_HIP, 2], lm[R_KNEE, 2], lm[R_ANKLE, 2])]
    elif which == "elbow":
        vals = elbow_angles(lm)
        vis = [min(lm[L_SHOULDER, 2], lm[L_ELBOW, 2], lm[L_WRIST, 2]),
               min(lm[R_SHOULDER, 2], lm[R_ELBOW, 2], lm[R_WRIST, 2])]
    else:
        raise ValueError(which)
    ok = [v for v, s in zip(vals, vis) if s >= VIS_THRESH and not math.isnan(v)]
    if not ok:
        return float("nan")
    return float(min(ok)) if combine == "min" else float(np.mean(ok))


def estimate_view(lm: np.ndarray) -> str:
    """'front' | 'side' | 'oblique' from shoulder width relative to torso length."""
    sw = _dist(_p(lm, L_SHOULDER), _p(lm, R_SHOULDER))
    mid_s = (_p(lm, L_SHOULDER) + _p(lm, R_SHOULDER)) / 2
    mid_h = (_p(lm, L_HIP) + _p(lm, R_HIP)) / 2
    torso = _dist(mid_s, mid_h)
    if torso < 1e-6:
        return "unknown"
    r = sw / torso
    if r > 0.55:
        return "front"
    if r < 0.25:
        return "side"
    return "oblique"


def frame_features(lm: np.ndarray, exercise: str) -> dict:
    """Features at a single (bottom-of-rep) frame. NaN when not measurable."""
    f: dict[str, float | str] = {}
    f["view"] = estimate_view(lm)
    mid_s = (_p(lm, L_SHOULDER) + _p(lm, R_SHOULDER)) / 2
    mid_h = (_p(lm, L_HIP) + _p(lm, R_HIP)) / 2
    up = mid_s - mid_h
    # trunk lean: 0 = upright, 90 = horizontal
    f["trunk_lean_deg"] = angle(mid_h + np.array([0.0, -1.0]), mid_h, mid_h + up) if np.linalg.norm(up) > 0 else float("nan")

    if exercise in ("squat", "lunge"):
        kl, kr = knee_angles(lm)
        f["knee_angle_left"], f["knee_angle_right"] = kl, kr
        f["hip_angle_left"] = angle(_p(lm, L_SHOULDER), _p(lm, L_HIP), _p(lm, L_KNEE))
        f["hip_angle_right"] = angle(_p(lm, R_SHOULDER), _p(lm, R_HIP), _p(lm, R_KNEE))
        ankle_w = abs(lm[L_ANKLE, 0] - lm[R_ANKLE, 0])
        knee_w = abs(lm[L_KNEE, 0] - lm[R_KNEE, 0])
        f["knee_to_ankle_width"] = float(knee_w / ankle_w) if ankle_w > 1e-6 else float("nan")
        thigh = np.mean([_dist(_p(lm, L_HIP), _p(lm, L_KNEE)), _dist(_p(lm, R_HIP), _p(lm, R_KNEE))])
        hip_y = (lm[L_HIP, 1] + lm[R_HIP, 1]) / 2
        knee_y = (lm[L_KNEE, 1] + lm[R_KNEE, 1]) / 2
        f["hip_below_knee"] = float((hip_y - knee_y) / thigh) if thigh > 1e-6 else float("nan")
        lifts = []
        for heel, foot, knee, ankle in ((L_HEEL, L_FOOT, L_KNEE, L_ANKLE), (R_HEEL, R_FOOT, R_KNEE, R_ANKLE)):
            shank = _dist(_p(lm, knee), _p(lm, ankle))
            if shank > 1e-6 and lm[heel, 2] >= VIS_THRESH and lm[foot, 2] >= VIS_THRESH:
                lifts.append((lm[foot, 1] - lm[heel, 1]) / shank)  # >0: heel above toes
        f["heel_lift"] = float(max(lifts)) if lifts else float("nan")
        # front leg = more bent knee
        front_left = (kl if not math.isnan(kl) else 999) <= (kr if not math.isnan(kr) else 999)
        knee, ankle, heel, foot = ((L_KNEE, L_ANKLE, L_HEEL, L_FOOT) if front_left
                                   else (R_KNEE, R_ANKLE, R_HEEL, R_FOOT))
        shank = _dist(_p(lm, knee), _p(lm, ankle))
        facing = np.sign(lm[foot, 0] - lm[heel, 0]) or 1.0
        f["knee_past_toe"] = float((lm[knee, 0] - lm[foot, 0]) * facing / shank) if shank > 1e-6 else float("nan")

    if exercise == "chest_press":
        el, er = elbow_angles(lm)
        f["elbow_angle_left"], f["elbow_angle_right"] = el, er
        f["elbow_asymmetry_deg"] = abs(el - er)
        f["shoulder_abduction_left"] = angle(_p(lm, L_HIP), _p(lm, L_SHOULDER), _p(lm, L_ELBOW))
        f["shoulder_abduction_right"] = angle(_p(lm, R_HIP), _p(lm, R_SHOULDER), _p(lm, R_ELBOW))
        sw = _dist(_p(lm, L_SHOULDER), _p(lm, R_SHOULDER))
        f["wrist_height_diff"] = float(abs(lm[L_WRIST, 1] - lm[R_WRIST, 1]) / sw) if sw > 1e-6 else float("nan")
    # NaN -> None so the JSON is valid and identical to what the app sends (null)
    return {k: ((None if math.isnan(v) else round(float(v), 3)) if isinstance(v, (float, np.floating)) else v)
            for k, v in f.items()}


def person_bbox(lm: np.ndarray, width: int, height: int, margin: float = 0.15) -> tuple[int, int, int, int]:
    """Square box (x0, y0, x1, y1) around visible landmarks, expanded by `margin` on each side.

    The box may extend past the image edge (e.g. a tall person in a portrait frame); crop_square()
    pads with black instead of stretching, and the app does the same, so aspect ratio is preserved.
    """
    pts = lm[lm[:, 2] >= VIS_THRESH, :2]
    if len(pts) < 4:
        pts = lm[:, :2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(round(max(x1 - x0, y1 - y0) * (1 + 2 * margin)))
    side = max(side, 16)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    return x0, y0, x0 + side, y0 + side


def crop_square(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop `box` from img (H, W, C), zero-padding wherever the box leaves the image."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], dtype=img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out
