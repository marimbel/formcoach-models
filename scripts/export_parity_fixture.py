#!/usr/bin/env python3
"""Export reference outputs of the Python geometry/temporal code for the app's Dart parity test.

    python scripts/export_parity_fixture.py ../formcoach-app/mobile/test/fixtures/parity.json

Run it whenever formcoach/geometry.py or formcoach/reps.py changes, then `flutter test` in the app.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from formcoach import geometry as G  # noqa: E402
from formcoach.labels import load_config  # noqa: E402
from formcoach.reps import OneEuroFilter, segment_reps  # noqa: E402
from synth import squat_frame, squat_sequence  # noqa: E402


def clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def main(out: Path):
    rng = np.random.default_rng(7)
    poses = []
    for k in range(12):
        lm = squat_frame(float(rng.uniform(70, 175)), x0=float(rng.uniform(200, 400)), valgus=float(rng.uniform(0, 40)))
        lm[:, :2] += rng.normal(0, 4, lm[:, :2].shape)
        lm[:, 2] = rng.uniform(0.3, 1.0, 33)
        lm[[11, 12, 23, 24, 25, 26, 27, 28], 2] = np.maximum(lm[[11, 12, 23, 24, 25, 26, 27, 28], 2], 0.55 if k % 4 else 0.0)
        lm = lm.round(3)  # the fixture stores rounded values, so compute expectations from them too
        ex = ["squat", "lunge", "chest_press"][k % 3]
        poses.append({
            "landmarks": lm.tolist(), "exercise": ex,
            "features": {k2: clean(v) for k2, v in G.frame_features(lm, ex).items()},
            "knee_mean": clean(G.primary_angle(lm, "knee", "mean")),
            "knee_min": clean(G.primary_angle(lm, "knee", "min")),
            "elbow_mean": clean(G.primary_angle(lm, "elbow", "mean")),
            "box": list(G.person_bbox(lm, 720, 1280)),
        })
    frames, _, fps = squat_sequence(n_reps=4, noise=2.5, seed=3)
    cfg = load_config()["exercises"]["squat"]["rep"]
    sig = [round(float(G.primary_angle(f, "knee", "mean")), 4) for f in frames]
    reps = segment_reps(np.array(sig), fps, cfg["top"], cfg["bottom_enter"], cfg["min_rep_seconds"])
    f = OneEuroFilter()
    xs = (100 + rng.normal(0, 5, 60)).round(3).tolist()
    euro = [float(f(x, i / 30)) for i, x in enumerate(xs)]
    fixture = {"poses": poses,
               "rep_signal": {"fps": fps, "angles": sig, "config": cfg,
                              "reps": [{"start": r.start, "bottom": r.bottom, "end": r.end, "min": r.min_angle} for r in reps]},
               "one_euro": {"t_step": 1 / 30, "x": xs, "y": euro}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture))
    print(f"wrote {out}: {len(poses)} poses, {len(reps)} reps")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
