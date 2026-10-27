#!/usr/bin/env python3
"""Write a tiny synthetic dataset (stick-figure videos + matching landmark files) to smoke-test the
whole pipeline without real recordings or the MediaPipe model:

    python tests/make_synthetic_dataset.py /tmp/fc_smoke
    python scripts/01_extract_reps.py --videos /tmp/fc_smoke/raw --out /tmp/fc_smoke/processed --reuse-landmarks
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import squat_sequence  # noqa: E402

EDGES = [(11, 12), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (11, 13), (13, 15),
         (12, 14), (14, 16), (27, 29), (29, 31), (28, 30), (30, 32)]


def main(root: Path):
    raw, lmdir = root / "raw", root / "processed" / "landmarks"
    raw.mkdir(parents=True, exist_ok=True)
    lmdir.mkdir(parents=True, exist_ok=True)
    W, H = 540, 960
    combos = [("squat", "correct", dict(bottom=80)), ("squat", "insufficient_depth", dict(bottom=125)),
              ("squat", "knee_valgus", dict(bottom=85, valgus=35)),
              ("lunge", "correct", dict(bottom=85)), ("lunge", "insufficient_depth", dict(bottom=128))]
    for pi in range(7):
        for ex, err, kw in combos:
            frames, _, fps = squat_sequence(n_reps=3, seed=pi, noise=1.0, x0=270 + 10 * pi, **kw)
            frames[:, :, 1] = frames[:, :, 1] * 0.85 + 40  # fit in frame
            name = f"p{pi:02d}_{ex}_{err}_t1"
            vw = cv2.VideoWriter(str(raw / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            for lm in frames:
                img = np.full((H, W, 3), (40 + 20 * pi, 60, 80), np.uint8)
                for a, b in EDGES:
                    cv2.line(img, tuple(map(int, lm[a, :2])), tuple(map(int, lm[b, :2])), (255, 255, 255), 6)
                vw.write(img)
            vw.release()
            np.savez_compressed(lmdir / f"{name}.npz", landmarks=frames, fps=fps, width=W, height=H)
    print(f"wrote synthetic dataset to {root}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
