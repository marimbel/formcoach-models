"""Synthetic stick-figure data for tests and pipeline smoke runs (NOT for real training)."""
from __future__ import annotations

import math

import numpy as np


def squat_frame(knee_angle: float, x0: float = 300.0, valgus: float = 0.0, vis: float = 0.95) -> np.ndarray:
    """33 landmarks (x, y, vis) in pixels for an oblique-view squat at the given knee angle."""
    lm = np.zeros((33, 3), dtype=np.float32)
    lm[:, 2] = vis
    phi = math.radians((180 - knee_angle) / 2)
    for side, dx in ((0, -45.0), (1, 45.0)):
        ankle = np.array([x0 + dx, 900.0])
        knee = ankle + 200 * np.array([math.sin(phi), -math.cos(phi)])
        knee[0] -= np.sign(dx) * valgus  # pull knees toward the midline
        hip = knee + 200 * np.array([-math.sin(phi), -math.cos(phi)])
        shoulder = hip + np.array([60 * math.sin(phi), -280.0])
        shoulder[0] = x0 + dx * 1.6
        elbow = shoulder + np.array([dx * 0.6, 120.0])
        wrist = elbow + np.array([0.0, 110.0])
        heel = ankle + np.array([-25.0, 15.0])
        foot = ankle + np.array([45.0, 18.0])
        idx = (11, 13, 15, 23, 25, 27, 29, 31) if side == 0 else (12, 14, 16, 24, 26, 28, 30, 32)
        for i, p in zip(idx, (shoulder, elbow, wrist, hip, knee, ankle, heel, foot)):
            lm[i, :2] = p
    lm[0, :2] = (lm[11, :2] + lm[12, :2]) / 2 + np.array([0, -70])
    return lm


def squat_sequence(n_reps=5, fps=30.0, period_s=2.0, pause_s=0.6, bottom=80.0, noise=1.5, seed=0, **kw):
    rng = np.random.default_rng(seed)
    angles = []
    for _ in range(n_reps):
        angles += [175.0] * int(pause_s * fps)
        n = int(period_s * fps)
        angles += list(175 - (175 - bottom) * np.sin(np.linspace(0, math.pi, n)))
    angles += [175.0] * int(pause_s * fps)
    frames = np.stack([squat_frame(a, **kw) for a in angles])
    frames[:, :, :2] += rng.normal(0, noise, frames[:, :, :2].shape)
    return frames, np.array(angles), fps
