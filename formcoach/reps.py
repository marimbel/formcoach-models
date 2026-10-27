"""Temporal processing: keypoint smoothing and rep segmentation.

Same state machine as mobile/lib/vision/rep_counter.dart:
  UP  --(angle < bottom_enter)-->  DOWN (track minimum angle = bottom frame)
  DOWN --(angle > top, and rep lasted >= min_rep_seconds)--> UP, emit rep
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012) for a scalar or vector signal."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.01, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x = None
        self._dx = None
        self._t = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x
        dt = max(t - self._t, 1e-6)
        dx = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


def smooth_landmarks(lms: np.ndarray, fps: float, **kw) -> np.ndarray:
    """lms: (T, 33, 3). Smooths x,y; leaves visibility untouched. NaN frames are held."""
    out = lms.copy()
    f = OneEuroFilter(**kw)
    last = None
    for t in range(len(lms)):
        xy = lms[t, :, :2]
        if np.isnan(xy).any():
            if last is not None:
                out[t, :, :2] = last
            continue
        last = f(xy, t / fps)
        out[t, :, :2] = last
    return out


def interpolate_gaps(sig: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Linear-interpolate NaN runs no longer than max_gap frames."""
    s = sig.astype(float).copy()
    n = len(s)
    i = 0
    while i < n:
        if math.isnan(s[i]):
            j = i
            while j < n and math.isnan(s[j]):
                j += 1
            if 0 < i and j < n and (j - i) <= max_gap:
                s[i:j] = np.linspace(s[i - 1], s[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return s


@dataclass
class Rep:
    start: int
    bottom: int
    end: int
    min_angle: float


def segment_reps(angle_sig: np.ndarray, fps: float, top: float, bottom_enter: float,
                 min_rep_seconds: float = 0.6) -> list[Rep]:
    reps: list[Rep] = []
    state = "init"  # must see a 'top' position first
    start = bottom = 0
    min_a = math.inf
    for t, a in enumerate(angle_sig):
        if math.isnan(a):
            continue
        if state in ("init", "up_wait") and a > top:
            state, start = "up", t
        elif state == "up":
            if a > top:
                start = t  # latest frame still at top
            elif a < bottom_enter:
                state, min_a, bottom = "down", a, t
        elif state == "down":
            if a < min_a:
                min_a, bottom = a, t
            if a > top:
                if (t - start) / fps >= min_rep_seconds:
                    reps.append(Rep(start, bottom, t, float(min_a)))
                state, start = "up", t
    return reps
