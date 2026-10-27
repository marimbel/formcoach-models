"""Scoring for base-vs-tuned Gemini comparison. Pure functions so they are unit-testable."""
from __future__ import annotations

import json
import re

import numpy as np

from formcoach.prompting import MAX_CUE_WORDS

_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%?)")


def evidence_numbers(situation: dict) -> list[float]:
    vals: list[float] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            vals.append(float(x))
            if 0 <= x <= 1:
                vals.append(float(x) * 100)  # probabilities / ratios quoted as percentages
    walk(situation)
    return vals


def number_grounding(explanation: str, situation: dict) -> tuple[int, int]:
    """(grounded, total) numbers in the explanation. A number is grounded if it is within 1.5 units
    (or 3%) of any evidence value. Heuristic: catches invented measurements, not wrong reasoning."""
    ev = evidence_numbers(situation)
    total = grounded = 0
    for m in _NUM.finditer(explanation or ""):
        v = float(m.group(1))
        total += 1
        if any(abs(v - e) <= max(1.5, 0.03 * abs(e)) for e in ev):
            grounded += 1
    return grounded, total


def score(ref: dict, obj: dict | None, problems: list[str], situation: dict) -> dict:
    s = {"json_valid": obj is not None, "schema_ok": obj is not None and not problems}
    if obj is None:
        return {**s, "action_match": False, "error_match": False, "cue_len_ok": False,
                "num_grounded": 0, "num_total": 0}
    s["action_match"] = obj.get("action") == ref.get("action")
    s["error_match"] = (obj.get("error") or None) == (ref.get("error") or None)
    cue = obj.get("cue") or ""
    s["cue_len_ok"] = isinstance(cue, str) and len(cue.split()) <= MAX_CUE_WORDS
    g, t = number_grounding(obj.get("explanation", ""), situation)
    s["num_grounded"], s["num_total"] = g, t
    return s


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    out = {k: float(np.mean([r[k] for r in rows])) for k in
           ("json_valid", "schema_ok", "action_match", "error_match", "cue_len_ok")}
    tot = sum(r["num_total"] for r in rows)
    out["number_grounding_rate"] = (sum(r["num_grounded"] for r in rows) / tot) if tot else None
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if lat:
        out["latency_p50_ms"] = float(np.percentile(lat, 50))
        out["latency_p95_ms"] = float(np.percentile(lat, 95))
    out["n"] = len(rows)
    return out


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
