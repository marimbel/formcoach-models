"""Participant-level splits so no person appears in more than one of train/val/test."""
from __future__ import annotations

import random
from collections import Counter


def participant_split(counts: dict[str, int], ratios=(0.70, 0.15, 0.15), seed: int = 663,
                      pinned: dict[str, str] | None = None) -> dict[str, str]:
    """Greedy assignment of participants to splits, aiming at the target share of *samples*.

    counts: participant -> number of samples. pinned: participant -> forced split
    (e.g. keep a participant who is not in the group as an unseen test user).
    """
    names = ["train", "val", "test"]
    pinned = dict(pinned or {})
    total = sum(counts.values())
    target = {n: r * total for n, r in zip(names, ratios)}
    have: Counter = Counter()
    out: dict[str, str] = {}
    for p, s in pinned.items():
        if p in counts:
            out[p] = s
            have[s] += counts[p]
    rest = [p for p in counts if p not in out]
    rng = random.Random(seed)
    rng.shuffle(rest)
    rest.sort(key=lambda p: -counts[p])  # place big participants first; shuffle breaks ties
    for p in rest:
        # every split must get at least one participant if possible
        empty = [n for n in names if have[n] == 0 and n not in pinned.values()]
        if empty and len(rest) >= 3:
            choice = empty[-1]  # fill test, then val, first
        else:
            choice = max(names, key=lambda n: (target[n] - have[n]) / max(target[n], 1e-9))
        out[p] = choice
        have[choice] += counts[p]
    return out
