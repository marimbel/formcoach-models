"""Label set + filename conventions shared by every script.

Video filename convention (one exercise, one form condition per clip):

    <participant>_<exercise>_<error>_<take>.mp4
    e.g.  p03_squat_knee_valgus_t2.mp4,  p01_chest_press_correct_t1.mov

Participants act out the error on purpose, so the label is known per video
and every rep in the clip inherits it. Reviewers should still spot-check reps
(see docs/data_collection_protocol.md) because acted errors are not always
performed consistently.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "labels.json"
SEP = "__"


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def exercises() -> list[str]:
    return list(load_config()["exercises"].keys())


def errors_for(exercise: str) -> list[str]:
    return list(load_config()["exercises"][exercise]["errors"])


def class_names() -> list[str]:
    """Global, ordered class list used as the Model A output layer."""
    return [f"{ex}{SEP}{err}" for ex in exercises() for err in errors_for(ex)]


def class_index(exercise: str, error: str) -> int:
    return class_names().index(f"{exercise}{SEP}{error}")


def exercise_class_indices(exercise: str) -> list[int]:
    names = class_names()
    return [i for i, n in enumerate(names) if n.startswith(exercise + SEP)]


def split_class(name: str) -> tuple[str, str]:
    ex, err = name.split(SEP, 1)
    return ex, err


@dataclass(frozen=True)
class VideoMeta:
    participant: str
    exercise: str
    error: str
    take: str

    @property
    def class_name(self) -> str:
        return f"{self.exercise}{SEP}{self.error}"


_FNAME = re.compile(r"^(?P<p>[A-Za-z0-9]+)_(?P<rest>.+)_(?P<take>t\d+)$")


def parse_video_name(path: str | Path) -> VideoMeta:
    """Parse '<participant>_<exercise>_<error>_<take>'. Exercise and error may contain underscores,
    so we match them against the configured vocab instead of splitting blindly."""
    stem = Path(path).stem
    m = _FNAME.match(stem)
    if not m:
        raise ValueError(f"Bad video name {stem!r}; expected <participant>_<exercise>_<error>_t<N>")
    rest = m.group("rest")
    for ex in sorted(exercises(), key=len, reverse=True):
        if rest.startswith(ex + "_"):
            err = rest[len(ex) + 1:]
            if err in errors_for(ex):
                return VideoMeta(m.group("p").lower(), ex, err, m.group("take"))
            raise ValueError(f"{stem!r}: error {err!r} not in {errors_for(ex)}")
    raise ValueError(f"{stem!r}: unknown exercise; expected one of {exercises()}")
