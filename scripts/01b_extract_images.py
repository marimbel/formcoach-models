#!/usr/bin/env python3
"""Labelled still images (e.g. bottom-of-squat photos) -> body crops + pose features, appended to the same
reps.csv / crops.csv that 01_extract_reps.py writes. Each image counts as one "rep" at its bottom frame.

    python scripts/01b_extract_images.py --manifest data/public/zenodo_squat_manifest.csv \
        --prefix zsq --out data/processed

Images from public datasets usually don't say who is in them, so images are grouped into --groups pseudo-
participants (by file order) just so train and validation can be split. These pseudo-participants are
marked train/val-only by 02_make_splits.py and never reach the test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach import geometry as G  # noqa: E402
from formcoach.labels import load_config  # noqa: E402

REP_FIELDS = ["rep_id", "video", "participant", "exercise", "error", "class_name", "rep_idx", "start", "bottom", "end",
              "fps", "min_angle", "visibility", "required_visible", "view", "features"]
CROP_FIELDS = ["image_path", "rep_id", "video", "participant", "exercise", "error", "class_name", "offset", "frame"]


class ImagePose:
    """MediaPipe Pose Landmarker in IMAGE mode (same BlazePose model as the video pipeline)."""

    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.PoseLandmarkerOptions(base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                                            running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                            min_pose_detection_confidence=0.5)
        self.lm = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not res.pose_landmarks:
            return None
        return np.array([[p.x * w, p.y * h, p.visibility] for p in res.pose_landmarks[0]], dtype=np.float32)


def append_rows(path: Path, fields: list[str], rows: list[dict]):
    new = not path.exists()
    if not new:
        header = next(csv.reader(open(path)), [])
        if header != fields:
            sys.exit(f"{path} has columns {header}; expected {fields}. Run 01b right after 01_extract_reps.py, "
                     "before 02_make_splits.py (which adds a split column).")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="from 00_import_public_data.py images")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=10, help="pseudo-participants for the train/val split")
    ap.add_argument("--pose-model", type=Path, default=Path("models/pose_landmarker_full.task"))
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--min-visibility", type=float, default=0.5)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from importlib import import_module
    ensure_model = import_module("01_extract_reps").ensure_model  # same model download helper
    pose = ImagePose(ensure_model(a.pose_model))
    cfg = load_config()["exercises"]
    rows = list(csv.DictReader(open(a.manifest)))
    crop_dir = a.out / "crops" / a.prefix
    crop_dir.mkdir(parents=True, exist_ok=True)
    # pseudo-participants: within each class, contiguous blocks of images in file order (neighbouring images are
    # usually frames of the same clip and must not end up in both train and val)
    by_cls: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(rows):
        by_cls.setdefault((r["exercise"], r["error"]), []).append(i)
    group_of = {i: j * a.groups // len(idx) for idx in by_cls.values() for j, i in enumerate(idx)}
    reps, crops, kept, dropped = [], [], 0, 0
    for i, r in enumerate(rows):
        img = cv2.imread(r["image_path"])
        if img is None:
            dropped += 1
            continue
        h, w = img.shape[:2]
        s = a.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        lm = pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        req = cfg[r["exercise"]]["required_landmarks"]
        if lm is None or float(np.mean(lm[req, 2])) < a.min_visibility:
            dropped += 1  # no person / body cut off: the app would ask to reposition, so don't train on it
            continue
        participant = f"{a.prefix}{group_of[i]:02d}"
        rep_id = f"{a.prefix}_img{i:05d}"
        feats = G.frame_features(lm, r["exercise"])
        rep_cfg = cfg[r["exercise"]]["rep"]
        prim = G.primary_angle(lm, rep_cfg["angle"], rep_cfg["combine"])
        feats["min_primary_angle"] = None if np.isnan(prim) else round(float(prim), 2)
        crop = cv2.resize(G.crop_square(img, G.person_bbox(lm, w, h)), (a.crop_size, a.crop_size),
                          interpolation=cv2.INTER_AREA)
        rel = Path("crops") / a.prefix / f"{rep_id}.jpg"
        cv2.imwrite(str(a.out / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cls = f"{r['exercise']}__{r['error']}"
        name = Path(r["image_path"]).name
        reps.append(dict(rep_id=rep_id, video=name, participant=participant, exercise=r["exercise"], error=r["error"],
                         class_name=cls, rep_idx=0, start=0, bottom=0, end=0, fps=0,
                         min_angle=feats["min_primary_angle"], visibility=round(float(np.mean(lm[req, 2])), 3),
                         required_visible=G.visible(lm, req), view=feats.get("view"), features=json.dumps(feats)))
        crops.append(dict(image_path=str(rel), rep_id=rep_id, video=name, participant=participant,
                          exercise=r["exercise"], error=r["error"], class_name=cls, offset=0, frame=0))
        kept += 1
    pose.lm.close()  # close explicitly: avoids a harmless but noisy error when Python exits
    append_rows(a.out / "reps.csv", REP_FIELDS, reps)
    append_rows(a.out / "crops.csv", CROP_FIELDS, crops)
    print(f"kept {kept} images, dropped {dropped} (unreadable, no person, or body not fully visible)")


if __name__ == "__main__":
    main()
