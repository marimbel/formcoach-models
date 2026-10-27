#!/usr/bin/env python3
"""Videos -> pose landmarks -> segmented reps -> bottom-of-rep body crops + features.

    python scripts/01_extract_reps.py --videos data/raw --out data/processed

Outputs (in --out):
  landmarks/<video>.npz   per-frame landmarks (T,33,3) in pixels + fps, for re-use
  crops/<video>/rep<k>_o<offset>.jpg   256x256 square body crops at the bottom frame +- offsets
  qa/<video>.png          angle signal with detected reps (check these before training!)
  reps.csv                one row per rep: labels, timing, pose quality, features (JSON)
  crops.csv               one row per crop image
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach import geometry as G  # noqa: E402
from formcoach.labels import load_config, parse_video_name  # noqa: E402
from formcoach.reps import interpolate_gaps, segment_reps, smooth_landmarks  # noqa: E402

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi"}


class MediaPipePose:
    """Model C (offline twin): MediaPipe BlazePose via the Tasks API. The app uses ML Kit,
    which wraps the same BlazePose topology (33 landmarks), so features match."""

    def __init__(self, model_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray | None:
        h, w = rgb.shape[:2]
        res = self.landmarker.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not res.pose_landmarks:
            return None
        return np.array([[p.x * w, p.y * h, p.visibility] for p in res.pose_landmarks[0]], dtype=np.float32)


def ensure_model(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def read_video(path: Path, max_side: int):
    cap = cv2.VideoCapture(str(path))  # OpenCV >= 4.5 applies phone rotation metadata automatically
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        frames.append(bgr)
    cap.release()
    return frames, float(fps)


def qa_plot(sig, reps, path: Path, title: str, top: float, bottom: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sig, lw=1)
    ax.axhline(top, ls="--", c="g", lw=0.8)
    ax.axhline(bottom, ls="--", c="r", lw=0.8)
    for r in reps:
        ax.axvspan(r.start, r.end, alpha=0.15)
        ax.plot(r.bottom, r.min_angle, "kv")
    ax.set_title(f"{title}: {len(reps)} reps")
    ax.set_xlabel("frame")
    ax.set_ylabel("angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pose-model", type=Path, default=Path("models/pose_landmarker_full.task"))
    ap.add_argument("--max-side", type=int, default=960, help="downscale frames for speed")
    ap.add_argument("--offsets", default="-2,0,2", help="frames around the bottom to crop")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--reuse-landmarks", action="store_true", help="skip pose inference when the .npz exists")
    args = ap.parse_args()

    cfg = load_config()
    offsets = [int(o) for o in args.offsets.split(",")]
    for d in ("landmarks", "crops", "qa"):
        (args.out / d).mkdir(parents=True, exist_ok=True)
    videos = sorted(p for p in args.videos.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        sys.exit(f"No videos under {args.videos}")
    pose = None

    reps_f = open(args.out / "reps.csv", "w", newline="")
    crops_f = open(args.out / "crops.csv", "w", newline="")
    rw = csv.DictWriter(reps_f, fieldnames=["rep_id", "video", "participant", "exercise", "error", "class_name",
                                            "rep_idx", "start", "bottom", "end", "fps", "min_angle",
                                            "visibility", "required_visible", "view", "features"])
    cw = csv.DictWriter(crops_f, fieldnames=["image_path", "rep_id", "video", "participant", "exercise", "error",
                                             "class_name", "offset", "frame"])
    rw.writeheader()
    cw.writeheader()
    n_reps = 0
    for vp in videos:
        try:
            meta = parse_video_name(vp)
        except ValueError as e:
            print(f"SKIP {vp.name}: {e}")
            continue
        ex_cfg = cfg["exercises"][meta.exercise]
        frames, fps = read_video(vp, args.max_side)
        if not frames:
            print(f"SKIP {vp.name}: unreadable")
            continue
        h, w = frames[0].shape[:2]
        npz = args.out / "landmarks" / f"{vp.stem}.npz"
        if args.reuse_landmarks and npz.exists():
            lms = np.load(npz)["landmarks"]
        else:
            pose = pose or MediaPipePose(ensure_model(args.pose_model))
            lms = np.full((len(frames), 33, 3), np.nan, dtype=np.float32)
            for t, bgr in enumerate(frames):
                out = pose(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(t * 1000 / fps))
                if out is not None:
                    lms[t] = out
            np.savez_compressed(npz, landmarks=lms, fps=fps, width=w, height=h)
        sm = smooth_landmarks(lms, fps)
        rep_cfg = ex_cfg["rep"]
        sig = np.array([G.primary_angle(f, rep_cfg["angle"], rep_cfg["combine"]) if not np.isnan(f).any()
                        else np.nan for f in sm])
        sig = interpolate_gaps(sig)
        reps = segment_reps(sig, fps, rep_cfg["top"], rep_cfg["bottom_enter"], rep_cfg["min_rep_seconds"])
        qa_plot(sig, reps, args.out / "qa" / f"{vp.stem}.png", vp.stem, rep_cfg["top"], rep_cfg["bottom_enter"])
        (args.out / "crops" / vp.stem).mkdir(exist_ok=True)
        req = ex_cfg["required_landmarks"]
        for k, r in enumerate(reps):
            lm_b = sm[r.bottom]
            rep_id = f"{vp.stem}_r{k:02d}"
            feats = G.frame_features(lm_b, meta.exercise)
            feats["min_primary_angle"] = round(r.min_angle, 2)
            seg = sm[r.start:r.end + 1, req, 2]
            vis = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0
            rw.writerow(dict(rep_id=rep_id, video=vp.name, participant=meta.participant, exercise=meta.exercise,
                             error=meta.error, class_name=meta.class_name, rep_idx=k, start=r.start,
                             bottom=r.bottom, end=r.end, fps=round(fps, 3), min_angle=round(r.min_angle, 2),
                             visibility=round(vis, 3), required_visible=G.visible(lm_b, req),
                             view=feats.get("view"), features=json.dumps(feats)))
            for o in offsets:
                t = int(np.clip(r.bottom + o, 0, len(frames) - 1))
                if np.isnan(sm[t]).any():
                    continue
                crop = G.crop_square(frames[t], G.person_bbox(sm[t], w, h))
                crop = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
                rel = Path("crops") / vp.stem / f"rep{k:02d}_o{o:+d}.jpg"
                cv2.imwrite(str(args.out / rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                cw.writerow(dict(image_path=str(rel), rep_id=rep_id, video=vp.name, participant=meta.participant,
                                 exercise=meta.exercise, error=meta.error, class_name=meta.class_name,
                                 offset=o, frame=t))
        n_reps += len(reps)
        print(f"{vp.name}: {len(frames)} frames @ {fps:.1f} fps -> {len(reps)} reps")
    reps_f.close()
    crops_f.close()
    print(f"Done: {n_reps} reps from {len(videos)} videos. Review {args.out / 'qa'} before training.")


if __name__ == "__main__":
    main()
