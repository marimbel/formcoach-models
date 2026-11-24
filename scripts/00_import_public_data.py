#!/usr/bin/env python3
"""Bring public datasets into the pipeline's format. Public data is only ever used for TRAINING and
VALIDATION; the test set is always the group's own recordings of unseen people (see 02_make_splits.py).

Four sub-commands:

  inspect   Print a folder tree with file counts and example names (run this first on any download).

      python scripts/00_import_public_data.py inspect data/public/mendeley

  videos    Video datasets labelled good/bad form (e.g. Mendeley "Multi-View Raw Video Dataset of Seven
            Fitness Exercises", CC BY 4.0). Matches files by keywords in their path. GOOD clips are copied
            into data/raw/ as <prefix><subject>_squat_correct_t<n>.mp4 so 01_extract_reps.py can use them.
            BAD clips have no error type, so they go to a review sheet: the group watches each one, writes
            the error (knee_valgus, forward_lean, insufficient_depth, heel_lift, or skip) and applies it.

      python scripts/00_import_public_data.py videos data/public/mendeley --exercise squat \\
          --keyword squat --good good --bad bad --prefix mdl --out data/raw --review data/review/mendeley
      # ... fill data/review/mendeley/review.csv (column `error`), then:
      python scripts/00_import_public_data.py apply-review data/review/mendeley/review.csv --out data/raw

      # Mendeley push-ups work the same way, in their own review folder:
      python scripts/00_import_public_data.py videos data/public/mendeley --exercise push_up \\
          --keyword push --good good --bad bad --prefix mdl --out data/raw --review data/review/mendeley_push_up

  rehab     REHAB24-6 (Zenodo 13305826, CC BY-NC 4.0): videos from 2 cameras + Segmentation.csv, which marks every
            repetition's frames and whether it was done correctly (exercise 3 = push-ups, 5 = lunges, 6 = squats).
            Runs of correct reps are cut into clips in data/raw/; runs of incorrect reps go to a review sheet, since
            the dataset doesn't say which error it was. Participants become reh01..reh10 (train/val only).

      python scripts/00_import_public_data.py rehab data/public/rehab24 --segmentation data/public/rehab24/Segmentation.csv \\
          --map 5=lunge --map 3=push_up --prefix reh --out data/raw --review data/review/rehab
      # ... fill data/review/rehab/<exercise>/review.csv, then apply-review on each

  images    Image datasets with one folder per class (e.g. Zenodo 17558630 squat images: Good / Bad Back /
            Bad Heel, CC BY 4.0). Writes a manifest for 01b_extract_images.py.

      python scripts/00_import_public_data.py images data/public/zenodo_squat --prefix zsq \\
          --map "good=squat:correct" --map "bad back=squat:forward_lean" --map "bad heel=squat:heel_lift" \\
          --out data/public/zenodo_squat_manifest.csv

Folder/label matching is case-insensitive and ignores spaces, dashes and underscores, so "Bad_Back",
"bad-back" and "BadBack" all match "bad back".
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from formcoach.labels import errors_for, exercises  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEWS = ("front", "side", "diagonal", "oblique", "back")


def norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower())


def tokens(path: Path) -> list[str]:
    return [t for t in re.split(r"[\\/_\-\s.]+", str(path).lower()) if t]


# ---------------------------------------------------------------- inspect
def cmd_inspect(a):
    root = a.root
    if not root.exists():
        sys.exit(f"{root} does not exist")
    by_dir: dict[Path, Counter] = defaultdict(Counter)
    examples: dict[Path, list[str]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file():
            d = p.parent.relative_to(root)
            by_dir[d][p.suffix.lower()] += 1
            if len(examples[d]) < 3:
                examples[d].append(p.name)
    total = Counter()
    for d in sorted(by_dir):
        total.update(by_dir[d])
        if len(d.parts) <= a.depth:
            print(f"{str(d) or '.'}/  " + ", ".join(f"{n} {ext or '(none)'}" for ext, n in by_dir[d].most_common()))
            for e in examples[d]:
                print(f"      e.g. {e}")
    print("\nTOTAL:", ", ".join(f"{n} {ext}" for ext, n in total.most_common()))


# ---------------------------------------------------------------- videos
SUBJ_RE = re.compile(r"(?:^|[\\/_\-\s])(?:s|sub|subj|subject|p|participant|person|user)[_\-\s]?0*(\d{1,3})(?=$|[\\/_\-\s.])", re.I)


def subject_of(path: Path) -> str | None:
    m = SUBJ_RE.search(str(path))
    return m.group(1) if m else None


def view_of(path: Path) -> str:
    toks = tokens(path)
    for v in VIEWS:
        if v in toks:
            return v
    return "view"


def cmd_videos(a):
    if a.exercise not in exercises():
        sys.exit(f"--exercise must be one of {exercises()}")
    vids = sorted(p for p in a.root.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    kw, good, bad = a.keyword.lower(), a.good.lower(), a.bad.lower()
    a.out.mkdir(parents=True, exist_ok=True)
    a.review.mkdir(parents=True, exist_ok=True)
    rv = a.review / "review.csv"
    if rv.exists() and any((r.get("error") or "").strip() for r in csv.DictReader(open(rv))) and not a.force:
        sys.exit(f"{rv} already has labels; refusing to overwrite them. Use --force, or a different --review folder.")
    seen, rows, n_good, unknown_subj = set(), [], 0, 0
    take = Counter()
    for v in vids:
        rel = v.relative_to(a.root)
        toks = tokens(rel)
        if not any(kw in t for t in toks):
            continue
        q = "good" if good in toks else ("bad" if bad in toks else None)
        if q is None:
            continue
        subj = subject_of(rel)
        if subj is None:
            unknown_subj += 1
            subj = "x"
        key = (subj, q, v.stat().st_size)  # the dataset ships the same clips in 3 folder layouts: keep one copy
        if key in seen:
            continue
        seen.add(key)
        participant = f"{a.prefix}{subj}"
        take[(participant, q)] += 1
        t = f"t{take[(participant, q)]}"
        if q == "good":
            dst = a.out / f"{participant}_{a.exercise}_correct_{t}{v.suffix.lower()}"
            if not dst.exists():
                shutil.copy2(v, dst)
            n_good += 1
        else:
            sheet = a.review / f"{participant}_{t}_{view_of(rel)}.jpg"
            contact_sheet(v, sheet)
            rows.append({"source_path": str(v), "participant": participant, "take": t, "view": view_of(rel),
                         "sheet": str(sheet), "exercise": a.exercise, "error": ""})
    with open(a.review / "review.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_path", "participant", "take", "view", "sheet", "exercise", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"{n_good} good clips -> {a.out}")
    print(f"{len(rows)} bad clips need an error label: open {a.review}/*.jpg and fill the `error` column of "
          f"{a.review / 'review.csv'} with one of {[e for e in errors_for(a.exercise) if e != 'correct']} or 'skip'.")
    if unknown_subj:
        print(f"WARNING: {unknown_subj} clips had no recognisable subject id; they were all put under participant "
              f"'{a.prefix}x'. Check the names with `inspect` and adjust SUBJ_RE if needed.")


def contact_sheet(video: Path, out: Path, n: int = 6):
    """n evenly spaced frames side by side, so a reviewer can label the error without opening the video."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    tiles = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = 240 / h
        tiles.append(cv2.resize(fr, (int(w * s), 240)))
    cap.release()
    if tiles:
        cv2.imwrite(str(out), np.hstack(tiles))


def cmd_apply_review(a):
    rows = list(csv.DictReader(open(a.review_csv)))
    done = skipped = missing = 0
    for r in rows:
        err = (r.get("error") or "").strip().lower()
        if not err:
            missing += 1
            continue
        if err == "skip":
            skipped += 1
            continue
        if err not in errors_for(r["exercise"]):
            sys.exit(f"row {r['sheet']}: '{err}' is not an error for {r['exercise']}: {errors_for(r['exercise'])}")
        src = Path(r["source_path"])
        # takes 101+ for reviewed clips, so they never collide with the good clips' t1, t2, ...
        dst = a.out / f"{r['participant']}_{r['exercise']}_{err}_t{100 + int(r['take'][1:])}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        done += 1
    print(f"copied {done} labelled clips to {a.out}; skipped {skipped}; {missing} rows still unlabelled")


# ---------------------------------------------------------------- rehab (REHAB24-6)
VID_ID_RE = re.compile(r"PM[_\-\s]?(\d{3})", re.I)
CAM_RE = re.compile(r"cam(?:era)?[_\-\s]?(\d{2})", re.I)
SEG_FPS = 30  # Segmentation.csv frame numbers are at 30 fps


def cut_clip(src: Path, dst: Path, first: int, last: int, pad: int = 10) -> int:
    """Copy frames first..last (given at 30 fps) of src into dst. Returns frames written."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or SEG_FPS
    s = fps / SEG_FPS
    a, b = max(0, int(round((first - pad) * s))), int(round((last + pad) * s))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    out, n = None, 0
    for _ in range(a, b + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if out is None:
            h, w = fr.shape[:2]
            out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        out.write(fr)
        n += 1
    cap.release()
    if out is not None:
        out.release()
    return n


def runs_of(rows: list[dict]) -> list[list[dict]]:
    """Group a video's repetitions (sorted by frame) into runs of consecutive reps with the same correctness."""
    out: list[list[dict]] = []
    for r in rows:
        if out and out[-1][-1]["correctness"] == r["correctness"] and \
                int(r["repetition_number"]) == int(out[-1][-1]["repetition_number"]) + 1:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def cmd_rehab(a):
    want = {}
    for m in a.map:
        eid, _, ex = m.partition("=")
        if ex not in exercises():
            sys.exit(f"--map {m!r}: exercise must be one of {exercises()}")
        want[eid.strip()] = ex
    seg = [r for r in csv.DictReader(open(a.segmentation, newline=""), delimiter=";") if r["exercise_id"] in want]
    if not seg:
        sys.exit(f"no rows in {a.segmentation} for exercise ids {sorted(want)}")
    cams = {c.strip() for c in a.cameras.split(",")}
    files: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in sorted(a.root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        rel = str(p.relative_to(a.root))
        m = VID_ID_RE.search(rel)
        if not m:
            continue
        c = CAM_RE.search(rel)
        cam = c.group(1) if c else "x"
        if cam in cams or cam == "x":
            files[f"PM_{m.group(1)}"].append((cam, p))
    a.out.mkdir(parents=True, exist_ok=True)
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in seg:
        by_video[r["video_id"]].append(r)
    review_rows: dict[str, list[dict]] = defaultdict(list)
    for ex in set(want.values()):
        rv = a.review / ex / "review.csv"
        if rv.exists() and any((r.get("error") or "").strip() for r in csv.DictReader(open(rv))) and not a.force:
            sys.exit(f"{rv} already has labels; refusing to overwrite them. Use --force, or a different --review folder.")
        (a.review / ex).mkdir(parents=True, exist_ok=True)
    take, missing, n_good = Counter(), [], Counter()
    for vid, rows in sorted(by_video.items()):
        if not files.get(vid):
            missing.append(vid)
            continue
        rows.sort(key=lambda r: int(r["first_frame"]))
        ex = want[rows[0]["exercise_id"]]
        participant = f"{a.prefix}{int(rows[0]['person_id']):02d}"
        for run in runs_of(rows):
            good = run[0]["correctness"].strip() == "1"
            first, last = int(run[0]["first_frame"]), int(run[-1]["last_frame"])
            for cam, path in files[vid]:
                key = (participant, ex, "good" if good else "bad")
                take[key] += 1
                t = f"t{take[key]}"
                view = f"cam{cam}"
                if good:
                    dst = a.out / f"{participant}_{ex}_correct_{t}.mp4"
                    if not dst.exists() and cut_clip(path, dst, first, last) == 0:
                        print(f"WARNING: could not read frames {first}-{last} of {path}")
                    n_good[ex] += 1
                else:
                    clip = a.review / ex / f"{participant}_{t}_{view}.mp4"
                    if not clip.exists():
                        cut_clip(path, clip, first, last)
                    sheet = clip.with_suffix(".jpg")
                    contact_sheet(clip, sheet)
                    review_rows[ex].append({"source_path": str(clip), "participant": participant, "take": t,
                                            "view": view, "sheet": str(sheet), "exercise": ex, "error": ""})
    for ex in set(want.values()):
        with open(a.review / ex / "review.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source_path", "participant", "take", "view", "sheet", "exercise", "error"])
            w.writeheader()
            w.writerows(review_rows[ex])
        errs = [e for e in errors_for(ex) if e != "correct"]
        print(f"{ex}: {n_good[ex]} correct clips -> {a.out}; {len(review_rows[ex])} incorrect clips to label in "
              f"{a.review / ex / 'review.csv'} with one of {errs} or 'skip'")
    if missing:
        print(f"WARNING: no video found for {len(missing)} video ids (e.g. {missing[:5]}). Run `inspect` on {a.root} and "
              "check that file names contain the id (PM_021) and the camera (Camera17).")


# ---------------------------------------------------------------- images
def cmd_images(a):
    mapping = {}
    for m in a.map:
        folder, _, target = m.partition("=")
        ex, _, err = target.partition(":")
        if ex not in exercises() or err not in errors_for(ex):
            sys.exit(f"--map {m!r}: target must be <exercise>:<error> from labels.json")
        mapping[norm(folder)] = (ex, err)
    rows, counts, unmatched = [], Counter(), Counter()
    for p in sorted(a.root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        hit = None
        for part in reversed(p.relative_to(a.root).parts[:-1]):  # nearest folder name wins
            if norm(part) in mapping:
                hit = mapping[norm(part)]
                break
        if hit is None:
            unmatched[str(p.parent.relative_to(a.root))] += 1
            continue
        ex, err = hit
        rows.append({"image_path": str(p), "exercise": ex, "error": err})
        counts[f"{ex}__{err}"] += 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "exercise", "error"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images -> {a.out}")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    if unmatched:
        print("Folders with no --map (ignored):", dict(unmatched.most_common(8)))
    print(f"Next: python scripts/01b_extract_images.py --manifest {a.out} --prefix {a.prefix} --out data/processed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect")
    s.add_argument("root", type=Path)
    s.add_argument("--depth", type=int, default=4)
    s = sub.add_parser("videos")
    s.add_argument("root", type=Path)
    s.add_argument("--exercise", required=True)
    s.add_argument("--keyword", required=True, help="token that marks this exercise in the path, e.g. squat")
    s.add_argument("--good", default="good")
    s.add_argument("--bad", default="bad")
    s.add_argument("--prefix", required=True, help="participant prefix marking this source, e.g. mdl")
    s.add_argument("--out", type=Path, default=Path("data/raw"))
    s.add_argument("--review", type=Path, required=True)
    s.add_argument("--force", action="store_true", help="overwrite an existing, partly labelled review sheet")
    s = sub.add_parser("apply-review")
    s.add_argument("review_csv", type=Path)
    s.add_argument("--out", type=Path, default=Path("data/raw"))
    s = sub.add_parser("rehab")
    s.add_argument("root", type=Path, help="folder with the unzipped REHAB24-6 videos")
    s.add_argument("--segmentation", type=Path, required=True, help="Segmentation.csv from the Zenodo record")
    s.add_argument("--map", action="append", required=True, help='"exercise_id=exercise", e.g. 5=lunge, 3=push_up')
    s.add_argument("--prefix", default="reh")
    s.add_argument("--cameras", default="17,18", help="which cameras to use (17 and/or 18)")
    s.add_argument("--out", type=Path, default=Path("data/raw"))
    s.add_argument("--review", type=Path, required=True)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("images")
    s.add_argument("root", type=Path)
    s.add_argument("--map", action="append", required=True, help='"folder name=exercise:error", repeatable')
    s.add_argument("--prefix", required=True)
    s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    {"inspect": cmd_inspect, "videos": cmd_videos, "apply-review": cmd_apply_review, "rehab": cmd_rehab,
     "images": cmd_images}[a.cmd](a)


if __name__ == "__main__":
    main()
