#!/usr/bin/env python3
"""Fail if files shared with the backend have drifted (train/inference format mismatch).

    python scripts/check_prompting_sync.py ../formcoach-app
"""
import filecmp
import sys
from pathlib import Path

SHARED = ["prompting.py", "coaching_kb.py"]
here = Path(__file__).resolve().parent.parent / "formcoach"
app = Path(sys.argv[1] if len(sys.argv) > 1 else "../formcoach-app") / "backend" / "app"
bad = []
for name in SHARED:
    a, b = here / name, app / name
    if not b.exists() or not filecmp.cmp(a, b, shallow=False):
        bad.append(f"cp {a} {b}")
if bad:
    sys.exit("OUT OF SYNC, run:\n  " + "\n  ".join(bad))
print("shared files in sync: " + ", ".join(SHARED))
