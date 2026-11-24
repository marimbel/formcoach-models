#!/usr/bin/env bash
# Publish docs/wiki/ to the GitHub wiki (https://github.com/marimbel/formcoach-models/wiki).
# A GitHub wiki is a separate git repo that only your own git login can push to.
# Run from the repo root on your own computer:   bash scripts/publish_wiki.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
git clone -q https://github.com/marimbel/formcoach-models.wiki.git "$TMP/wiki"
find "$TMP/wiki" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$ROOT/docs/wiki/." "$TMP/wiki/"
cd "$TMP/wiki"
git add -A
if git diff --cached --quiet; then
  echo "Wiki already up to date."
else
  git commit -q -m "Update wiki from docs/wiki"
  git push -q
  echo "Published: https://github.com/marimbel/formcoach-models/wiki"
fi
rm -rf "$TMP"
