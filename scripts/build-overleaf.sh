#!/bin/bash
# Regenerate the two standalone Overleaf projects and prove each one still
# produces the same paper as the toggled source it came from.
#
#   ./scripts/build-overleaf.sh
#
# Compares extracted text, not just page counts, so a branch resolved the wrong
# way fails loudly instead of quietly shipping the other venue's paper.
set -e
cd "$(dirname "$0")/.."
SCRATCH="${TMPDIR:-/tmp}/overleaf-verify.$$"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

./scripts/build-paper.sh both >/dev/null

fail=0
for pair in "long:vlm4rwd:8" "short:interp4discovery:5"; do
  venue="${pair%%:*}"; rest="${pair#*:}"; dir="${rest%%:*}"; limit="${rest##*:}"
  python3 scripts/flatten-paper.py "$venue" --outdir "overleaf/$dir" >/dev/null
  mkdir -p "$SCRATCH/$dir"
  ( cd "overleaf/$dir" && tectonic -X compile main.tex --outdir "$SCRATCH/$dir" >/dev/null 2>&1 ) \
    || { echo "  $dir: BUILD FAILED"; fail=1; continue; }
  pdftotext "$SCRATCH/$dir/main.pdf" "$SCRATCH/$dir.flat.txt"
  pdftotext "paper-build/$venue.pdf" "$SCRATCH/$dir.toggled.txt"
  pages=$(pdfinfo "$SCRATCH/$dir/main.pdf" | awk '/^Pages:/{print $2}')
  if diff -q "$SCRATCH/$dir.flat.txt" "$SCRATCH/$dir.toggled.txt" >/dev/null; then
    printf '  %-18s %sp total, limit %sp — identical to toggled build\n' "$dir" "$pages" "$limit"
  else
    printf '  %-18s DIFFERS from toggled build:\n' "$dir"
    diff "$SCRATCH/$dir.toggled.txt" "$SCRATCH/$dir.flat.txt" | head -10
    fail=1
  fi
done
exit $fail
