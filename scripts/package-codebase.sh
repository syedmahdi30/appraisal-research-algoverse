#!/usr/bin/env bash
# package-codebase.sh — Build and verify an anonymous, reproducible supplementary code archive.
#
# Usage:
#   ./scripts/package-codebase.sh [output_zip_path]
#
# Design note: this stages an explicit ALLOWLIST into a clean directory rather than
# zipping the repo and subtracting exclusions. A blacklist fails open — a new strategy
# doc, transcript, or handoff lands in the archive by default and nobody notices until
# a reviewer reads it. An allowlist fails closed. See the identity guard below, which
# is the backstop for anything the allowlist lets through anyway.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ZIP="${1:-submission_code.zip}"
case "$OUTPUT_ZIP" in /*) ;; *) OUTPUT_ZIP="$ROOT_DIR/$OUTPUT_ZIP" ;; esac
SCRATCH_DIR="${TMPDIR:-/tmp}/submission_verify_$$"
STAGE_DIR="$SCRATCH_DIR/stage"

echo "=== Packaging Codebase for Submission ==="
echo "Source: $ROOT_DIR"
echo "Target: $OUTPUT_ZIP"

rm -f "$OUTPUT_ZIP"
rm -rf "$SCRATCH_DIR"
mkdir -p "$STAGE_DIR"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

# ---------------------------------------------------------------------------
# 1. Stage the allowlist
# ---------------------------------------------------------------------------
# Everything a reviewer needs to re-derive the paper's numbers on CPU, and
# nothing else. Deliberately omitted: paper/ (the manuscript is submitted
# separately), docs/ (internal review panels, retraction audits, and a dropped
# venue's plan), and every root-level strategy/transcript/handoff file.
ALLOW_FILES=(
  README.md
  LICENSE
  requirements.txt
  requirements-qwen.txt
)
ALLOW_DIRS=(
  src
  tests
  config
  results
)

for f in "${ALLOW_FILES[@]}"; do
  [ -f "$f" ] || { echo "ERROR: allowlisted file missing: $f"; exit 1; }
  cp "$f" "$STAGE_DIR/"
done
for d in "${ALLOW_DIRS[@]}"; do
  [ -d "$d" ] || { echo "ERROR: allowlisted directory missing: $d"; exit 1; }
  rsync -a --quiet \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude '.pytest_cache' \
    "$d/" "$STAGE_DIR/$d/"
done

# scripts/ is staged file-by-file: colab_bootstrap.py hardcodes a Google Drive
# folder named after the research program, which is identifying.
mkdir -p "$STAGE_DIR/scripts"
for f in scripts/check_environment.py scripts/download_data.py \
         scripts/inspect_emotic.py scripts/smoke_test.py; do
  [ -f "$f" ] && cp "$f" "$STAGE_DIR/scripts/"
done

# ---------------------------------------------------------------------------
# 2. Sanitize machine-written provenance
# ---------------------------------------------------------------------------
# Analysis scripts record the parquet they read. On the authoring machine that
# is an absolute path containing the author's home directory and the program
# name — a deanonymization vector inside otherwise innocuous JSON.
echo "=== Sanitizing absolute paths ==="
sanitized=0
while IFS= read -r file; do
  if grep -q '"/\|: /\|/Users/\|/home/\|/content/' "$file" 2>/dev/null; then
    python3 - "$file" "$ROOT_DIR" <<'PY'
import re, sys
path, root = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8", errors="ignore") as fh:
    text = fh.read()
out = text.replace(root.rstrip("/") + "/", "")
# Any absolute prefix sitting in front of a repo-relative anchor is machine
# provenance, not data. Colab runs bury the program name in it
# (/content/<program-name>/data/raw/...), so match the prefix generically
# rather than enumerating the roots we happen to have seen.
out = re.sub(r'(?<=["\s,\[])(?:/[^/"\s,]+)+/(?=(?:data|results)/)', '', out)
if out != text:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print("  sanitized:", path.split("/stage/", 1)[-1])
PY
    sanitized=$((sanitized + 1))
  fi
done < <(find "$STAGE_DIR" -type f \( -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' -o -name '*.py' -o -name '*.csv' \))
[ "$sanitized" -eq 0 ] && echo "  (nothing to sanitize)"

# ---------------------------------------------------------------------------
# 3. Identity guard — fail closed
# ---------------------------------------------------------------------------
# The allowlist is only as good as the person who wrote it. This is the check
# that must never be removed: it fails the build rather than shipping a leak.
echo "=== Scanning staged tree for identifying strings ==="
IDENTITY_RE='syedmahdi|syedm|Sneheel|[Aa]lgoverse|github\.com/syed|/Users/|/home/[a-z]'
if leak=$(grep -rInE "$IDENTITY_RE" "$STAGE_DIR" 2>/dev/null); then
  echo "ERROR: identifying strings found in the staged archive."
  echo "This submission is double-blind. Fix the source or the allowlist; do not"
  echo "weaken this check."
  echo "$leak" | sed "s|$STAGE_DIR/||" | cut -c1-160 | head -20
  exit 1
fi
echo "  clean — no identifying strings"

# ---------------------------------------------------------------------------
# 4. Zip
# ---------------------------------------------------------------------------
( cd "$STAGE_DIR" && zip -qr "$OUTPUT_ZIP" . -x '.*' )
echo "Archive created successfully: $(du -h "$OUTPUT_ZIP" | awk '{print $1}')"

# ---------------------------------------------------------------------------
# 5. Verify in a clean sandbox
# ---------------------------------------------------------------------------
echo "=== Verifying Archive in Clean Sandbox ==="
unzip -q "$OUTPUT_ZIP" -d "$SCRATCH_DIR/unpacked"

for req_file in README.md LICENSE requirements.txt config/stage_f.yaml \
                src/experiments/stage_f_conflict.py tests/test_labels.py; do
  if [ ! -f "$SCRATCH_DIR/unpacked/$req_file" ]; then
    echo "ERROR: Missing required file in archive: $req_file"
    exit 1
  fi
done

echo "Running test suite inside unpacked archive..."
( cd "$SCRATCH_DIR/unpacked" && pytest -q tests/ )

echo "Running CPU analysis verification in sandbox..."
( cd "$SCRATCH_DIR/unpacked"
  python -m src.experiments.analyze_stage_f
  python -m src.experiments.analyze_stage_f_unbounded
  python -m src.experiments.analyze_stage_c_mechanism )

echo "=== Verification Succeeded ==="
echo "Supplementary archive is verified and ready for upload: $OUTPUT_ZIP"
