#!/usr/bin/env bash
# package-codebase.sh — Build and verify an anonymous, reproducible supplementary code archive.
#
# Usage:
#   ./scripts/package-codebase.sh [output_zip_path]
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ZIP="${1:-submission_code.zip}"
SCRATCH_DIR="${TMPDIR:-/tmp}/submission_verify_$$"

echo "=== Packaging Codebase for Submission ==="
echo "Source: $ROOT_DIR"
echo "Target: $OUTPUT_ZIP"

rm -f "$OUTPUT_ZIP"
rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

# Create zip archive while excluding private, personal, temporary, and gated files
zip -qr "$OUTPUT_ZIP" . \
  -x "*.git*" \
  -x "*.DS_Store*" \
  -x "*__pycache__*" \
  -x "*.pytest_cache*" \
  -x "*.worktrees*" \
  -x "*.superpowers*" \
  -x "*.agents*" \
  -x "*.claude*" \
  -x "data/raw/*" \
  -x "data/processed/*" \
  -x "context*.md" \
  -x "session*context*.md" \
  -x "compass_artifact_*.md" \
  -x "graphify-out/*" \
  -x "overleaf/*" \
  -x "paper-build/*" \
  -x "archive/*" \
  -x "$OUTPUT_ZIP"

echo "Archive created successfully: $(du -h "$OUTPUT_ZIP" | awk '{print $1}')"

echo "=== Verifying Archive in Clean Sandbox ==="
unzip -q "$OUTPUT_ZIP" -d "$SCRATCH_DIR/unpacked"

# Check that critical files exist
for req_file in README.md LICENSE requirements.txt config/stage_f.yaml src/experiments/stage_f_conflict.py tests/test_labels.py; do
  if [ ! -f "$SCRATCH_DIR/unpacked/$req_file" ]; then
    echo "ERROR: Missing required file in archive: $req_file"
    exit 1
  fi
done

# Run unit tests in the sandbox
echo "Running test suite inside unpacked archive..."
(
  cd "$SCRATCH_DIR/unpacked"
  pytest tests/
)

# Run CPU analysis verification
echo "Running CPU analysis verification in sandbox..."
(
  cd "$SCRATCH_DIR/unpacked"
  python -m src.experiments.analyze_stage_f
  python -m src.experiments.analyze_stage_f_unbounded
  python -m src.experiments.analyze_stage_c_mechanism
)

echo "=== Verification Succeeded ==="
echo "Supplementary archive is verified and ready for upload: $OUTPUT_ZIP"
