#!/bin/bash
# Build both venue variants of the paper and report main-text page counts.
#   ./scripts/build-paper.sh          -> build both
#   ./scripts/build-paper.sh long     -> VLM4RWD only (8pp limit)
#   ./scripts/build-paper.sh short    -> Interp4Discovery only (5pp limit)
#
# Requires tectonic (brew install tectonic) and poppler's pdfinfo/pdftotext.
# venue.tex is rewritten per target and restored to \shortfalse on exit.
#
# Fails the build on two conditions, both of which are only visible in the
# rendered PDF: a target over its page limit, and a float deferred out of the
# appendix onto a bibliography page.
set -e
cd "$(dirname "$0")/../paper"
OUT="${BUILD_OUT:-../paper-build}"
mkdir -p "$OUT"
trap 'sed -i "" "s/^\\\\shorttrue/\\\\shortfalse/" venue.tex 2>/dev/null || true' EXIT

measure() { # $1=pdf -> echoes "<pages before References>|<pages where a float hit the bibliography>"
  python3 - "$1" <<'PYEOF'
import re
import subprocess
import sys

pdf = sys.argv[1]
info = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout
total = int(info.split('Pages:')[1].split()[0])
pages = [subprocess.run(['pdftotext', '-f', str(p), '-l', str(p), pdf, '-'],
                        capture_output=True, text=True).stdout
         for p in range(1, total + 1)]

ref = next((i + 1 for i, t in enumerate(pages) if 'References' in t), None)
if ref is None:
    raise SystemExit(f"no References page found (total {total})")

# A float sharing a page with reference entries was deferred out of the appendix
# and into the bibliography. LaTeX does this silently, so check the output.
bib = re.compile(r'arXiv:|In Proceedings|In International Conference')
flo = re.compile(r'(Table|Figure) \d+:')
bad = [i + 1 for i, t in enumerate(pages)
       if i + 1 >= ref and bib.search(t) and flo.search(t)]

print(f"{ref - 1}|{','.join(map(str, bad))}")
PYEOF
}

build() { # $1=short|long  $2=limit  $3=venue name
  if [ "$1" = short ]; then sed -i "" 's/^\\shortfalse/\\shorttrue/' venue.tex
  else sed -i "" 's/^\\shorttrue/\\shortfalse/' venue.tex; fi
  tectonic -X compile neurips_2026.tex --outdir "$OUT" >/dev/null 2>&1 \
    || { echo "  $3: BUILD FAILED"; tectonic -X compile neurips_2026.tex --outdir "$OUT" 2>&1 | grep -i error | head -10; return 1; }
  mv "$OUT/neurips_2026.pdf" "$OUT/$1.pdf"
  result="$(measure "$OUT/$1.pdf")"
  pages="${result%%|*}"
  collide="${result#*|}"
  if [ "$pages" -le "$2" ]; then status="OK"; else status="OVER LIMIT"; fi
  printf '  %-18s limit %sp -> %sp main text, References p%s [%s]\n' \
    "$3" "$2" "$pages" "$((pages + 1))" "$status"
  if [ -n "$collide" ]; then
    printf '  %-18s FLOAT LANDED IN BIBLIOGRAPHY on page(s) %s\n' "" "$collide"
    return 1
  fi
  [ "$pages" -le "$2" ]
}

case "${1:-both}" in
  long)  build long  8 VLM4RWD ;;
  short) build short 5 Interp4Discovery ;;
  both)  build long  8 VLM4RWD; build short 5 Interp4Discovery ;;
  *) echo "usage: $0 [long|short|both]"; exit 1 ;;
esac
