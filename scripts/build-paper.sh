#!/bin/bash
# Build both venue variants of the paper and report main-text page counts.
#   ./scripts/build-paper.sh          -> build both
#   ./scripts/build-paper.sh long     -> VLM4RWD only (8pp limit)
#   ./scripts/build-paper.sh short    -> Interp4Discovery only (5pp limit)
#
# Requires tectonic (brew install tectonic) and poppler's pdfinfo/pdftotext.
# venue.tex is rewritten per target and restored to \shortfalse on exit.
set -e
cd "$(dirname "$0")/../paper"
OUT="${BUILD_OUT:-build}"
mkdir -p "$OUT"
trap 'sed -i "" "s/^\\\\shorttrue/\\\\shortfalse/" venue.tex 2>/dev/null || true' EXIT

measure() { # $1=pdf  -> echoes main-text page count
  python3 - "$1" <<'PY'
import subprocess,sys
pdf=sys.argv[1]
tot=int(subprocess.run(['pdfinfo',pdf],capture_output=True,text=True).stdout.split('Pages:')[1].split()[0])
for p in range(1,tot+1):
    t=subprocess.run(['pdftotext','-f',str(p),'-l',str(p),pdf,'-'],capture_output=True,text=True).stdout
    if 'References' in t:
        print(f"{p-1} full + partial p{p}  (total {tot})"); break
else:
    print(f"no References page found (total {tot})")
PY
}

build() { # $1=short|long  $2=limit  $3=venue name
  if [ "$1" = short ]; then sed -i "" 's/^\\shortfalse/\\shorttrue/' venue.tex
  else sed -i "" 's/^\\shorttrue/\\shortfalse/' venue.tex; fi
  tectonic -X compile neurips_2026.tex --outdir "$OUT" >/dev/null 2>&1 \
    || { echo "  $3: BUILD FAILED"; tectonic -X compile neurips_2026.tex --outdir "$OUT" 2>&1 | grep -i error | head -10; return 1; }
  mv "$OUT/neurips_2026.pdf" "$OUT/$1.pdf"
  printf '  %-18s limit %sp -> %s\n' "$3" "$2" "$(measure "$OUT/$1.pdf")"
}

case "${1:-both}" in
  long)  build long  8 VLM4RWD ;;
  short) build short 5 Interp4Discovery ;;
  both)  build long  8 VLM4RWD; build short 5 Interp4Discovery ;;
  *) echo "usage: $0 [long|short|both]"; exit 1 ;;
esac
