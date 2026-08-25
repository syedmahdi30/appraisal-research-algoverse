#!/usr/bin/env python3
"""Flatten the venue-toggled paper into standalone source for one venue.

paper/neurips_2026.tex carries both submissions behind \\ifshort. That is the
right shape for the repo and the wrong shape for a reviewer, who would see both
venues' abstracts and introductions in every file. This resolves the toggle for
one venue and inlines the shared table files, producing source that reads as a
single paper.

    python3 scripts/flatten-paper.py short --outdir overleaf/interp4discovery
    python3 scripts/flatten-paper.py long  --outdir overleaf/vlm4rwd

Correctness is checked by build, not by inspection: scripts/build-overleaf.sh
compiles the flattened output and compares it against the toggled build.

\\if@anonymous appears in the same file and a LaTeX comment discusses it, so the
scanner masks comments and tracks conditional nesting rather than pattern-matching
\\else and \\fi directly.
"""
import argparse
import pathlib
import re
import shutil
import sys

COND = re.compile(r'\\(if[a-zA-Z@]*|else|fi)(?![a-zA-Z@])')


def mask_comments(text: str) -> str:
    """Blank out comment bodies, preserving every character offset."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '\\' and i + 1 < n:          # skip escaped chars, incl. \%
            i += 2
            continue
        if c == '%':
            j = text.find('\n', i)
            j = n if j == -1 else j
            for k in range(i, j):
                out[k] = ' '
            i = j
            continue
        i += 1
    return ''.join(out)


def resolve(text: str, short: bool) -> str:
    """Replace every \\ifshort...[\\else...]\\fi with the branch this venue keeps."""
    while True:
        shadow = mask_comments(text)
        m = re.search(r'\\ifshort(?![a-zA-Z@])', shadow)
        if not m:
            return text
        body_start = m.end()
        depth, else_pos, pos = 1, None, body_start
        while depth:
            m2 = COND.search(shadow, pos)
            if not m2:
                sys.exit("unbalanced \\ifshort: no matching \\fi")
            kind = m2.group(1)
            pos = m2.end()
            if kind.startswith('if'):
                depth += 1
            elif kind == 'fi':
                depth -= 1
                if depth == 0:
                    fi_start, fi_end = m2.start(), m2.end()
            elif kind == 'else' and depth == 1:
                else_pos = (m2.start(), m2.end())
        if else_pos:
            true_branch = text[body_start:else_pos[0]]
            false_branch = text[else_pos[1]:fi_start]
        else:
            true_branch, false_branch = text[body_start:fi_start], ''
        kept = true_branch if short else false_branch
        text = text[:m.start()] + kept.strip('\n') + text[fi_end:]


def inline_tables(text: str, src: pathlib.Path) -> str:
    def sub(m):
        f = src / (m.group(1) + '.tex')
        if not f.exists():
            sys.exit(f"missing {f}")
        return f.read_text().rstrip('\n')
    return re.sub(r'\\input\{(tables/[^}]+)\}', sub, text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('venue', choices=['short', 'long'])
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()

    src = pathlib.Path('paper')
    out = pathlib.Path(a.outdir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    text = (src / 'neurips_2026.tex').read_text()
    # the toggle itself is meaningless once resolved
    text = re.sub(r'^% Venue page-limit toggle\..*?\n\\input\{venue\}\n', '',
                  text, count=1, flags=re.S | re.M)
    text = resolve(text, short=(a.venue == 'short'))
    text = inline_tables(text, src)
    if re.search(r'\\ifshort|\\shorttrue|\\shortfalse', mask_comments(text)):
        sys.exit("toggle survived flattening")

    (out / 'main.tex').write_text(text)
    for f in ('neurips_2026.sty', 'references.bib', 'checklist.tex'):
        shutil.copy(src / f, out / f)

    # ship only the figures this venue actually references; the unused ones
    # include plots from withdrawn experiments
    used = sorted(set(re.findall(r'\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}', text)))
    (out / 'figures').mkdir()
    for name in used:
        f = src / 'figures' / name
        if not f.exists():
            sys.exit(f"missing figure {f}")
        shutil.copy(f, out / 'figures' / name)
    print(f"{a.venue} -> {out}/main.tex ({len(used)} figures)")


if __name__ == '__main__':
    main()
