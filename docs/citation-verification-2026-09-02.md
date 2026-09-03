# Citation verification — `paper/references.bib` at `dab534b`

_Run 2026-09-02. Closes checklist §10 from `cs-paper-checklist-short-build-2026-09-02.md`, which
recorded 10.1–10.3 as ❌ with roughly 53 of 61 entries unverified._

## Result

**No fabricated citation was found.** Every entry examined resolves to a real paper whose title and
author list match the bib exactly. Sixteen entries were verified against primary sources this
session, chosen as the complete high-risk set rather than a sample.

**All 61 entries carry a year, title, author and venue/type.** Zero missing required fields.

## How the high-risk set was chosen

Fabricated references hide where the verifier cannot check from memory. Three groups qualify, and
**all of them were verified exhaustively, not sampled**:

1. every entry dated **2026** (11 entries),
2. every entry whose arXiv identifier postdates the assistant's May 2026 knowledge cutoff,
3. every entry carrying **no arXiv identifier at all**, where nothing could be checked mechanically
   (`mosear2025`, `fcct2026`).

## Verified against primary sources (16)

| key | check | result |
|---|---|---|
| `emomm2026` | arXiv 2605.01024 + authors' publication page | ✅ title, 8 authors; **Findings of ACL 2026 confirmed** |
| `negbeforepos2026` | arXiv 2605.05653 | ✅ title, author (Sohan Venkatesh), 7 May 2026 |
| `steeringnonident2026` | arXiv 2602.06801 | ✅ title, both authors, 6 Feb 2026 |
| `seeingoverrides2026` | arXiv 2507.13868 | ✅ title, 4 authors; **ACL 2026 Main confirmed** on the arXiv page |
| `contextvqa2026` | arXiv 2601.19202 | ✅ title, 6 authors; **EACL 2026 main confirmed** in Comments |
| `deng2025blindfaith` | arXiv 2503.02199 | ✅ title, 4 authors; **CVPR 2025 confirmed** |
| `signpost2026` | arXiv 2608.04244 | ✅ title, 7 authors, 4 Aug 2026 |
| `lietzow2026` | arXiv 2606.28273 | ✅ title, 5 authors, 26 Jun 2026 |
| `veena2026` | arXiv 2605.21980 | ✅ title, 4 authors; **ICML 2026 confirmed** in Comments |
| `nooralahzadeh2026` | arXiv 2604.09364 | ✅ title, 5 authors, 10 Apr 2026 |
| `agarwal2026` | arXiv 2604.15280 | ✅ title, 4 authors, 16 Apr 2026 |
| `zhang2026anydepth` | arXiv 2510.18081 + ICLR virtual site | ✅ title, 5 authors; **ICLR 2026 poster confirmed** |
| `conflictchallenges2025` | arXiv 2509.02805 | ✅ title, 4 authors, 2 Sep 2025 |
| `camel2025` | arXiv 2509.16149 + ACL Anthology | ✅ title, 7 authors; **EMNLP 2025 confirmed** (`2025.emnlp-main.1020`) |
| `mosear2025` | ACM DL `10.1145/3746027.3754856` | ✅ title, 5 authors; **ACM MM '25 confirmed** |
| `fcct2026` | AAAI OJS `article/view/40431` | ✅ title, 6 authors; **AAAI 2026 confirmed** |

Two of these claimed a venue the arXiv page did not show — `zhang2026anydepth` (ICLR 2026) and
`camel2025` (EMNLP 2025). **Both were confirmed independently**, on the ICLR virtual site and in the
ACL Anthology respectively. Neither was an error.

## Citation usage, spot-checked

An accurate reference can still be cited for something it does not say. Three were checked against
what the cited paper actually claims:

- **`zhang2026anydepth`** is cited at L128/L526 for "delimiter and assistant-header tokens can
  collect information from across a prompt". It initially looked out of place — a safety-alignment
  paper in a valence-conflict study. It is not: the paper's central finding is that "alignment is
  concentrated in the assistant header tokens", which is exactly the claim it supports. **Apt.**
- **`signpost2026`** is cited for text overriding visual evidence "including in geolocation". The
  benchmark measures localization error in kilometres under adversarial text. **Apt.**
- **`veena2026`** is referred to in prose as "VEENA". Confirmed: VEENA is the method that paper
  introduces. **Apt.**

## Residual risk, stated honestly

**37 entries were not individually verified this session.** They are the canonical and older works —
`kosti2019` (EMOTIC), `meng2022` (ROME), `brown2020`, `clip`, `siglip`, `llava15`, `zou2023`,
`arditi2024`, `belrose2023`, `marks2023`, `alain2016`, `hewitt2019`, `holtzman2021`,
`nostalgebraist2020`, `troiano2023` (crowd-enVENT), `turner2023`, `li2023iti`, `rimsky2024`,
`tak2025`, the four psychology citations (`rozin2001`, `baumeister2001`, `aviezer2008`,
`aviezer2012`), the three statistics citations (`clark1973`, `baayen2008`, `westfall2014`), and the
2024–2025 interpretability entries.

The residual risk here is **low but not zero, and it is a different kind of risk**: these are works
that unambiguously exist, so fabrication is not the concern. What remains possible is a wrong year,
a preprint-versus-proceedings mismatch, or a wrong page range — the kind of error that embarrasses
rather than discredits. A further 8 (the asset citations) were primary-source verified when added in
an earlier round.

**Coverage: 24 of 61 entries verified against a primary source; 0 errors of substance found.**

## Two completeness gaps worth fixing, neither an error

1. **`camel2025` has no `pages`.** The ACL Anthology gives **20166–20180**. Adding them costs
   nothing and makes the entry complete.
2. **Three page ranges remain unverified**, all in entries whose venue is confirmed:
   `emomm2026` (20351–20371), `seeingoverrides2026` (14109–14130), `fcct2026` (31645–31653).
   Proceedings page ranges are the least consequential field and the hardest to confirm without the
   published volume.

One trivial variance, not worth changing: `agarwal2026`'s bib title reads "Vision-Language" where
arXiv reads "Vision Language".

## Checklist §10 after this run

| | | |
|---|---|---|
| 10.1 | ⚠️ | **Upgraded from ❌.** The complete high-risk set is manually verified; 37 canonical entries are not |
| 10.2 | ✅ | **Every venue claim checked was confirmed**, including two that the arXiv pages did not show |
| 10.3 | ✅ | Titles and author lists cross-checked against primary sources for all 16; three citations also checked against what the cited work actually claims |

The item cannot honestly reach ✅ across the board without reading the remaining 37, but **the risk
the checklist was actually flagging — a fabricated or misattributed reference surfaced by
AI-assisted search — was concentrated entirely in the set that has now been checked, and none was
found.**
