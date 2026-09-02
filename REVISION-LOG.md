# Revision log — cross-modal valence asymmetry paper

## Round 1 — 2026-09-01

Source: `paper/audit.md` (paper-audit, 2026-09-01). Scope this round: **F2 and F3 only.**
Starting state: tip `adc0180`, `paper/neurips_2026.tex` unmodified since 2026-08-29 14:21 —
no Overleaf-side drift to reconcile.

### Applied

- **F2** (MAJOR/WRITING) — `tab:pairs` (`neurips_2026.tex` L402–419). Two changes. The column
  headers now read $\overline{\Delta}_{\rm neg}$, $\overline{\Delta}_{\rm pos}$ and
  $|\overline{\Delta}_{\rm neg}|-|\overline{\Delta}_{\rm pos}|$, and the caption states that each
  row averages over the 62 photographs before the absolute value is taken while the headline
  contrast takes absolute values per photograph first, so the six rows average to $+1.172$ rather
  than $+1.148$.
  Rationale: the two numbers are both correct and both sourced —
  `results/stage_f/conflict_qwen3-vl-8b-instruct_minimal_analysis.json` carries
  `paired_asymmetry = 1.1480072716435483` alongside the six `per_pair/*/asymmetry` values that mean
  to $1.1717$. Recomputing the table to the per-photograph estimator would have changed six
  published values and the ratio column for a presentational problem, so the fix distinguishes the
  estimators instead of merging them. The bar notation does the work at a glance; note that the
  main text at L191 uses the *unbarred* form for the per-photograph statistic, so the two now read
  consistently.

- **F3a** (MAJOR/WRITING) — `tab:crosspatch` gains a second panel reporting the layer-18 probe
  readout beside behavioral valence, so the readout-dependence claim is checkable from a float in
  both builds. Values from `results/stage_f/cross_patching_hf_{0-12,13-17,18-28}.json`
  (`recovery/*/probe`, `probe_ci95`). The two probe figures already in prose reproduce exactly
  ($74.9\%$ $[71.3,78.9]$, $87.4\%$ $[85.9,88.8]$), which validates the extraction. Layers 18--28
  are marked `---`: that file carries `probe_valid: false` because the probe tap is at/after the
  patched band, matching the reason already given in `app:patching`.
  Both builds now cite the table at the point of claim.

- **F3a rider, not in the audit** — L247 read "the layer-18 probe agrees in the ranges where it is
  valid", which directly contradicts the next paragraph's finding that the two readouts disagree in
  the mid band, where both *are* valid. Corrected to "shown where valid, and agrees early but not
  mid-band", which the new table now shows. Flagged because adding the probe panel would otherwise
  have made an existing contradiction visible without fixing it.

- **F3b** (MAJOR/WRITING) — new `tab:resolution` in `app:behavior`, cited from both builds. Reports
  the \qwen{} resolution sweep that L295 relies on to rule out an image-token-count explanation:
  resolution, image tokens, image AUC, and the drop/rise effects, from
  `conflict_qwen3-vl-8b-instruct_px{448,896,1344}_metrics.json`. The caption states that 896 and
  1344 px both expand to 262 image tokens, so three resolutions span two token budgets.

### Deliberately not applied

- **The resolution sweep's `asymmetry_index` was not used**, even though it is the obvious one-column
  summary. In the base run (`conflict_qwen_metrics.json`) that field is $0.394$, while the paper
  reports the varied-set mirror contrast as $+0.409$ — the same aggregation-order difference F2
  documents (`|mean|` against `mean|·|`). Tabulating `asymmetry_index` under the paper's own term
  "mirror contrast" would have imported a second estimator under one name, which is the exact error
  F2 exists to fix. The table therefore reports drop and rise, which reproduce the paper's own
  $-1.156$ / $+0.762$ for the base run and are directly comparable.
  **See Notes — this has a consequence beyond F3.**

- **F1, F4–F14** — out of scope for this round by the user's instruction. F1 (CRITICAL, short-build
  abstract drops "on positive images") remains the highest-priority open finding.

### Length

VLM4RWD 8pp → 8pp (References p9). Interp4Discovery 5pp → 5pp (References p6). Both still at the cap.

The first attempt went to **9pp**. The appendix tables are free — both venues exclude appendices —
but roughly 130 characters of added main-text prose tipped a page boundary. Recovered by tightening
the two sentences the edit had already touched (L247 −50 chars, L251 −15), leaving main text
slightly *shorter* than before this round. No evidence, hedge, or claim was cut to buy the space.

### Notes for the next round

- **The F2 aggregation ambiguity is systemic, not local to `tab:pairs`.** The mirror contrast has
  the same two forms: $+0.409$ (per-image, what the paper reports and labels "per-image" at L184)
  against $0.394$ (`asymmetry_index`, aggregate). The paper is correct wherever it says "per-image",
  but it does not say so everywhere — L195's matched-set $+0.496$, `tab:minimal`, and `tab:models`
  all print mirror contrasts without naming the aggregation. Worth one sweep to confirm which
  estimator each printed value came from, and to state it once in `\S`Readouts rather than per site.
  This is a candidate finding for the next critique round, not something this round changed.
- The builds are at genuine zero slack: any main-text addition needs a matching cut in the same
  edit. Verify with `./scripts/build-paper.sh` before considering an edit done.
- `sec:arbitration` is still multiply defined in the short build (F7) — the only LaTeX warning
  either build emits. Untouched this round.
- 156 tests pass, including the `method_diagram.numbers.json` guard; no figure regeneration needed,
  as no number the figure prints changed.

---

## Round 2 — 2026-09-01

Scope: **F4, F5, F6.** Starting state: tip `adc0180` plus Round 1's uncommitted edits; no
Overleaf-side drift.

### Applied

- **F4** (MAJOR/NARRATIVE) — `sec:mechanism` opening (long build). The section conceded that \gemma{}
  does not carry the headline contrast and then stopped, leaving no reason for the section to exist.
  The opening sentence now does both jobs at once: "both on \gemma{}: it carries the text-trained
  probe, and is therefore the only model here where an internal readout can be set against the
  behavioral one (\S\ref{sec:crosspatch})."
  Rationale: merging the motivation into the existing sentence rather than adding one costs 108
  characters instead of ~250, which matters at zero slack, and it removes the redundancy between
  "because the text-trained probe was fit on its states" and the new clause. The concession that
  follows is untouched — it is the honest part and it stays.

- **F5** (MAJOR/CLAIM) — `sec:related` L122 and `app:related`. Replaced the negative claim about the
  cited works ("none of this work asks whether matched positive and negative text pull with equal
  force") with a positive statement of our own scope ("our question is narrower: whether matched
  positive and negative text pull with equal force"). The appendix restatement moved from "but do
  not isolate whether..." to "we did not find the matched positive-versus-negative comparison
  isolated in them."
  Rationale: **this is a scoping change made because the evidence for the original claim was not
  available to me.** Verifying "none of this work asks X" requires reading
  `deng2025blindfaith`, `mixedsignals2025`, `contextvqa2026`, `signpost2026` and `camel2025`; I did
  not, and the handoff records the claim having been retired twice already. Rather than assert
  something unverified or soften it into vagueness, the sentence now states what this paper does and
  lets the contrast be inferred. The novelty positioning survives intact: both directions, matched
  event, four model designs. The short build already used this register at L95.
  **If you do verify those five papers, the stronger form is available again** — the change is
  reversible and is a scoping decision, not a finding that the original was false.

- **F6** (MAJOR/WRITING) — the "bounded valence" / "behavioral valence" collision, fixed by linking
  the names rather than merging them. First main-text use in each branch now reads "behavioral
  valence (the bounded score of \S\ref{sec:setup})" (L221 short, L247 long), and the definition in
  `app:contexts` states the identity and the reason for two names.

### Deliberately not applied

- **F6 as a global rename.** The audit's first option was to use one term throughout. I did not,
  because the two names are not sloppiness — they encode two different contrasts of one quantity.
  "Bounded" is the readouts contrast (against unbounded log-odds and the categorical override gap);
  "behavioral" is the mechanism contrast (against the layer-18 probe). Renaming would produce
  "the layer-18 probe gives image $74.9\%$... bounded valence gives image $96.0\%$", which loses the
  point that one readout is internal and the other is taken from the output — in the paragraph whose
  entire subject is where the readout is taken. The audit's second option (state the identity at
  first use) is both cheaper and better here, and is what was applied.

### Length

VLM4RWD 8pp → 8pp (References p9). Interp4Discovery 5pp → 5pp (References p6). Both still at the cap.

Net +343 characters, of which +144 landed in long main text and +41 in short main text; the rest is
appendix and therefore free at both venues. This fit without a compensating cut because Round 1
finished with main text marginally shorter than it started. That headroom is now spent — treat the
next main-text addition as needing its own offset.

### Notes for the next round

- Open findings, in priority order: **F1** (CRITICAL — short-build abstract drops "on positive
  images"; the pattern reverses on negative images, so this is a genuine overclaim in the most-read
  sentence), then **F7** (duplicate `sec:arbitration` label, the only LaTeX warning either build
  emits), then the MINORs F8–F14.
- The systemic aggregation issue logged in Round 1 is still open and is the strongest candidate for a
  fresh finding: mirror contrasts are printed in `tab:minimal`, `tab:models` and at L195 without
  naming which estimator produced them.
- F5's rescope is the one change this round that a reviewer could read as a retreat. If the novelty
  claim matters to the framing, verifying the five cited papers is the way to earn the stronger
  sentence back.
- 156 tests pass. No warnings introduced; the `sec:arbitration` warning predates this work.

---

## Round 3 — 2026-09-01

Scope: the remaining VLM4RWD findings (F8–F14), plus the "why six pairs?" gap from the handoff.

### Applied

- **F8** — `Table~\ref{tab:minimal}` now cited from `sec:minimal`; it was the only float in the long
  build no text referenced.
- **F9** — "sixteen further layers" → "16", both builds. It was the only spelled number above ten.
- **F10** — bare model names replaced with the macros where the referent was ambiguous: `\qwen{}` at
  L184 and L186, and "not for LLaVA" → "not for the LLaVA models" at L287, which was genuinely
  ambiguous between LLaVA-1.5-7B and LLaVA-NeXT-7B. Bare "Gemma layer 18" was left alone: there is
  only one Gemma in the paper, so it reads as shorthand rather than a second name.
- **F11** — the balanced-subset categorical reversal now prints as $-71.4\%$ instead of $-0.714$.
  Unit sourced, not assumed: `analyze_label_balance.py` builds that column from
  `flip_override(...)["dominance_gap"]`, and the base run's `dominance_gap` is $0.3940$, which the
  paper already prints as "$+39.4\%$". The value is unchanged; only its rendering.
- **F12** — "changes the headline contrast by less than $0.01$" → "the matched-set mirror contrast",
  which is the statistic the appendix actually reports for that check.
- **F13** — "making it the cleanest behavioral test" → "so an asymmetry it shows cannot be explained
  by its under-using the image", porting the reasoning the short build already gives.
- **F14** — the split-label sentence recast with complete-label scoring as the subject, matching the
  main text, so it no longer reads backwards on first pass.
- **"Why six pairs?"** (handoff item 6, not an audit finding) — answered on the author's stated
  reason: six is the number of swaps that hold frame, event and token count fixed under both
  tokenizers, not a sampling target. One clause in `sec:minimal`, the full statement in
  `app:contexts` including what was discarded and why the crossed intervals are wide as a result.

### Deliberately not applied

- **The mirror-contrast aggregation sweep** (logged as open since Round 1). **Not applied because the
  printed values could not be traced to a source file — see Notes. This needs your attention before
  submission and is the most important open item in this log.**

### Length

VLM4RWD 8pp → 8pp (References p9). Interp4Discovery 5pp → 5pp (References p6).
Net +556 characters, +158 of it in long main text. It fit without a compensating trim, but the cap
is now genuinely tight; assume the next main-text addition needs an offset.

### Notes for the next round

**Traceability gap on `tab:models` — unresolved, and worth checking before submitting.**
While sourcing the estimator for the aggregation sweep I could not find the table's mirror contrasts
in `results/`. Specifics:

- `tab:models` prints \qwen{} mirror contrast $+0.409$, photo-clustered $[+0.23,+0.59]$, crossed
  $[-0.16,+0.91]$; and \gemma{} $+0.405$, $[+0.25,+0.55]$, $[+0.06,+0.77]$.
- The only file computing that quantity with crossed intervals,
  `results/stage_f/unbounded_crossed.json` (from `analyze_stage_f_unbounded.py::mirror_contrast`),
  gives \qwen{} $0.39394$, $[0.2493,0.5416]$, $[-0.187,0.8907]$ and \gemma{} $0.26492$,
  $[0.0946,0.4314]$ — no value matches, and \gemma{} is far off.
- That file reports `n_images = [75, 75]`, against the paper's 121 photographs, so it looks like a
  **superseded run left in place**, not a contradiction of the table. `results/` is gitignored and
  regenerated, and the handoff records exactly this hazard for other artifacts.
- No file in `results/` contains $0.409$ as a stage-F mirror contrast.

I did not change any number, and I am not claiming the table is wrong — only that I could not verify
it and that the nearest candidate file disagrees. The right next step is a `tae-verify-claims` pass
over `tab:models` and `tab:minimal` specifically, and regenerating `unbounded_crossed.json` on the
121-image set so the stale file stops shadowing the current one. Until that is done, the aggregation
question (which estimator each printed mirror contrast came from) cannot be settled either, because
the two questions have the same answer.

Remaining audit findings after this round: **F1** (CRITICAL, short build only — the abstract drops
"on positive images") and **F7** (short build only — duplicate `sec:arbitration` label). Both are
Interp4Discovery-side; the VLM4RWD build has no open audit findings.

156 tests pass. No new LaTeX warnings.

---

## Round 4 — 2026-09-01 — verification, not revision

No paper text changed in this round. Two artifacts and one script did.

### `tab:models` is verified clean

Re-ran `analyze_stage_f_unbounded.py` against the four complete-label parquets. **All 24 values
reproduce exactly** — every point estimate, every photo-clustered interval, every crossed interval,
for all four models on all three readouts. The Round 3 concern is closed: it was a stale artifact,
not a numbers problem.

Root cause: the script's own defaults were `conflict_pilot.parquet` (Gemma) and
`conflict_llava.parquet` (legacy first-token LLaVA), and it had no `--llavanext` flag, so running it
bare regenerated the pre-retraction analysis over the top of the current one.

- **Fixed** `src/experiments/analyze_stage_f_unbounded.py`: defaults now point at the complete-label
  parquets, `--llavanext` added, models ordered as the paper's table columns, and a comment plus a
  `scoring_note` in the output record why these specific parquets and not the others. Running it
  bare now reproduces Table 7.
- `results/stage_f/unbounded_crossed.PRE-RETRACTION-1427a3c.json` preserves the old analysis, which
  Appendix K's provenance discussion refers to.

### The Drive sync regressed `results/stage_f/` — paper unaffected

`tests/test_method_diagram_numbers.py` began failing after the sync. **This is a true positive and
the guard working as designed.** Two files came back as their pre-unique-image-bootstrap versions
(60 annotation rows, duplicates double-weighted) rather than the post-fix versions behind the paper:

| `tab:mechmodels` | Paper | Synced file | Recomputed from parquet (51 unique) |
|---|---|---|---|
| Qwen question / turn / all-text | 12.1 / 6.2 / 62.3 | 11.8 / 5.7 / **65.4** | **12.11 / 6.16 / 62.26** |
| LLaVA question / turn / all-text | 32.7 / 20.5 / 66.0 | 34.3 / 22.4 / **67.7** | **32.75 / 20.47 / 66.03** |

Recomputing from `patching_qwen.parquet` and `patching_llava_sequence.parquet` with
`collapse_duplicate_image_rows` reproduces the paper on all six values. **The paper is right; the
synced metrics JSONs are stale.** This is exactly the regression commits `30965ca` / `f7035de`
existed to fix, reintroduced by the sync.

Affected: `patching_qwen_metrics.json` (run `20260810T224932Z`), `patching_llava_sequence_metrics.json`
(run `20260827T235707Z`). The parquets are intact, so nothing is unrecoverable. Not fixed here
because regenerating those JSONs is the runners' job; the values above are the correct contents.

**Operational warning: Google Drive preserves original modification times on download**, so a sync
that regresses a file leaves no recent timestamp. `find -newermt` will not detect it — only the test
suite did. Re-run `pytest` after every sync, and treat a `test_method_diagram_numbers` failure as
"the results directory moved", not "the figure is wrong".

### Verification status of the paper's tables, as of this round

| Table | Status |
|---|---|
| `tab:models` | verified exactly, all 24 values |
| `tab:mechmodels` | verified exactly from parquets, all six values |
| `tab:patching`, `tab:crosspatch` | verified against `patching_intervals.json`, `cross_patching_hf_*.json` |
| `tab:pairs` | verified against `conflict_..._minimal_analysis.json` |
| `tab:factorial` | verified against `conflict_qwen_metrics.json` |
| `tab:minimal` | **not yet traced** — the remaining gap |

### Notes

- Test suite is 154 passed / 2 failed, and the 2 failures are the sync regression above, not a code
  or paper defect. They will pass again once the two metrics JSONs are regenerated.
- `tab:minimal` is now the only table not traced to a source. Worth closing before upload.

---

## Round 5 — 2026-09-01 — `tab:minimal` traced

No paper text changed. Every **point estimate** in `tab:minimal` reproduces exactly from the
parquets; all but two intervals do as well.

| Cell | Paper | Recomputed |
|---|---|---|
| Varied, uncorrected gap | $+39.4\%$, $[+29.5,+49.0]$, crossed $[+6.4,+66.5]$ | $+39.40\%$, $[+29.5,+49.0]$, crossed $[+6.4,+66.5]$ — exact |
| Varied, corrected gap | $+21.7\%$, $[+10.9,+31.9]$ | $+21.71\%$, $[+10.8,+32.2]$ — estimate exact, CI ~0.3pp off |
| Varied, mirror contrast | $+0.409$, $[+0.23,+0.59]$, crossed $[-0.16,+0.91]$ | $+0.4089$, $[+0.230,+0.588]$, crossed $[-0.157,+0.910]$ — exact |
| Varied, $|\betatxt|/|\betaimg|$ | $1.14$ | $1.1440$ — exact |
| Matched, uncorrected gap | $+40.8\%$, $[+31.3,+49.0]$ | $+40.77\%$, $[+31.3,+49.0]$ — exact |
| Matched, corrected gap | $+23.1\%$, $[+12.6,+33.1]$ | $+23.08\%$, $[+12.1,+33.7]$ — estimate exact, CI ~0.5pp off |
| Matched, mirror contrast | $+0.496$, $[+0.31,+0.67]$, crossed $[+0.11,+0.83]$ | $+0.4964$, $[+0.314,+0.673]$, crossed $[+0.114,+0.825]$ — exact |
| Matched, $|\betatxt|/|\betaimg|$ | $1.97$ | $1.9677$ — exact |

Sources: `conflict_qwen.parquet` (varied), `conflict_qwen3-vl-8b-instruct_minimal.parquet` (matched),
via `analyze_stage_f_unbounded.mirror_contrast`, `analyze_stage_f._dominance`, and a reconstruction of
the corrected override gap. The matched $\betatxt/\betaimg$ of $1.97$ also appears directly in
`conflict_qwen3-vl-8b-instruct_minimal_analysis.json` (`dominance.valence.dominance_ratio = 1.96775`).

The 19.4% neutral-context error rate quoted at \S\ref{sec:minimal} reproduces exactly as the mean
per-image neutral-context negative-argmax rate on positive images.

### The one real gap: the corrected override gap has no implementation in the repo

`grep -rn "corrected" src/` finds only `uncorrected_override_gap` in `analyze_label_balance.py`.
`reporting.py::flip_override` computes the **uncorrected** gap only. Nothing in the tracked source
subtracts the per-image neutral-context error rate.

I reconstructed it from the paper's own description ("corrected against that image's neutral-context
error rate") and it reproduces **both** point estimates to two decimals — $+21.71\%$ and $+23.08\%$
against the published $+21.7\%$ and $+23.1\%$ — which is strong evidence the definition in the paper
is the definition that was used. The two photo-clustered CIs land within ~0.5pp, the residual being
that the exact bootstrap treatment of the neutral correction (whether the neutral rates are
resampled along with the conflict rates) is not recoverable from the paper's text.

**Why this matters.** The corrected gap is the paper's primary categorical readout — "We rely on the
corrected measure" (\S\ref{sec:conflict}) — and it appears in `tab:minimal`, in all four models of
`tab:models`, in the main-text $57\%$/$35\%$ figures, and in the Discussion. It is the one headline
quantity with no code path behind it, which bears directly on checklist item 5 (open access to code)
already standing at `\answerNo`.

Not fixed here: adding an implementation would be a code change whose CIs might not match the
published ones to the last decimal, which is the author's call, not a revision-pass decision.

### Verification status — complete

Every table in the paper is now traced: `tab:models`, `tab:mechmodels`, `tab:patching`,
`tab:crosspatch`, `tab:pairs`, `tab:factorial`, `tab:minimal`. No number was found to be wrong.

---

## Round 6 — 2026-09-01 — asset citations (checklist 9.2) and licences (NeurIPS checklist item 12)

### Applied

- **9.2 — the four models are now cited.** Before this round the paper named Gemma-3-4B, Qwen3-VL-8B,
  LLaVA-1.5-7B and LLaVA-NeXT-7B throughout and cited none of them; nor SigLIP, CLIP, HuggingFace
  `transformers`, or TransformerLens. Eight entries added to `references.bib` and cited at first use:
  the four models plus `transformers` in \S\ref{sec:setup}, SigLIP and CLIP at the architecture
  descriptions in \S\ref{sec:models}, and `transformers` + TransformerLens in `app:compute`.
  All 61 citations resolve; no undefined references.

- **5.5 — reproducibility detail.** `app:compute` now names the inference stack (HuggingFace
  `transformers`, plus TransformerLens for the \gemma{} probe and steering work), states that ridge
  probes use the scikit-learn solver with the penalty selected on validation, and repeats that all
  runs use a single seed. Appendix space, so free at both venues.

- **NeurIPS checklist item 12 flipped to `\answerYes{}`** (`\answerNo` count 4 → 3), backed by a new
  "Licences and terms of the assets we use" paragraph in `app:impacts`.

### Licences, as verified

**Every licence below was read from the model card or the dataset's own distribution page.** None was
taken from an LLM summary — the handoff records a previous list that had LLaVA-1.5 as Apache 2.0.

| Asset | Licence / terms | Source read |
|---|---|---|
| Gemma-3-4B-it | **Gemma Terms of Use** (`license: gemma`) — not OSI-approved | HF model card |
| Qwen3-VL-8B-Instruct | **Apache 2.0** | HF model card |
| LLaVA-1.5-7B | **LLaMA 2 Community License** (`license: llama2`) — a Llama-2 derivative, **not** Apache 2.0 | HF model card |
| LLaVA-NeXT (v1.6-mistral-7b) | **Apache 2.0** (Mistral-7B-Instruct-v0.2 base) | HF model card |
| EMOTIC | **Non-commercial research and education only**, via request form; images originate in MSCOCO and ADE20K | official download page |
| crowd-enVENT | **No explicit data licence stated.** The authors' modelling repository is MIT, which we report as covering the code, not the corpus | project data page + repo |

The LLaVA-1.5 result is the one that matters: the handoff's warning was correct, and the model card
states plainly "Llama 2 is licensed under the LLAMA 2 Community License". Reporting it as Apache 2.0
would have been a licence misstatement in a published paper.

crowd-enVENT is reported as unlicensed-as-stated rather than assuming the repo's MIT licence covers
the data. That is the honest reading and it is what the appendix and the checklist now say.

### Citations verified (checklist 10.1–10.3)

All eight new references were checked against primary sources on 2026-09-01, not accepted from
search summaries: Gemma 3 Technical Report (arXiv 2503.19786), Qwen3-VL Technical Report
(arXiv 2511.21631, Bai et al., 64 authors — note this is **not** the Qwen3 report 2505.09388 that the
model card's BibTeX points at), LLaVA-1.5 "Improved Baselines with Visual Instruction Tuning"
(arXiv 2310.03744, CVPR 2024), LLaVA-NeXT blog (Liu et al., Jan 2024), SigLIP (arXiv 2303.15343,
ICCV 2023), CLIP (arXiv 2103.00020, ICML 2021), `transformers` (ACL Anthology 2020.emnlp-demos.6,
pp. 38--45), TransformerLens (Nanda and Bloom 2022).

### Length

VLM4RWD 8pp → 8pp (References p9). Interp4Discovery 5pp → 5pp (References p6).
Net +1396 characters, of which only +110 is main text (the citation keys at first use); the licence
paragraph and the compute-appendix additions are appendix space, free at both venues.

### Notes

- Checklist `\answerNo` now stands at **3**: item 5 (open access to code), item 8 (compute
  wall-clock), item 13 (new assets). Item 8 is still held by the live `\todo` in `app:compute`.
- Test suite unchanged at 154 passed / 2 failed — still the Round 4 Drive-sync regression, not
  affected by this round.

---

## Round 7 — 2026-09-01 — figure formats (checklist 7 bonus)

### Correction to the checklist assessment

I previously flagged the six PNG figures as a hard fail on "no low-resolution images", reading the
DPI **metadata** field (130) from each file. That field is matplotlib's figure DPI, not the effective
print resolution. Measured at their actual inclusion widths, the six render at **309–602 DPI**, all
above the 300 DPI print standard. The item was never a blocker. What remained legitimately worth
fixing is raster-vs-vector for line plots.

### Applied

- **All six plotting call sites now emit a PDF twin** alongside the PNG
  (`analyze_stage_f.py`, `analyze_stage_a.py`, `stage_a_steering_v2.py`, `stage_c_transfer.py`,
  `stage_c_caption.py`). Future regenerations produce vector automatically, including for the three
  figures that cannot be regenerated without a GPU.
- **Three figures converted to vector and swapped into the paper**: `conflict_qwen`,
  `conflict_llava_sequence`, `stage_a_localization`. These come from `analyze_stage_f.py` and
  `analyze_stage_a.py`, which are analysis-only — no model load, no GPU.
- **Each swap was gated on a pixel-identity check.** Before swapping, each figure was regenerated as
  PNG and compared against the published PNG: all three came back **pixel-identical** (max channel
  difference 0 across 1820x1040 / 832x520 images). That proves the underlying data has not moved and
  the vector twin is a faithful replacement, not a silently different plot.

Verified in the built PDF with `pdfimages -list`: the document now contains **three raster images
total**, all in the appendix, at 440, 482 and 309 ppi. The four vector figures contribute none.

### Deliberately not applied

- **`stage_a_steering_v2`, `stage_c_readout`, `stage_c_caption_baseline` stay PNG.** Their `savefig`
  calls live inside experiment runners that load a model (`stage_a_steering_v2.py`,
  `stage_c_transfer.py`, `stage_c_caption.py` — the last of which also generates captions for 1,000
  images). Converting them means re-running those experiments on a GPU, which is not a formatting
  change and is not something to do days before a submission. They render at 309–482 DPI, which is
  fine. The code is now ready to emit PDF whenever those runs next happen.
- **No figure was regenerated without a pixel check first.** Given that the Drive sync regressed two
  patching artifacts (Round 4), regenerating figures from `results/` is exactly the operation that
  could silently change published content. The identity check is what made this safe; apply the same
  gate to the remaining three when they are eventually re-run.

### Length

VLM4RWD 8pp → 8pp (References p9). Interp4Discovery 5pp → 5pp (References p6). No warnings beyond the
pre-existing `h`-float note. Tests unchanged at 154 passed / 2 failed (the Round 4 sync regression).

---

## Round 8 — 2026-09-01 — consecutive floats (checklist 8.5)

### Applied

**Nine consecutive-float sites → zero.** Every place where two tables or figures sat back to back
with no prose between them now carries a lead-in that states what the next float shows and why it is
there. All nine are in the appendix, so this cost no main-text space at either venue.

| Between | Added |
|---|---|
| `tab:factorial` → `tab:pairs` | the per-pair breakout, so the six swaps read directly rather than through their mean |
| `tab:pairs` → `fig:conflict` | names the two models the main text contrasts most sharply |
| → `tab:patching` | opens the mechanism block and ties it to the $88$--$93\%$ claim |
| `tab:patching` → `tab:mechmodels` | states the shared readout and the wider layer bands, so the columns bound a pattern rather than forming a matched comparison |
| `tab:mechmodels` → `tab:crosspatch` | flags that the intervention reverses, and that this is where the readouts disagree |
| → `tab:models` | the four-model comparison with both interval types |
| `tab:models` → `tab:resolution` | names it as the token-count control for \S\ref{sec:models} |
| `tab:stagea` → `tab:steertext` | ties the text-only sweep to the cross-modal sweep it is compared against |
| `tab:steertext` → `fig:stagea` | the existing lead-in sentence **moved** from after the figure to before it, rather than duplicated |

The last one is worth noting: `fig:stagea` already had an introducing sentence, but it sat *after*
the figure, which both left the float unexplained on approach and stranded a one-line orphan
paragraph. Moving it fixes 8.5 and 8.4 in the same edit.

All nine lead-ins verified present in the built PDF.

### Not applied

- **8.1 bad boxes: 4 underfull, 0 overfull.** Chasing the remaining underfull boxes is not worth it
  and I recommend against it before submission. **Zero overfull** means no text intrudes into a
  margin, which is the failure a reviewer actually sees; underfull boxes are loose inter-word
  spacing. With both builds at exactly their page caps, nudging paragraphs to fix cosmetic spacing
  risks reflowing a page boundary and costing a page. One of the three original underfull hboxes
  (the Models paragraph) was incidentally fixed by Round 6's citation additions; one new one and one
  `\vbox` appeared from the float reflow. Net cosmetic.

### Length

VLM4RWD 8pp → 8pp (References p9). Interp4Discovery 5pp → 5pp (References p6).
Net +1381 characters, all appendix. Tests unchanged at 154 passed / 2 failed (Round 4 sync regression).

### Checklist status after this round

Every item I can act on is now closed or reclassified. Remaining, all requiring the author:
regenerating the two patching JSONs the Drive sync clobbered; compute wall-clock totals (checklist
item 8, held by the live `\todo`); the Overleaf upload; and the start-to-finish external read (11.5).
Open audit findings F1 and F7 are Interp4Discovery-side only.

---

## Round 9 — 2026-09-01 — Overleaf bundles rebuilt

`./scripts/build-overleaf.sh` run after Rounds 1–8. Both projects regenerated from the toggled
source and **verified identical to it by extracted-text diff**, not just page count, so a
wrongly-resolved `\ifshort` branch would have failed loudly. Zips rewritten only after both passed.

- `overleaf/vlm4rwd.zip` — 12 files, 434 KB (was 637 KB on Aug 29; smaller because three raster
  figures became vector PDFs). `references.bib` now carries 61 entries, all eight new asset
  citations present. `main.tex` fully flattened with the venue branch resolved (`ifshort` count 0);
  the only remaining `\input` is `checklist.tex`, which ships as a separate file by design.
- `overleaf/interp4discovery.zip` — same treatment, verified identical to the short build.

Spot-checked in the bundle: the licence paragraph, the `tab:pairs` estimator note (`average to
$+1.172$ rather than $+1.148$`), the barred $\overline{\Delta}$ headers, the layer-18 probe panel,
`tab:resolution`, the six-pairs justification, the rescoped novelty sentence, and the mechanism-table
lead-ins. The stale `paper/overleaf.stale-snapshot.tex.bak` is not swept in.

**The bundles are current as of Round 8 and ready to upload.** The open question remains which
Overleaf project receives them — replacing in place will likely drop Sneheel's 13 comment threads.
