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

---

## Round 10 — 2026-09-01 — F1 and F7 closed; the audit has no open findings at either venue

Both remaining findings were Interp4Discovery-side (`\ifshort`). Both are now applied, so all 14
findings in `paper/audit.md` are either applied or deliberately declined with a logged reason.

### F1 (CRITICAL) — the short abstract now carries the scope restriction

`neurips_2026.tex` L66. The short abstract stated the four-to-five-times result with no restriction,
while the pattern reverses on the excluded half (`tab:factorial`: on negative images a *positive*
context moves valence $+0.762$ against a negative context's $-0.017$). The within-item contrast is
only ever computed on the 62 positive images. Applied the audit's exact wording:

> …moves \qwen{}'s judgment **on positive images** four to five times farther when the word is
> negative, on all six tested pairs.

Verified in the rendered short PDF, not only the source. +19 characters; the short build still
measures 5pp main text, references p6.

**Not changed, and deliberately:** L111, the long build's first contribution bullet, states the
result without the scope too. It is a genuine instance but a much weaker one — the long abstract
(L70), the intro paragraph directly above it (L105) and the conclusion (L344) all carry the scope,
so the bullet is read in a scoped context. At zero page slack in the 8pp build, and with VLM4RWD
already submitted, the edit was not worth the risk. Revisit if the long build is ever reopened.

### F7 (MAJOR) — `sec:arbitration` is no longer multiply defined

The duplicate was **deleted**, not renamed. L206–L208 were a cluster of three compatibility labels
in the short branch, but only `sec:arbitration` actually collided, and the asymmetry is the reason:

| label | short-branch def | twin | collides? |
|---|---|---|---|
| `sec:patching` | L206 | L234, long branch | no — mutually exclusive branches |
| `sec:crosspatch` | L207 | L245, long branch | no — mutually exclusive branches |
| `sec:arbitration` | L208 | **L634, unconditional appendix** | **yes, in the short build only** |

L208 had been added by pattern-matching the other two. Its twin is in the appendix rather than the
long main text, so both definitions compile in the short build. All three `\ref`s to it — L255 (long
main text), L522 and L529 (unconditional appendix) — want the appendix subsection, and nothing in
the short build wants a short main-text target, so the label had no referent to lose.

Confirmed from the compile log rather than by inspection: the short build now emits **no**
`multiply defined`, **no** undefined reference or citation, and **no** overfull box. The remaining
warnings are 4 underfull hboxes, 3 underfull vboxes, a bibliography underfull, and a `lineno.sty`
UTF-8 note — all pre-existing and all previously judged not worth chasing at zero page slack.
The rendered reference resolves to §D.4, "Does the text-derived direction still move the answer
under conflict?" — the intended target, now by design rather than by ordering accident.

### Page counts and tests after this round

`./scripts/build-paper.sh`: VLM4RWD 8pp (references p9) OK, Interp4Discovery 5pp (references p6) OK.

**Tests are now 156 passed / 0 failed.** The two Round 4 failures are resolved: both clobbered
patching metrics JSONs were regenerated on CPU from the intact parquets with the runners' existing
`--reanalyze` flags —

```
python -m src.experiments.stage_f_qwen_patching   --reanalyze
python -m src.experiments.stage_f_llava_patching  --reanalyze
```

— reproducing Round 4's recomputed column exactly (Qwen 12.11 / 6.16 / 62.26; LLaVA 32.75 / 20.47 /
66.03), with `n_unique_images: 51` from 60 rows restored in both. Round 4 recorded this as "the
runners' job"; it is not, because the parquets survived the sync and the reanalysis path is CPU-only.
`results/` is gitignored, so this fix does not ride in a commit and must be re-run after any sync
that regresses those files.

### Still open, and none of it is an audit finding

The Overleaf bundles predate this round and no longer match the source — **`interp4discovery.zip`
must be rebuilt before submission**. Also outstanding: the compute wall-clock `\todo` in
`app:compute` (checklist item 8), the corrected override gap implementation (checklist item 5), the
`emomm2026` venue confirmation, and the start-to-finish external read (checklist 11.5).

---

## Round 11 — 2026-09-02 — the corrected override gap now has an implementation

Closes the one gap Round 5 left open. No paper text changed and no published number moved; this adds
a code path behind a number that already existed.

Round 5 found that the paper's primary categorical readout — "We rely on the corrected measure"
(\S`sec:conflict`), carried by `tab:minimal`, all four models of `tab:models`, the main-text 57%/35%
figures and the Discussion — had no implementation anywhere in `src/`. The only related code was
`flip_override` in `src/experiments/shared/reporting.py`, which computes the **uncorrected** gap, and
an `uncorrected_override_gap` dict key in `analyze_label_balance.py`. Round 5 declined to fix it on
the grounds that the CIs might not match; that reasoning is now resolved below.

### What was added

`corrected_override_gap(df, n_boot=2000, seed=0)` in `src/experiments/shared/reporting.py`, beside
`flip_override`. Two helpers (`_argmax_category_frame`, `_per_image_category_rate`) were factored out
of `flip_override` and are shared by both; `flip_override`'s public return contract is unchanged and
its values are bit-identical.

The correction subtracts **each image's own neutral-context error rate**, paired by `image_path`
(inner join, so an image missing either member is dropped from that arm) rather than differencing
group means:

    corrected_gap = mean_pos_images(P(neg argmax | neg ctx) − P(neg argmax | neutral))
                  − mean_neg_images(P(pos argmax | pos ctx) − P(pos argmax | neutral))

Clustered bootstrap over photographs, resampling the paired per-image differences.

### It reproduces the published estimates exactly

| Source parquet | Uncorrected | Corrected | Paper |
|---|---|---|---|
| `conflict_qwen.parquet` (varied) | $+39.40\%$ | $+21.71\%$ | $+21.7\%$ |
| `conflict_qwen3-vl-8b-instruct_minimal.parquet` (matched) | $+40.77\%$ | $+23.08\%$ | $+23.1\%$ |

The positive-image neutral-context negative-argmax rate is $19.35\%$, rounding to the $19.4\%$ the
paper quotes at \S`sec:minimal`. Both artifacts retain 62 positive and 60 negative photographs.

### The CI residual is real, expected, and deliberately not tuned away

| | Observed | Published |
|---|---|---|
| Varied | $[+10.8, +32.2]$ | $[+10.9, +31.9]$ |
| Matched | $[+12.1, +33.7]$ | $[+12.6, +33.1]$ |

0.1–0.6pp, the same residual Round 5 measured. The cause is unchanged and still not recoverable from
the paper's text: whether the neutral rates are resampled alongside the conflict rates. **The
bootstrap was left alone rather than fitted to close the gap** — matching a published interval by
adjusting the estimator until it agrees is exactly the move that would make this implementation
worthless as verification. The point estimates agreeing to two decimals is the evidence that the
definition in the paper is the definition that was used.

### Tests

`tests/test_experiment_reporting.py` gains three: a synthetic frame with hand-computed expectations
that also covers images missing a neutral row, and an artifact-backed regression parameterized over
both parquets, asserting the corrected estimates to two decimals and the 19.4% neutral rate. The
artifact tests `pytest.skip` when the parquet is absent, since `results/` is gitignored — the pattern
from `test_method_diagram_numbers.py`.

**Suite: 159 passed / 0 failed** (was 156).

---

## Round 12 — 2026-09-02 — short-build audit applied (F1, F2, F3), paid for with two cuts

A second audit pass, scoped to the `\ifshort` (Interp4Discovery) build, is recorded in
`paper/audit-short-build.md` — written alongside `paper/audit.md`, not over it, since that file's 14
findings are closed and REVISION-LOG references them throughout. Nine findings; the three MAJOR ones
are applied here. The short build had **zero page slack**, so every addition below is paid for.

### F1 — the abstract and intro promised three tools and adjudicated two

Both the short abstract (L66) and the short intro (L91) announced three tools — a frozen
text-trained probe, activation patching, and a difference-of-means intervention — then gave verdicts
on two. The third tool's verdict existed only in \S4: it moves behavioral valence with slope
$+0.335$ under conflict *but* does not show that either cue's weight changed. That is a **qualified**
verdict, which is the paper's own thesis arriving a third time, so burying it cost the paper its
best-fitting example and left a reviewer free to assume the third tool was dropped for failing.

Added to both, compactly: "A text-derived direction still moves the answer under conflict without
showing that either cue's weight changed" (abstract) and "The intervention still moves the answer
under conflict, but shows nothing about either cue's weight" (intro).

### F2 — the varied-set mirror contrast is not the difference of the cells the paper bolds

\S3 gives $-1.156$ and $+0.762$ and reports the mirror contrast as $+0.409$. Those cells differ by
$0.394$. `tab:factorial` renders in **both** builds, bolds exactly those two cells, and its caption
instructed the reader to compare them — an invitation to a subtraction that does not reproduce the
printed number. This is the same two-estimator issue the earlier audit's F2 fixed for `tab:pairs`
(whose caption already explains why its rows average to $+1.172$ against a $+1.148$ headline), one
table over and unexplained.

The caption now states it: computed per photograph then averaged, hence $+0.409$ rather than the
$+0.394$ from differencing cell means, with a pointer to the `tab:pairs` note.

**This is the one edit that touches the long build**, since `tab:factorial` is unconditional. VLM4RWD
is past its deadline, so the repo now differs from that submitted PDF by this appendix caption. It is
a clarification of an existing number, not a change to one, and it carries forward to camera-ready.

### F3 — three mechanism numbers had no table in the short build

\S5 quoted $62\%$, $66\%$ and $82\%$ from `tab:mechmodels`, which was gated `\ifshort\else` and never
rendered in the short build. No `\ref` pointed at it, so LaTeX raised nothing and the numbers were
simply unverifiable — in the paragraph carrying the cross-model mechanism claim, the one most exposed
to the paper's own thesis. The short branch now gets its own lead-in and `\input`, and \S5 cites it
as Table~6. The long branch is untouched; each build still inputs the table exactly once, and
`tables/patching` remains main-text-only in the short build and appendix-only in the long one.

### What paid for it

The additions pushed the short build to 6pp — five lines over. Rather than dilute the fixes, two
genuine redundancies were cut from the short main text:

1. **\S2's both-groups caveat.** The same fact appeared three times in the short build: \S2, \S6
   (Limitations), and `app:behavior`, the last with the exact $+0.496 \to +0.505$ figures. \S2's copy
   was the weakest and is gone — **from the short build only**. Because \S2 is unconditional, cutting
   it outright had silently altered the long build too; it is now wrapped `\ifshort\else` so VLM4RWD
   renders exactly as before. Verified in both PDFs.
2. **\S3's re-derivation of why Qwen3-VL-8B is the behavioral model.** \S2's Models paragraph already
   says so. Short branch only.

### Verification

- `./scripts/build-paper.sh`: VLM4RWD **8pp** (refs p9) OK, Interp4Discovery **5pp** (refs p6) OK.
- Short-build compile log: **zero** multiply-defined, undefined-reference and overfull warnings. The
  remaining warnings are the same pre-existing underfull boxes and the `lineno.sty` UTF-8 note.
- All five changes confirmed in the **rendered** short PDF, not just the source; the long PDF
  confirmed to retain the \S2 sentence, to lack the short-branch abstract wording, and to input
  `tab:mechmodels` exactly once.
- Tests **159 passed / 0 failed**.
- `./scripts/build-overleaf.sh`: both bundles rebuilt, both *identical to toggled build*;
  `interp4discovery/main.tex` carries all three fixes with zero `ifshort` leaks.

### Not applied

F4–F9 (abstract sentence density beyond the layer-range trim already taken, a sentence-initial
"And" in the abstract, a pair-1/pair-2 attribution, the compat-label stacks, the absent related-work
section, and one paragraph title) are MINOR and unapplied. At zero slack none of them is worth
displacing a MAJOR fix, and the "And" is the only one a reviewer is likely to notice.

---

## Round 13 — 2026-09-02 — short-build audit MINORs: F4, F5, F6, F7, F9 applied; F8 withdrawn

Closes `paper/audit-short-build.md` except F8. Short branch only this round — the long build's text
is byte-unchanged, verified against `paper-build/long.pdf`, not merely assumed from the diff.

- **F4** — the abstract's densest sentence is split after the probe clause. It ran ~60 words across
  two different experiments; it is now two sentences. Combined with the layer-range trim already
  taken in Round 12, the sentence carrying the evidence for both surviving claims is materially
  easier to read cold. Net $-3$ characters.
- **F5** — the abstract no longer opens a sentence with "And": *"The cross-model conclusion was
  likewise an artifact of a scoring choice…"*. This was the only such instance in the short main
  text and it sat in the most-read paragraph in the paper.
- **F6** — the super-additivity observation is no longer attributed to pair 2 alone. It holds for
  both: pair 1 gives $49.1 + 45.2 = 94.3 > 93.2$, pair 2 gives $54.8 + 38.2 = 93.0 > 87.9$. Since
  non-additivity is the stated reason for refusing to assign causal shares, attributing it to one
  pair understated the paper's own justification. Now: *"In both, the two estimates sum to more than
  the joint recovery."*
- **F7** — the two compatibility-label stacks (`sec:conflict`/`integration`/`asymmetry`/`minimal`,
  and `sec:patching`/`crosspatch`) now carry source comments recording why they exist and the
  invariant that keeps them safe: **a compat label may duplicate a long-branch definition, which is
  mutually exclusive, but never an unconditional one.** That is precisely the rule `sec:arbitration`
  violated until `ad482dd`. Comments only; nothing rendered changes.
- **F9** — `\paragraph{Nor does averaging fix it.}` became `\paragraph{Averaging does not fix it
  either.}`, declarative and parallel with the other titles in \S5.

### F8 withdrawn as overstated, not deferred

F8 argued that the short build, having no related-work section, leaves positioning to two
disclaimers. **On re-reading the paragraph while implementing it, the premise was wrong.** Its
*first* sentence already states the positive contribution — "We therefore present this as a study of
when localization evidence does and does not license a conclusion, not as a claim about
vision-language architecture" — and the paragraph closes by pointing at `app:related` for the full
comparison. The disclaimers are the middle of a positive-disclaimer-pointer structure, not the whole
of it.

A clause naming the novelty was drafted and built ("what is new is the disagreement between tools on
one controlled phenomenon"), then reverted: it restated the first sentence, and at $+77$ characters
it was the single most expensive of the six fixes in a build with no slack. Recording it as withdrawn
rather than outstanding, so a later round does not re-derive it.

### Page budget

The first pass at all six pushed the short build to 6pp — the same five-line spill as Round 12, the
conclusion paragraph moving wholesale. Rather than cut real content to buy space for the weakest
finding, F8 was dropped ($-77$) and F6 tightened ($+18 \to +1$). Net for the round is roughly
$+10$ characters and the build returns to 5pp. **No content was sacrificed to fit these MINORs**,
which was the correct trade: Round 12's cuts were justified by genuine triplication, and there was
no comparable redundancy left to spend here.

### Verification

`./scripts/build-paper.sh`: VLM4RWD **8pp** (refs p9) OK, Interp4Discovery **5pp** (refs p6) OK.
Short-build log: **zero** multiply-defined, undefined-reference and overfull warnings. F4/F5/F6/F9
each confirmed in the rendered short PDF; F7 confirmed *absent* from it, being comments. Long PDF
confirmed to carry neither the new abstract wording nor the \S5 title change. Tests **159 passed /
0 failed**. Both Overleaf bundles rebuilt, both *identical to toggled build*, `interp4discovery`
carrying all four rendered fixes with zero `ifshort` leaks.

**`paper/audit-short-build.md` is now fully dispositioned: F1–F3 applied in Round 12, F4–F7 and F9
applied here, F8 withdrawn with reason. Neither audit has an open finding at either venue.**

---

## Round 14 — 2026-09-02 — the short build's five orphaned targets, closed

`docs/cs-paper-checklist-short-build-2026-09-02.md` re-ran the 2026-08-27 CS-paper checklist against
the `\ifshort` build. Two items failed for a reason specific to the venue toggle, and this round
closes both.

### The failure mode is structural, and worth recording

Five targets were defined and never referenced **in the short build**: `tab:models`, `tab:minimal`,
`app:headroom`, `app:pilot`, `app:patching`. Every one of them *is* referenced in the long build.
The cause is not a missing label but a missing sentence: each pointing sentence sits inside an
`\ifshort\else` branch, so the short build inherits the float or appendix section while dropping the
prose that cites it. **A float can therefore go orphaned in one venue and not the other from a single
source file, and neither build warns**, because a label with no `\ref` is silent in LaTeX.

Measured before: long build 0 real orphans, short build 5. The long build's 8 orphans were all
`sec:` labels, which is the harmless class.

`tab:models` was the worst of the five — the four-model comparison, sitting in the short build's
**main text** with no textual pointer at all.

### What was added, and where it was paid for

At zero page slack, only references that must be in the main text were put there:

| target | new reference | cost |
|---|---|---|
| `tab:models` | \S5, "On the varied set (Table~\ref{tab:models})…" | **main text**, $+24$ chars |
| `tab:minimal` | new `app:behavior` lead-in in the short branch | appendix, free |
| `app:pilot` | same lead-in | appendix, free |
| `app:headroom` | same lead-in | appendix, free |
| `app:patching` | appended to the Round 12 `tab:mechmodels` lead-in | appendix, free |

`app:headroom` was first drafted as a \S5 parenthetical and **moved to the appendix after the build
went to 6pp**. Two attempts overflowed — $+53$ and then $+24{+}30$ characters — before the split
above landed at 5pp with no content cut. Recording the number because it is the operative constraint
on this build: **roughly 25 characters of \S5 is the entire remaining budget.**

### Verification

- Orphan audit re-run on both flattened builds: **short 42 labels, long 44; zero duplicate
  definitions, zero dangling references, and zero real orphans in either.** Both retain only `sec:`
  label orphans (short 5, long 8), which includes the two documented compat stacks.
- `./scripts/build-paper.sh`: VLM4RWD **8pp** (refs p9) OK, Interp4Discovery **5pp** (refs p6) OK.
- Short-build log: zero multiply-defined, undefined-reference and overfull warnings.
- All five references confirmed in the **rendered** short PDF (`tab:models` resolves to Table 2,
  `tab:minimal` to Table 3). The long PDF confirmed to carry **neither** short-branch lead-in.
- Tests **159 passed / 0 failed**. Both Overleaf bundles rebuilt, both *identical to toggled build*.

### Checklist items still failing, and why

`2.4` (no itemised contributions list in the short build), `2.6` (no figure in the main text at all),
`5.1` (two datasets), `10.1`–`10.3` (≈53 citations unverified) and `11.5` (no external start-to-finish
read) are unchanged. `8.1` remains ⚠️ — zero overfull boxes but five underfull sites, all in the
appendix, none in the five main pages. None of these is mechanically fixable and none blocks
submission.

---

## Round 15 — 2026-09-02 — citations verified; no fabricated reference found

Full record in `docs/citation-verification-2026-09-02.md`. This closes the item the CS-paper
checklist called "the highest-risk open item in the whole document" and "the one item no script can
close for you" — 10.1–10.3, roughly 53 of 61 entries unverified, with project notes recording that
several references came from AI-assisted literature search.

### Result

**No fabricated citation exists in `references.bib`.** Sixteen entries were checked against primary
sources and every one resolves to a real paper whose title and author list match the bib exactly.
All 61 entries carry a complete year, title, author and venue.

### The high-risk set was defined, then verified exhaustively rather than sampled

Fabrication hides where it cannot be checked from memory, so the set was every entry dated 2026,
every entry whose arXiv id postdates the assistant's May 2026 knowledge cutoff, and every entry with
**no arXiv id at all** (`mosear2025`, `fcct2026`) where nothing could be checked mechanically. All
16 were verified; none was skipped.

Two entries claimed venues their arXiv pages do not show — `zhang2026anydepth` (ICLR 2026) and
`camel2025` (EMNLP 2025). **Both were confirmed independently**, on the ICLR virtual site and in the
ACL Anthology. Neither was an error. `emomm2026`, `seeingoverrides2026`, `contextvqa2026`,
`veena2026`, `deng2025blindfaith`, `mosear2025` and `fcct2026` likewise had their venues confirmed.

### Two false alarms, recorded so they are not re-raised

- **`negbeforepos2026` and `steeringnonident2026` share a first author** (Sohan Venkatesh), are both
  2026 arXiv-only, and both support exactly what this paper needs — asymmetric valence processing
  and steering non-identifiability. That is the classic shape of a fabricated pair. **Both are real**,
  with exact title, author and date matches.
- **`zhang2026anydepth` looked out of place**, a safety-alignment paper cited in a valence-conflict
  study. It is not: its central finding is that alignment "is concentrated in the assistant header
  tokens", which is precisely the claim it is cited for at L128/L526. Citation *usage* was also
  spot-checked for `signpost2026` (geolocation) and `veena2026` (VEENA is genuinely that paper's
  method name). All three apt.

A third false alarm was mine: an initial field-completeness pass reported 25 entries missing a
`year`. That was a regex artifact — the entry-body capture dropped the trailing newline, so the last
field of every entry failed to match. Re-run correctly: **zero entries missing any required field.**

### One real gap, fixed

`camel2025` carried no `pages` and an abbreviated `booktitle`. Both corrected from the ACL Anthology
record (`2025.emnlp-main.1020`): pages **20166--20180** and the full proceedings title. Verified in
the rendered bibliography; both builds remain at cap and both Overleaf bundles were rebuilt.

Three page ranges remain unverified, all in entries whose venue is confirmed — `emomm2026`,
`seeingoverrides2026`, `fcct2026`. Page ranges are the least consequential field and the hardest to
confirm without the published volume.

### Residual risk, stated plainly

**37 entries were not individually verified**: the canonical and older works (EMOTIC, ROME, GPT-3,
CLIP, SigLIP, LLaVA-1.5, crowd-enVENT, the psychology and statistics citations, and the 2024–2025
interpretability entries). These unambiguously exist, so fabrication is not the concern; what remains
possible is a wrong year or a preprint-versus-proceedings mismatch — an error that embarrasses rather
than discredits. Coverage is **24 of 61 verified against a primary source, 0 errors of substance**.

Checklist §10 moves from ❌/❌/❌ to **⚠️/✅/✅**. It cannot honestly reach ✅ throughout without
reading the remaining 37, but the risk it was flagging was concentrated entirely in the set now
checked, and nothing was found.

---

## Round 16 — 2026-09-02 — em dashes removed from prose; en dashes deliberately kept

Author request after reading the Conclusion. A humanizer scan (Wikipedia "Signs of AI writing")
had already been run over the short build and returned **clean on every content pattern**: zero
inflated-importance phrasing, sales language, vague sources, shallow `-ing` analysis, "not X but Y",
filler, qualifier pileup, deeper-truth phrasing, formulaic sayings, fake-candid openings, unraised
objections, rejected fake alternatives, dramatic-fragment runs, repeated sentence openings, curly
quotes, emoji, or title-case headings. The em dashes were the one pattern the scan flagged and I had
argued to keep. The author disagreed, and that is the author's call.

### What changed: 8 prose sites, 10 dash characters

Each replacement was chosen individually rather than substituted globally, because the right
punctuation differs by clause:

| site | build | replacement |
|---|---|---|
| L66 abstract | short | colon, the clause explains the scoring choice |
| L93 intro | short | comma |
| L175 (paired) | short | parentheses; the clause already contains a citation |
| L225 mechanism | short | colon |
| L230 mechanism | short | semicolon, the clause is independent |
| L269 models | short | colon, the clause supplies the examples |
| L318 conclusion | short | colon |
| L660 appendix (paired) | **both** | parentheses |

Seven are short-branch only. **L660 is unconditional and therefore also changes the long build**,
which is past its VLM4RWD deadline; this is the second such shared edit, after Round 12's
`tab:factorial` caption.

### What was deliberately NOT changed

- **All 36 en dashes (`--`) are untouched.** They are numeric ranges (`$88$--$93\%$`, `layers
  13--17`, `$0.983$--$0.986$`). These are required LaTeX typography, not a style tell; replacing them
  would corrupt the numbers the paper reports.
- **`L353`** is a source comment, never rendered.
- **`L462`** reads "those cells are marked ---" and describes the literal character used in
  `tab:crosspatch`. Replacing it would make the caption describe a mark the table does not contain.
- **`L479`** is three table cells where `---` means *not applicable* under standard table notation.

A blind find-and-replace would have broken all four of these categories. That is why the sites were
enumerated and classified (prose / comment / table-literal) before anything was edited.

### Verification

**Zero em dashes remain in the rendered main text of either build** (`pdftotext` over pages 1–5 short
and 1–8 long). All 36 en dashes intact. Both builds at cap: VLM4RWD 8pp (refs p9), Interp4Discovery
5pp (refs p6). Short-build log free of multiply-defined, undefined-reference and overfull warnings.
Tests **159 passed / 0 failed**. Both Overleaf bundles rebuilt and *identical to toggled build*.

The replacements are net shorter than the dashes they replace, so no page pressure was created.

---

## Round 17 — 2026-09-02 — reviewer suggestions triaged against a full page 5

External (Perplexity) camera-ready suggestions, assessed against the hard 5pp cap. **Page 5 carries
52 numbered lines, the maximum any full text page reaches in this build** (p2 reaches 50; the
table-bearing p3/p4 hold 40 and 36). There is no line available, so each suggestion was judged on
whether it could be absorbed by re-wording rather than added.

### Applied — three near-neutral rewrites, all absorbed by line wrapping

1. **Discovery framing, introduction.** "…when localization evidence does and does not license a
   conclusion, **a precondition for turning model internals into knowledge**, rather than as a claim
   about vision-language architecture." Connects the paper to the workshop's question directly.
2. **Discovery framing, conclusion.** The closing sentence now reads "…only as good as the readout it
   is taken through: **before model internals can yield discoveries, the readout is what has to be
   checked first.**"
3. **Localization takeaway sharpened.** \S5 now reads "They do not establish necessity **or identify
   a unique circuit**", adopting the reviewer's explicit contrast between what patching supports
   (where information is readable and sufficient for restoration) and what it does not (a unique
   causal circuit).

Both builds remain at cap and the compile log stays warning-free.

### Already implemented before the review

- **"Elevate the key failure cases into a dedicated section."** The short build already has \S6
  *Where the measurement misleads* as a full main-text section, and \S5 carries the paragraph
  *The localization is readout-dependent*. Both failures are also named in the abstract and the
  introduction. The suggestion appears to describe the 8pp build, where this material is a
  subsection of a combined results section.
- **"Add a short takeaway paragraph on sufficiency versus necessity."** \S5 has ended with exactly
  that paragraph since before this review; only the unique-circuit clause was missing, and it is now
  added.

### Declined, with reasons

- **Main-text schematic of the experimental setup.** `figures/method_diagram.pdf` exists but needs
  15–20 lines. At 52/52 on page 5 it would cost a page, and the only way to pay is deleting a
  section. The checklist already records this as 2.6 ❌ and it is not closable at 5pp.
- **A bulleted "practical lessons" list.** Two reasons. It needs 4–5 lines that do not exist, and
  the lessons it would list (test alternative scoring rules, compare readouts, check tokenizer
  artifacts) are **already the content of \S6**, delivered as narrative. Converting narrative to
  bullets adds no information. It would also reintroduce the "list of bold mini-headings" pattern
  (§16 of the humanizer checklist) that Round 16 was cleaning up.
- **Restructuring the two failures into one subsection.** They currently sit in the sections whose
  evidence supports them, and both are surfaced in the abstract and introduction. Merging them is a
  structural change with real regression risk hours before a deadline, for presentational gain.

### The one suggestion that is fully feasible and not mine to complete

**Anonymous code repository.** The snapshot is built, audited and clean at
`~/Desktop/anon-code-snapshot` (98 files, zero identity strings, no git history, 144 passed / 2
skipped). Publishing it flips checklist items **5 and 13** from `\answerNo` to `\answerYes` — two of
the three remaining. Note that `anonymous.4open.science` proxies a GitHub repository, so the snapshot
must be pushed to one first, under an account not tied to the author.

---

## Round 18 — 2026-09-02 — the method schematic is in the short build

The reviewer's accessibility point (a mixed audience needs to see the setup) is now addressed:
`fig:method` renders in **both** builds. It was previously long-build only, and the CS-paper
checklist had recorded its absence as 2.6 ❌ across two runs.

### What it cost, and the measurement that decided the route

The figure needs ~27 lines (21 at `\linewidth` plus a 6-line caption) and page 5 was at 52/52.
Two routes were tested against the real build rather than estimated.

**Prose compression alone does not work, and the reason is worth recording.** Three aggressive
passes over \S2--\S7 removed **651 characters** and reduced the rendered main text from **205 lines
to 204** — one line. Character savings are mostly reabsorbed by paragraph re-wrapping: a tightened
sentence only removes a line when its paragraph sheds enough to drop a whole wrapped line. An
earlier estimate of 8--10% recoverable was extrapolated from character counts and was **wrong**;
freeing 30 lines this way would require 15--20% *content* removal, not tightening.

**The compression was still worth keeping**, and it is what made the figure possible: with the figure
pulled back out, it left page 5 at 47 lines instead of 52. That five-line margin closed the gap on
the configuration below, which had come up three lines short without it.

### Final configuration

- `fig:method` unconditional, at `\linewidth`, floating to page 2.
- `tables/patching` and `tables/models` move from the short build's main text to its appendix. **This
  is exactly the arrangement the 8pp build already uses** — both tables are appendix floats there —
  so the short build now matches it rather than inventing a structure.
- The \S2 `(Figure~\ref{fig:method})` pointer becomes unconditional, having been long-build only.
- \S5's "The lower rows are alignment checks" became "Its image, BOS and prefix-delimiter rows are
  alignment checks", since the table is no longer adjacent to the sentence.
- Prose compression retained across \S2, \S3, \S4, \S5, \S6 and \S7.

Float placement was also tested (`[h]` to `[tbp]` on both tables) and made **no difference**; that
change was reverted rather than left in as noise.

### Verification

Both builds at cap: VLM4RWD **8pp** (refs p9), Interp4Discovery **5pp** (refs p6). Short build now
lays out 35 / 25+figure / 47 / 50 / 49 lines across pages 1--5. Every float is defined exactly once
and referenced in **both** flattened builds (`tab:patching`, `tab:models`, `tab:mechmodels`,
`tab:minimal`, `tab:crosspatch`, `fig:method`) — no duplicates, no dangling references, no new
orphans. Short-build log free of multiply-defined, undefined-reference and overfull warnings. Zero em
dashes remain in the rendered main text. Tests **159 passed / 0 failed**. Both Overleaf bundles
rebuilt and *identical to toggled build*.

Checklist **2.6 moves from ❌ to ✅** in the short build. 7.2 (label sizes 5.8--7.6pt against an 8pt
guideline) still applies and is unchanged; the figure is at full `\linewidth`, which is the largest
it can be, so this is as legible as it gets.
