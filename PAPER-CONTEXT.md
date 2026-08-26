# Paper context — carry-over brief

*Put this in the project folder. It is the accumulated state from the venue-selection work, so the new project does not start cold.*

---

## The paper

**Current title:** *Negative Words Defeat Positive Images: Valence-Asymmetric Conflict Resolution in Vision-Language Models*

**Working TAE title direction:** something that makes measurement validity the subject and the VLM result the evidence — e.g. *Mechanism or Artifact? Measurement Validity in Multimodal Conflict Evaluation*. The current title reads as an interpretability paper and will misdirect a TAE reviewer in the first two seconds.

**Source:** `neurips_2026.tex` in the Algoverse Research folder. Main text ran ~10 pages as of 21 Aug 2026 (refs from p.10, appendix from p.11 of 25). Needs a ~2-page cut for TAE's 8-page cap.

## Claims and their evidence

**1. Valence-asymmetric conflict resolution (Gemma-3-4B).** Negative context moves positive-image judgments −0.598; positive context moves negative-image judgments +0.333. Mirror contrast +0.265, 95% CI [+0.095, +0.431], ~1.8×, Mann-Whitney p = .005. Override rates 84% vs 21%, gap +64% CI [+55, +72]. Survives head-room normalisation (0.632 vs 0.259).

**2. It is valence, not event semantics.** Six token-matched minimal pairs flipping one valence word (won/lost, wonderful/devastating, celebration/memorial…) reproduce the gap (+65% vs +64%) and sharpen the mirror contrast (+0.330). Text-only bank strength is comparable (1.06, minimal pairs 1.25).

**3. Architecture-dependent, not universal.** Gemma-3-4B +64%, Qwen3-VL-8B +39%, LLaVA-1.5-7B **−13%** (head-room-normalised pulls exactly symmetric at 0.33/0.33). Gemma-3-12B retains +51% but both directions become more text-driven (neg 84%→93%, pos 19%→42%).

**4. Mechanism (Gemma).** Frozen text-trained pleasantness probe transfers to image-conditioned activations (ρ = .507, polarity AUC .898, semipartial ρ = .201 after joint caption control). Same-image patching: assistant-turn boundary recovers 57–65% of the context delta, all aligned text 85–87%, image tokens ~0%. Cross-image patching: image-token recovery 80% early → 9% late, non-image text 10% → 68%. Steering at layer 18 shifts conflict outcomes at slope +0.215.

## Known soft spots — all already disclosed in the paper

Single seed throughout. Within-item paired test marginal (p = .052). Uncertainty propagated over images but not jointly over context sentences. Whole EMOTIC images without target-person marking. One dataset and task family. Mechanism thorough on one model, coarse on a second, absent on the third. Bounded readout with no unbounded logit-margin replication. Full-bank patching pairs not length-matched. The ~0% image-token row is partly guaranteed by causal ordering. Assistant-boundary concentration lacks a matched non-affective control, so a generic-bottleneck reading is open.

## Related work — do not overclaim against these

The author has read all four in depth. Between them they already establish that text can override vision; that arbitration varies with task difficulty, model, semantic relation, text relevance, unimodal certainty and token order; that emotional rhetoric is one persuasion strategy among several; and that controlled semantic counterfactuals and directional-shift measurement exist.

- **Deng et al. 2025** — blind faith in text. Varies relevance, correctness, certainty, token order.
- **Pezeshkpour et al. 2025** — Mixed Signals. Modality preference flips with task complexity and semantic relation.
- **Zhang et al. 2026** — ConText-VQA. Persuasive misinformation incl. an *emotional appeal* condition — but that is rhetoric, not valence polarity, and emotional appeal was the *weakest* strategy. Say so explicitly; a reviewer will otherwise think emotion was already covered.
- **Li et al. 2026** — SIGNPOST-Bench. Paired counterfactuals, continuous directional shift toward a text-specified target.

**Retired framing:** any sentence claiming prior work treats competing content as interchangeable, ignores content properties, or never measures directional influence. All three are now false.

**Safe framing:** prior work shows arbitration varies with relevance, difficulty, semantic relation, rhetorical strategy and scene-text specificity; what remains untested is whether it is *directionally asymmetric in valence* under otherwise matched evidence.

## Repo state

`.gitignore` contains `/results/` — "Results (regenerable — not tracked)". So experiment outputs are deliberately untracked and live only on whichever machine ran them.

On the local machine as of 21 Aug 2026: `results/stage_a`, `stage_c` and `stage_d` hold output JSON. **`results/stage_e` and `results/stage_f` are empty.**

Stage F is where every headline behavioural number comes from — the 84%/21% override rates, the mirror contrast, the minimal pairs, the cross-model comparison, and both patching tables. The experiment code is all present (`src/experiments/stage_f_conflict.py`, `stage_f_patching.py`, `stage_f_cross_patching.py`, `stage_f_llava.py`, `stage_f_text_only.py`, `stage_f_attribution.py`, and more), and the figures were plotted (`paper/figures/stage_f_*.png`), so the runs happened — most likely on Colab, given `scripts/colab_bootstrap.py`. The raw outputs just never came back to this machine.

**Action before the first `tae-verify-claims` run:** retrieve the stage_e and stage_f outputs from wherever they were generated and restore them under `results/`. Until then the claim-checker can verify the probe, caption-control and steering numbers, and nothing else.

This is worth treating as more than housekeeping. The paper argues that evaluations must license the claims built on them, at a venue devoted to that proposition. Headline numbers that cannot currently be traced to stored outputs are exactly the thing the paper is about.

## Bibliography status

`references_fixed.bib` (21 Aug) has all BibTeX syntax repaired — 23 malformed author fields, duplicate arXiv IDs, acronym bracing, encoding, page ranges. Compiles clean under `plainnat` with zero warnings.

Still unverified against the actual papers:
- `contextvqa2026` and `signpost2026` titles differ substantially from what the PDFs appear to be called — check the title pages.
- `narrowgate2024` and `nikankin2025` both give NeurIPS volume 38 with year 2026; volume 38 is NeurIPS 2025.
- ICLR entries (`towardsinterp2024`, `xiao2024`, `zhang2026anydepth`) carry page ranges; ICLR papers have none.
- `camel2025` lists "Li Peihang" among given-name-first entries; probably "Peihang Li".
- Fifteen 2026-dated arXiv-only entries have not been confirmed to exist.

The header comment in the .bib must be deleted before submission.

## Outstanding pre-submission items

1. Verify the fifteen unconfirmed citations.
2. Write the responsible-use / broader-impact section — there is none anywhere in the .tex.
3. Cut main text 10 pp → 8 pp.
4. Confirm with TAE chairs (`aiteval2026@gmail.com`) whether concurrent NeurIPS-main review and concurrent submission to VLM4RWD are permitted; their CFP is silent on both.

## The other submission

The same work also goes to **VLM4RWD** (NeurIPS 2026, 30 Aug, 8 pp, Sydney) under a grounding-failure framing — override rate reframed as a grounding-failure rate, architecture result as procurement guidance. Same 8-page base, different title and framing paragraph. Keep the two versions' shared material in sync.
