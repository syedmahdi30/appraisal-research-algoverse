# Stage C — a plain-language walkthrough for the team

This explains what Stage C was, what we did, and what we found — without assuming you've
read the code. The technical version with exact methods is in `docs/stage-c-results.md`;
this is the "follow along and understand what happened" version. It builds on Stage A
(`docs/stage-a-explainer.md`).

---

## TL;DR

Stage A showed the model represents "appraisals" (the good/bad-type judgments behind
emotions) **in text**. Stage C asks the payoff question:

> **Do those same text-learned appraisal directions still work when the model looks at a
> picture?**

Two findings:
1. **Yes — the transfer is real.** We take the pleasantness "reader" we trained on *text*,
   freeze it, and point it at the model's brain activity while it looks at *images*. It
   predicts how pleasant/positive the image is (compared to human valence ratings) far
   better than chance — correlation **0.51** across the full test set.
2. **And it's not just the model secretly captioning the image.** The obvious boring
   explanation is: "the model silently describes the photo in words, and the text reader
   reads that." We tested this directly and ruled it out — even when we let the model write
   a *detailed* description and account for it, the image still carries appraisal signal the
   words don't.

The honest ceiling: this is **correlational** evidence. It strongly suggests a shared
appraisal representation, but the clean proof is the causal test (Stage D).

---

## Background: what's the question again?

Appraisals are the mental judgments emotion theory says cause emotions (is this
**pleasant** or **unpleasant**? sudden? my fault?). Stage A found the model computes these
internally **for text**, and we built little linear "readers" (probes) that decode them
from the model's activations.

The whole point of the project is images. So Stage C takes the **frozen** text probe —
not retrained, not adjusted — and asks: when the model is looking at a photo of a person,
does that same reader still decode the appraisal? If yes, the model's appraisal
representation is (at least partly) **shared across seeing and reading** — the interesting
claim.

---

## The setup, in one picture

- **The reader:** the pleasantness probe from Stage A, frozen. It reads one spot in the
  model (layer 18) and outputs a "how pleasant" number.
- **The images:** EMOTIC — ~7,000 photos of people, each rated by humans for **valence**
  (how positive/negative they feel, 1–10). Valence is our stand-in for pleasantness.
- **What we do:** show the model each image, grab its internal state, run the frozen
  pleasantness reader on it, and check whether the reader's output lines up with the human
  valence rating.

One important detail we got right: the text probe outputs on a **1–5** scale, but valence is
**1–10**. So we never compare them with raw "accuracy" (r²) — that would look like failure
just from the scale mismatch. We use **correlation** (do they rise and fall together?),
which doesn't care about scale.

---

## Part 1 — Does the text reader work on images? ✅

**Result:** yes, clearly.

| | correlation with human valence | ranks positive vs negative images (AUC) |
|---|---:|---:|
| **Pleasantness** reader | **+0.51** | 0.90 |
| **Unpleasantness** reader | **−0.45** | mirror image ✓ |

**How to read this:** the frozen text pleasantness reader tracks how positive an image
feels at correlation 0.51, and it correctly sorts positive-emotion photos above
negative-emotion ones 90% of the time. Unpleasantness does the exact mirror (negative
correlation), which is a nice sanity check — the two readers behave oppositely, as they
should.

**The control:** we compared against 100 *random* readers (random directions of the same
size). Because the model's activations are "lumpy," random readers aren't at zero — they can
hit correlation ~0.36. But our real reader (0.51) beats **all 100** of them. So the signal
is real, not an accident of the geometry.

📊 Figure: `results/figures/stage_c_readout.png`.

---

## Part 2 — Is it real, or is the model just captioning the image? ✅

Here's the catch a good skeptic raises: maybe the model isn't "seeing" pleasantness at all.
Maybe it silently **describes the photo in words** ("a smiling woman…"), and our text reader
is just reading *that implicit caption*. If so, the "transfer" is mundane — it's the text
pipeline all along.

So we tested it head-on. For each image we had the model **actually write a caption**, fed
that caption through the **text** pipeline, and ran the same reader on it. Then we asked:
**how much does the image reader know that the caption reader doesn't?** (Statisticians call
this the "unique contribution" — what's left after you account for the caption.)

We did this at two levels of caption detail:

| We account for… | image reader's leftover unique signal |
|---|---:|
| a **plain** one-line caption | **+0.31** |
| a **detailed** caption (facial expression, posture, body language) | **+0.26** |
| **both captions at once** | **+0.20** |

(All three are statistically rock-solid, p < 0.001. Unpleasantness mirrors: −0.28, −0.20, −0.15.)

**How to read this:** as we let the model describe the image in more and more detail, the
caption catches *more* of the signal — so the image reader's "leftover" shrinks (0.31 → 0.26
→ 0.20). That's expected. **But it never collapses.** Even after accounting for a plain *and*
a detailed description of the person's expression, the image reader still knows a
significant amount the words don't. The picture-based appraisal signal is **not just the
model captioning to itself.**

📊 Figures: `results/figures/stage_c_caption_baseline.png` and `..._rich.png`.

---

## The key numbers to remember

**0.31 → 0.26 → 0.20.** That's the image reader's unique appraisal signal after accounting
for a plain caption, then a rich caption, then both. It shrinks (richer words explain some
of it) but stays large and significant (words don't explain all of it).

And **0.51** — the headline: how well the frozen *text* pleasantness reader predicts *image*
valence.

---

## What we can and can't claim

- ✅ **The transfer is real** (0.51, beats every random control).
- ✅ **It's not merely the model captioning the image** — a detailed verbal description
  doesn't account for it.
- ⚠️ **We have not *proven* a shared non-verbal appraisal representation.** This is
  correlational, and it's an *upper bound* — an even richer description might explain more.
  The residual could also, in principle, be some visual feature that happens to track
  valence rather than "appraisal" specifically.

That last caveat is exactly why the next stage exists.

---

## What this means for the project

Stage C is the bridge from "the model does appraisals in text" (Stage A) to "the model does
appraisals when it sees" — and it holds up under the obvious skeptical attack. That's a
genuine, reportable result.

**Next: Stage D — the causal test.** Everything in Stage C is *correlational* ("the reader's
output lines up with valence"). Stage D flips it to *causal*: we **push** the model's internal
state along the appraisal direction *while it looks at an image* and check whether its emotion
output moves the way theory predicts. If it does, no "it's just captioning" argument can
explain it away — you've shown the appraisal direction *drives* image-based behavior. And we
already know the trick from Stage A: use **difference-of-means** directions, not the probe's
own weights.

---

## The honest caveats (so nobody over-claims)

- **Correlational, not causal** — the shared-representation claim waits on Stage D.
- **Valence is a proxy** — EMOTIC never rated "pleasantness"; we use its valence and
  positive/negative emotion labels as stand-ins.
- **Lossy labels** — mapping EMOTIC's 26 emotions down to a 7-way space (for the AUC) throws
  away information and keeps only single-label images (440 of them).
- **We don't tell the model which person** — EMOTIC photos can have several people with
  boxes; we feed the whole image and ask about "this person" without pointing. That adds
  noise (the signal came through anyway).
- **One setup** — one model (Gemma-3-4B), one layer (18), one seed; the deep mechanism test
  used 1,000 images, the headline number used all ~7,280.

---

## Where everything is / how to reproduce

```bash
python -m src.experiments.stage_c_transfer --full          # Part 1: read-out on all ~7,280 images
python -m src.experiments.stage_c_caption                  # Part 2: plain-caption baseline
python -m src.experiments.stage_c_caption --style rich     # Part 2: rich-caption robustness
python -m src.experiments.analyze_stage_c_mechanism        # Part 2: combined (CPU-only, seconds)
```

Outputs (saved to Google Drive under `results/`):
- `results/stage_c/metrics.json` — the read-out numbers (Part 1)
- `results/stage_c/caption_metrics{,_rich}.json` + `mechanism_summary.json` — the caption tests
- `results/stage_c/caption_readout{,_rich}.parquet` — per-image data (so re-analyses skip re-captioning)
- `results/figures/stage_c_{readout, caption_baseline, caption_baseline_rich}.png` — the figures

Running it yourself needs the Colab A100 setup — see `docs/colab.md`.
