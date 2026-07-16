# Stage D — a plain-language walkthrough for the team

This explains what Stage D was, what we did, and what we found — without assuming you've
read the code. The technical version with exact methods is in `docs/stage-d-results.md`;
this is the "follow along and understand what happened" version. It's the finale of the
arc: Stage A (text) → Stage C (reading appraisals off images) → Stage D (this one).

---

## TL;DR

Stage C showed that a text-trained "pleasantness reader" still works when the model looks
at images — but that's a *correlation* (the reader's output lines up with how positive the
image is). A skeptic can always ask: "correlation with what? maybe it's a fluke of how the
model happens to encode things."

Stage D settles it by switching from **reading** to **pushing**:

> If we **push** the model's internal state along the text-learned "pleasantness" direction
> *while it's looking at a photo*, does its emotion answer actually **change** the way emotion
> theory predicts?

**Yes.** Push toward pleasantness → the model's emotion output gets more positive. Push toward
unpleasantness → more negative. Smoothly, in proportion to how hard we push. A **random** push
of the same size does almost nothing, and pushing along an *unrelated* appraisal (suddenness)
barely moves it either.

That's a **causal** result — and a causal result can't be explained by "the model is just
describing the image in words." This is the strongest evidence in the whole project that the
model really shares an appraisal representation across reading and seeing.

---

## Background: why "pushing" beats "reading"

Everything up to now was **correlational**: we *read* a number off the model and checked it
lines up with human ratings. Correlations can always be second-guessed ("maybe it's really
measuring something else").

**Steering** is different. Instead of passively reading, we **intervene** — we reach into the
model mid-thought and nudge its internal state along a specific direction, then watch what
comes out. If a specific, meaningful nudge produces a specific, predicted change in behavior,
that's cause and effect. There's no "maybe it's secretly captioning" loophole, because we
didn't ask it to describe anything — we directly moved the appraisal dial and the emotion
answer followed.

One thing we already knew from Stage A: the direction that's good for *reading* a feature is
**not** the direction that *pushes* it. For pushing you need a "difference-of-means"
direction (literally: the average internal state for high-pleasantness minus the average for
low-pleasantness). We reuse that recipe here.

---

## The setup, in one picture

- **The dial:** the text-learned "pleasantness" direction (and "unpleasantness," and
  "suddenness"). These come **entirely from text** — the model never saw an image while we
  built them. That's the whole point: a *text*-learned dial steering *image* behavior.
- **The nudge:** while the model looks at a photo, we add "β × direction" to its internal
  state at layer 18. β is how hard we push (we sweep −3 to +3).
- **What we measure:** the model's emotion answer, boiled down to one number — **valence** =
  (chance it says a positive emotion) − (chance it says a negative one). We compare that number
  with and without the nudge.

---

## The result ✅

Pushing along each direction, here's how the model's valence moved (150 images):

| We pushed along… | hard − | ← | → | hard + | overall trend |
|---|---:|---:|---:|---:|:--|
| **Pleasantness** | −1.05 | | | +0.76 | **rises — correct** |
| **Unpleasantness** | +0.72 | | | −1.01 | **falls — correct (mirror)** |
| **Suddenness** (unrelated) | +0.20 | | | −0.23 | barely moves |
| **Random** (control) | +0.02 | | | −0.13 | barely moves |

**How to read this:** as we push harder toward pleasantness (left → right), the model's emotion
output climbs steadily from very negative to positive. Unpleasantness does the exact opposite.
Meanwhile a **random** push of the same size, and a push along an **unrelated** appraisal
(suddenness), stay close to flat — roughly **4–12× smaller**. So it's specifically the
*appraisal* direction causing the change, not just "poking the model."

📊 Figure: `results/figures/stage_d_steering.png` — pleasantness sloping up, unpleasantness
sloping down, the two controls flat through the middle.

---

## Why the two controls matter

A steering result is only convincing if you show a nudge that *shouldn't* work, doesn't:

- **Random control** — a random direction of the exact same size. If this moved the output as
  much as pleasantness, we'd have just been shoving the model around. It stayed ~12× smaller. ✓
- **Suddenness (specificity) control** — a *real* appraisal direction, but one that isn't about
  good/bad. If *every* appraisal direction moved valence, the effect wouldn't be specific to
  pleasantness. Suddenness stayed ~4.5× smaller. ✓ (It's not perfectly zero — sudden events tend
  to be a bit unpleasant, so a tiny leak is actually what theory expects.)

---

## What we can and can't claim

- ✅ **The text-learned appraisal direction causally steers image behavior.** Push pleasantness,
  the emotion output gets more positive; push unpleasantness, more negative; controls stay flat.
- ✅ **This can't be "the model is just captioning."** We didn't ask for a description — we moved
  an internal dial directly. Verbalization has no way to explain a steering effect.
- ⚠️ **It's one model, one layer, one seed, 150 images.** The effect is large and clean, but
  broader robustness (more seeds, a second model) is still "nice to have" for a paper.

---

## What this means — the whole story now closes

Put the three stages together:

| Stage | Question | Answer |
|---|---|---|
| **A** | Does the model do appraisals in **text**? | Yes — readable *and* causally steerable |
| **C** | Do those text directions **read** appraisals off **images**? | Yes (ρ=0.51), and not just from captioning |
| **D** | Do those text directions **causally steer** image behavior? | **Yes** |

So: a direction the model learned purely from **reading** about emotions can be used to
**read** and to **steer** what it does when it **sees** — correlationally *and* causally, with
the "it's just describing the picture" alternative ruled out both ways. That's a coherent,
strong claim about a **shared appraisal representation across language and vision.**

---

## The honest caveats (so nobody over-claims)

- **One setup** — Gemma-3-4B, layer 18, seed 0, 150 images. Large clean effect, but not yet
  multi-seed or multi-model.
- **Valence is a blunt readout** — it squashes 13 emotions into "positive vs negative." The
  *direction* of the effect is robust; the exact sizes depend on this choice.
- **Steered two appraisals** — pleasantness and unpleasantness (the ones with a clear
  good/bad meaning); suddenness was a control, not a claim.
- **No person pointing** — as in Stage C, we feed the whole image and say "this person" without
  marking who, which adds noise (the effect came through anyway).

---

## Where everything is / how to reproduce

```bash
python -m src.experiments.stage_d_steering --limit 10   # quick sanity dry run
python -m src.experiments.stage_d_steering              # full sweep (150 images)
```

Outputs (saved to Google Drive under `results/`):
- `results/stage_d/steering_metrics.json` — the steering numbers
- `results/figures/stage_d_steering.png` — the figure above

Running it yourself needs the Colab A100 setup — see `docs/colab.md`.
