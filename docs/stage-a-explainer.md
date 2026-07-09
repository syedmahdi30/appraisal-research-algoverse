# Stage A — a plain-language walkthrough for the team

This explains what Stage A was, what we did, and what we found — without assuming you've
read the code. The technical version with exact methods is in `docs/stage-a-results.md`;
this is the "follow along and understand what happened" version.

---

## TL;DR

We asked: **does Gemma-3-4B (a language model) internally represent "appraisals" — the
mental judgments that emotion theory says cause emotions — and can we use them?**

Two findings, both positive:
1. **Yes, we can read appraisals out of the model.** Simple linear readers ("probes")
   recover six appraisal dimensions from the model's mid-layer activations, far better than
   chance. Pleasantness and unpleasantness are the clearest.
2. **Yes, those appraisals are causal.** When we push the model's internal state along an
   appraisal direction, its emotion output shifts the way emotion theory predicts — and a
   random push of the same size does nothing. (This one took two tries; see below.)

Bonus lesson: **the direction you use to *read* a feature is not the direction that *steers*
it.** That surprised us and is a genuinely useful result.

---

## Background: what's an "appraisal"?

Appraisal theory says emotions come from how you *evaluate* a situation along a few
dimensions. For example:
- **Pleasantness / Unpleasantness** — is this good or bad?
- **Suddenness** — did it happen out of nowhere?
- **Event predictability** — could I have seen it coming?
- **Own responsibility** — was it my fault?
- **Others' responsibility** — was it someone else's fault?

The idea: if you know the appraisals, you can predict the emotion. Feeling that something is
*unpleasant + someone else's fault + sudden* → anger. *Pleasant + my accomplishment* → pride.

Prior work (Tak et al. 2025) found that language models seem to compute these appraisals
internally. **Stage A's job was to reproduce that in our specific model (Gemma-3-4B)** before
we try anything fancy with images (that's Stage C).

---

## The setup, in one picture

- **Model:** Gemma-3-4B — a 34-layer network. Think of it as doing 34 rounds of "thinking,"
  and at each round it holds an internal state (a big vector of 2,560 numbers) for each word.
- **Data:** crowd-enVENT — 6,600 short real event descriptions people wrote, each rated by
  the writer on the appraisal dimensions (1–5) and labeled with the emotion they felt.
- **What we grab:** we feed each event to the model and record its internal state at the last
  word, at every layer. Then we ask two questions of those internal states.

---

## Part 1 — Can we READ appraisals out of the model? (Read-out) ✅

**Method (simple version):** for each appraisal, train a tiny linear model (a "probe") that
looks at the model's internal state and tries to predict that appraisal's 1–5 rating. If the
probe does well, the information is in there. We measure success with **r²** (0 = no better
than guessing the average, 1 = perfect). We check every layer to see *where* in the network
the information lives, and we compare against a **shuffled control** (train the same probe on
scrambled labels — it should score ~0).

**Result:** all six appraisals are clearly readable, and they all live in the **middle of the
network (layers ~17–21 of 34)**, specifically in the attention part of those layers. This
matches the prior paper's finding.

| Appraisal | Best layer | r² (higher = better) | shuffled control |
|---|---:|---:|---:|
| Pleasantness | 18 | **0.64** | 0.01 |
| Unpleasantness | 18 | **0.60** | −0.04 |
| Own responsibility | 21 | 0.44 | −0.01 |
| Suddenness | 17 | 0.32 | 0.00 |
| Others' responsibility | 19 | 0.31 | −0.02 |
| Event predictability | 17 | 0.24 | −0.03 |

**How to read this:** every appraisal scores well above its shuffled control (which sits at
~0), so this isn't luck. Pleasantness/unpleasantness are the strongest — good news, because
those map onto "valence" (good/bad feeling), which we'll use as our bridge to images later.

📊 Figure: `results/figures/stage_a_localization.png` — shows each appraisal's r² rising to a
peak in the middle layers and the flat control line at 0.

---

## Part 2 — Are those appraisals CAUSAL? (Steering) ✅ (after a plot twist)

Reading a feature out doesn't prove the model *uses* it. To test causality we **steer**: we
nudge the model's internal state along an appraisal's direction and see if its emotion output
changes the way theory predicts.

Our readout here is a **valence score** = (probability the model says a positive emotion) −
(probability it says a negative emotion). Prediction: push **+pleasantness** → valence should
go **up**; push **+unpleasantness** → valence should go **down**. And a **random** nudge of the
same size should do nothing (that's the control that proves it's the *direction* that matters).

### Attempt 1 — it failed 🚫

Our first steering method did **not** work: pushing along the appraisal direction moved the
output no more than a random push did. We couldn't tell signal from noise.

Two reasons we figured out:
- **Gemma's internal states are huge** (their "length" is ~37,000). The model normalizes by
  that size at every layer, so a small nudge gets washed out — but a big nudge just scrambles
  everything (even the random control moved). There was no "just right" size.
- **We used the wrong kind of direction** (see the lesson below).

### Attempt 2 — it worked ✅

We changed the **direction** we steer with. Instead of the probe's "reading" direction, we
used a **difference-of-means** direction: take the average internal state of *high-pleasantness*
events, subtract the average of *low-pleasantness* events. That difference is literally "which
way the model's state moves when pleasantness goes up." We steer along that, in natural units
(β = 1 means "one full low→high shift").

This gave a clean result:

| Push direction | strong − | ← | weak | weak | → | strong + | trend |
|---|---:|---:|---:|---:|---:|---:|:--|
| **Pleasantness** (expect ↑) | −0.19 | −0.12 | −0.06 | +0.10 | +0.21 | +0.32 | **rises, correct** |
| **Unpleasantness** (expect ↓) | +0.28 | +0.18 | +0.10 | −0.05 | −0.10 | −0.17 | **falls, correct** |
| **Random** (expect flat) | +0.03 | +0.02 | +0.01 | 0.00 | −0.01 | −0.02 | **flat ✓** |

**How to read this:** as we push harder on pleasantness (left→right), valence climbs smoothly
and in the right direction. Unpleasantness does the mirror image. The random control barely
moves — about 10× smaller. That gap is the whole point: **the appraisal direction causes the
change; a random direction of the same size does not.**

📊 Figure: `results/figures/stage_a_steering_v2.png` — pleasantness sloping up, unpleasantness
sloping down, random flat through the middle.

---

## The key lesson (worth remembering for the whole project)

**The direction that *reads* a feature is not the direction that *changes* it.**

- The **probe** direction is great for *decoding* an appraisal but useless for *steering*.
- The **difference-of-means** direction is what actually *moves* the model's behavior.

So when we do cross-modal steering later (Stage D on images), we'll use difference-of-means,
not the probe weights. This saved us from repeating the same failure on images.

---

## What this means for the project

Stage A was a **go/no-go gate**: if the model didn't represent appraisals, cross-modal
transfer would be meaningless. Both halves passed:
- Appraisals are **decodable** (strong probes) ✅
- Appraisals are **causal** (steering works) ✅

So we have the green light for **Stage C**: take these exact appraisal probes, freeze them,
and apply them to the model's activations *when it's looking at images* (EMOTIC dataset). The
core question there: do the same appraisal directions we found in text also show up when the
model processes pictures of people? That's the actual novel contribution.

---

## The honest caveats (so nobody over-claims)

- **Data split:** the public download is one big file, so we made our own train/val/test
  split (reproducible, fixed seed) — it may not be identical to the original paper's split, so
  exact r² numbers aren't directly comparable.
- **One run / one seed:** we haven't yet repeated across seeds to measure variability.
- **Steering shown for 2 appraisals:** we demonstrated causal steering for pleasantness and
  unpleasantness (they have a clean good/bad readout). The other four don't have an obvious
  single readout, so we don't claim steering for them.
- **Valence is a blunt readout:** it lumps 13 emotions into "positive vs negative." The clean
  trend survives that, but the exact effect sizes depend on the metric.

---

## Where everything is / how to reproduce

Code (all in the repo):
```bash
python -m src.experiments.stage_a_text          # trains probes, finds best layers
python -m src.experiments.analyze_stage_a       # Part 1 table + figure
python -m src.experiments.stage_a_steering_v2   # Part 2 steering (the one that works)
# (src/experiments/stage_a_steering.py is the failed v1, kept as the contrast/control)
```

Outputs (saved to Google Drive under `results/`):
- `results/stage_a/metrics.json` — all the read-out numbers
- `results/stage_a/probes.npz` — the frozen probes we carry into Stage C
- `results/stage_a/steering_v2_metrics.json` — the steering numbers
- `results/figures/stage_a_localization.png` and `stage_a_steering_v2.png` — the two figures above

Running it yourself needs the Colab A100 setup — see `docs/colab.md`.
