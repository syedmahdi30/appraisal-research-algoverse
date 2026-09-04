# Reviewer controls — Colab runbook

Five GPU runs answering an external reviewer's three priority objections. Written 2026-09-03, VLM4RWD
deadline 2026-09-05 ~21:00 PT.

All five go through `src/experiments/stage_f_controls.py`, which sits on `stage_f_qwen`'s **raw
HuggingFace** path. It must stay there: the withdrawn prompt sweep (`stage_f_prompts.py`) booted via
`boot_gemma`, and that wrapper altered the multimodal forward pass, which is why its results were
withdrawn from the paper. Nothing in these runs may reintroduce it.

## Before you start

```bash
pip install -r requirements-qwen.txt        # transformers>=4.57 for Qwen3-VL
```

### Staging the images — this is where the first two attempts failed

`data/processed/emotic_test.parquet` stores **absolute** `/content/...` paths from the original run, so
a session that mounts EMOTIC anywhere else resolves nothing. Both first attempts scored **4 of 150**
images. Diagnose before running anything:

```python
import pandas as pd, os, glob
d = pd.read_parquet("data/processed/emotic_test.parquet")
print("present:", d.image_path.map(os.path.exists).mean())          # want 1.0
print(d.assign(ok=d.image_path.map(os.path.exists)).groupby("folder").ok.agg(["mean", "size"]))

# Where are the images actually? Search for one known filename.
name = d.filename.iloc[0]
print("found at:", glob.glob(f"/content/**/{name}", recursive=True)[:3])
```

**The 150 selected images need all four sub-corpora:** `emodb_small` 100, `framesdb` 26, `mscoco` 23,
`ade20k` 1. `emodb_small` and `framesdb` ship inside the EMOTIC archive; **`mscoco` and `ade20k`
images do not** — EMOTIC distributes only their annotations, and the images come from MSCOCO and
ADE20K themselves. A correct EMOTIC-only extraction therefore reaches 126/150 (84%), not 100%.

Once the search above tells you the real root, point the runner at it — paths are rebuilt exactly from
the parquet's own `folder` and `filename` columns, not by prefix guessing:

```bash
python -m src.experiments.stage_f_controls --person box --images-root /content/emotic
```

The root is whatever directory contains `framesdb/`, `mscoco/`, `emodb_small/` and `ade20k/`.

### If you cannot get mscoco/ade20k in time

The runner aborts above 5% missing, because a control silently scored on a fraction of the images is
worse than a crash. For **`--person box` / `--person crop` that gate is conservative rather than
necessary**: those modes now score a *paired ungrounded arm on the same images in the same run*, so the
grounded-vs-ungrounded contrast is internally valid on whatever subset is mounted. Losing images costs
precision and the comparison to the published 121-image numbers, not validity. So:

```bash
python -m src.experiments.stage_f_controls --person box --images-root <root> --allow-missing
```

is defensible **if and only if** you report it as a paired contrast on n images rather than as a
replication of the published number. The `--axis` sweeps carry their own `original` baseline, so the
same argument applies to them.

A `mirror` column of `n/a` is a symptom, not a result: the mirror contrast needs both conflict
directions, so it disappears when one image group has no usable images.

## The runs, in priority order

Run them in this order. If you run out of time, everything above the line you stop at is still
publishable on its own.

| # | Command | Passes | Rough A100 time |
|---|---|---|---|
| 1 | `python -m src.experiments.stage_f_controls --person box` | 4,500 (2 arms) | ~20 min |
| 2 | `python -m src.experiments.stage_f_controls --axis frame` | 9,000 | ~40 min |
| 3 | `python -m src.experiments.stage_f_controls --person crop` | 4,500 (2 arms) | ~20 min |
| 4 | `python -m src.experiments.stage_f_controls --axis question` | 9,000 | ~40 min |
| 5 | `python -m src.experiments.stage_f_controls --generate` | 2,250 generations | ~20 min |

**Do not pass `--limit`.** It changes which images are selected (`select_extreme_rows` takes `n//2`
from each valence end), so a smaller run is a *different* image set, not a subset of this one, and
would not be comparable to the published numbers.

### What each one is for

1. **`--person box`** outlines the annotated EMOTIC person and keeps the scene. This is the cleanest
   answer to the reviewer's biggest objection: the prompt asks about "this person" while 65% of our
   images carry more than one annotated person (81% of the positive group). It separates *grounding*
   from *image content*.
2. **`--axis frame`** re-renders the six valence swaps in four context frames — the published one
   plus the caption, report and user-message carriers §5 names. Tests whether the asymmetry is a
   property of valence or of the `"This photo was taken …"` construction.
3. **`--person crop`** crops to the person, deleting the scene. Stronger than `box` but confounded,
   since it removes information as well as ambiguity. Informative mainly in **agreement or
   disagreement with `box`**: if `box` holds and `crop` collapses, the effect is scene-mediated.
4. **`--axis question`** varies the question, holding the context frame fixed. Closes the withdrawn
   sweep and lets the paper drop "question phrasing remains untested".
5. **`--generate`** decodes greedily instead of scoring forced-choice log-probabilities, so the effect
   can be checked against what the model would actually emit.

## Reading the output

Every run prints a per-variant table and saves `results/stage_f/<stem>.parquet` plus
`<stem>_metrics.json`. The reference line printed under the table is the published matched-set result:

```
within-item +1.148 [+0.943,+1.344];  mirror +0.496, crossed [+0.11,+0.83]
```

The two sweeps contain their own replication: **`original` in `--axis frame` and `--axis question`
should land on those published numbers.** If it does not, something in the run differs from the
published one and the other variants cannot be interpreted — stop and report that rather than the
variant table.

For `--person box` / `--person crop` the table has **two rows**: `none` (paired ungrounded baseline,
same images) and the grounded mode. **Compare those two to each other** — that is the control. The
published line is a secondary check that only applies if the full image set was mounted. `n_ungrounded` in the metrics counts rows whose bbox was unusable and which were
therefore scored *ungrounded*; if that is not ~0 the control is diluted.

**What counts as the control passing:** the mirror contrast stays positive with a crossed interval
excluding zero, and the within-item contrast stays in the neighbourhood of the published value. A
mirror contrast whose crossed interval now includes zero means the control did not reproduce, which
is a reportable result, not a failed run.

## Bring back

Just the files — I will do the analysis and the write-up on CPU:

```
results/stage_f/controls_person_box_qwen.parquet          + _metrics.json
results/stage_f/controls_person_crop_qwen.parquet         + _metrics.json
results/stage_f/controls_frame_qwen.parquet               + _metrics.json
results/stage_f/controls_question_qwen.parquet            + _metrics.json
results/stage_f/controls_generate_qwen.parquet            + _metrics.json
```

Anything already saved can be re-analysed without a GPU:

```bash
python -m src.experiments.stage_f_controls --reanalyze controls_person_box_qwen
```

## Agreed in advance

These are being run **to be reported whichever way they come out.** If a control contradicts the
headline it goes into Limitations as a disclosed control rather than being dropped — the paper's
credibility already rests on disclosed nulls (the varied set failing crossed resampling, LLaVA-NeXT
reversing, the two withdrawn experiments), and a reviewer finding this later is far worse than us
finding it now.
