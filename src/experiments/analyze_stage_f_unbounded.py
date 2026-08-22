"""Stage F robustness re-analysis: an UNBOUNDED read-out and a CROSSED (image x sentence) bootstrap.

Two limitations are stated in the paper and closed here. Both are pure re-analysis of the stored
conflict parquets (`lp_{label}` log-probs, every row, every model) — no forward passes, no GPU.

  (1) BOUNDED READ-OUT. Behavioral valence is P(pos) - P(neg) in [-1, 1]. It saturates (visibly on
      Qwen, whose cells sit at +-0.999), so a "smaller" effect can mean a squashed one rather than a
      weaker one, and the head-room normalization that patches this is itself a modelling choice.
      The unbounded analogue needs no correction:

          margin = logsumexp(lp_pos) - logsumexp(lp_neg) = log[ P(pos) / P(neg) ]

      a log-odds score with no ceiling. If the asymmetry is an artefact of squashing near +-1 it must
      shrink or vanish here; if it is real it survives. NOTE the units change: margin contrasts are
      log-odds, NOT comparable in magnitude to the valence contrasts (+0.265 etc). Compare the SIGN,
      whether the CI clears zero, and the scale-free ratio |drop|/|rise|.

  (2) IMAGE-ONLY UNCERTAINTY. The published CIs bootstrap over images, holding the 6+6 context
      sentences fixed, so they answer "would this hold on new photos?" but not "...on new sentences?".
      With only 6 sentences per polarity that second source is not negligible. `_crossed_bootstrap`
      resamples images AND context ids independently with replacement (a two-way cluster bootstrap
      over crossed random effects) and therefore widens the interval to cover both.

Anchor discipline mirrors `analyze_judge_robustness`: the bounded + image-only cell of the output
grid must reproduce the published number before the other cells are read.

    python -m src.experiments.analyze_stage_f_unbounded --gemma <pq> --qwen <pq> --llava <pq>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..data.labels import EMOTION_LABELS
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, run_stamp, save_json
from .analyze_stage_f import _POSITIVE, _NEGATIVE

LP = [f"lp_{w}" for w in EMOTION_LABELS]
_POS_IDX = [EMOTION_LABELS.index(w) for w in _POSITIVE]
_NEG_IDX = [EMOTION_LABELS.index(w) for w in _NEGATIVE]


# --------------------------------------------------------------------------- read-outs
def add_readouts(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the unbounded log-odds margin and the per-image block id used for grouping.

    `image_path` recurs across rows (EMOTIC annotates per person, so one photo can appear twice), so
    images are keyed the way the published analyzer keys them: the base pass writes each image as
    [none, <contexts...>], hence a cumsum on the `none` row uniquely labels each image block.
    """
    d = df.copy()
    lp = d[LP].to_numpy(dtype=float)
    d["margin"] = logsumexp(lp[:, _POS_IDX], axis=1) - logsumexp(lp[:, _NEG_IDX], axis=1)
    # Cluster by PHOTO, not by person-annotation. EMOTIC annotates each person, but the model is shown
    # the whole image with no bounding box, so two annotated people in one photo yield byte-identical
    # forward passes — duplicate rows, not independent observations. The previous
    # `(condition == "none").cumsum()` counted person-annotations (150 units over 121 distinct photos),
    # which treats those 29 duplicates as independent and narrows every interval. `image_path` is also
    # the unit `analyze_stage_f._flip_override` clusters on, so the two tools now agree.
    d["_img"] = d["image_path"]
    return d


def _effect_matrix(d: pd.DataFrame, group: str, cond: str, value: str):
    """(images x context sentences) matrix of effects vs each image's own neutral baseline.

    Rows are image blocks in `group`, columns are the `context_id`s of condition `cond`, cells are
    value(image, sentence) - mean value(image, neutral). Keeping the sentence axis un-collapsed is
    what makes the crossed bootstrap possible; the published estimator averages it away immediately.
    """
    sub = d[d["image_group"] == group]
    neutral = sub[sub["condition"] == "neutral"].groupby("_img")[value].mean()
    cells = sub[sub["condition"] == cond]
    if cells.empty or neutral.empty:
        return None
    m = cells.pivot_table(index="_img", columns="context_id", values=value, aggfunc="mean")
    m = m.loc[m.index.intersection(neutral.index)]
    return m.sub(neutral.loc[m.index], axis=0)


def _crossed_bootstrap(drop_m, rise_m, n_boot: int, seed: int, statistic):
    """Resample image rows AND sentence columns of both matrices, independently, with replacement.

    Returns the bootstrap distribution of `statistic(drop_sample, rise_sample)` where each sample is
    the mean over the resampled sub-grid. Resampling both axes propagates image AND sentence
    variance; resampling rows only reproduces the published (image-clustered) interval.
    """
    rng = np.random.default_rng(seed)
    D, R = drop_m.to_numpy(dtype=float), rise_m.to_numpy(dtype=float)
    out = np.empty(n_boot)
    for k in range(n_boot):
        di = rng.integers(0, D.shape[0], D.shape[0]); dj = rng.integers(0, D.shape[1], D.shape[1])
        ri = rng.integers(0, R.shape[0], R.shape[0]); rj = rng.integers(0, R.shape[1], R.shape[1])
        out[k] = statistic(np.nanmean(D[np.ix_(di, dj)]), np.nanmean(R[np.ix_(ri, rj)]))
    return out


def _image_only_bootstrap(drop_m, rise_m, n_boot: int, seed: int, statistic):
    """Published-style interval: resample image rows only, sentences held fixed."""
    rng = np.random.default_rng(seed)
    D, R = drop_m.to_numpy(dtype=float), rise_m.to_numpy(dtype=float)
    out = np.empty(n_boot)
    for k in range(n_boot):
        di = rng.integers(0, D.shape[0], D.shape[0])
        ri = rng.integers(0, R.shape[0], R.shape[0])
        out[k] = statistic(np.nanmean(D[di, :]), np.nanmean(R[ri, :]))
    return out


def mirror_contrast(d: pd.DataFrame, value: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """|drop| - |rise| on read-out `value`, with image-only AND crossed intervals side by side."""
    drop_m = _effect_matrix(d, "positive", "negative", value)   # negative sentence on positive image
    rise_m = _effect_matrix(d, "negative", "positive", value)   # positive sentence on negative image
    if drop_m is None or rise_m is None:
        return {}
    drop, rise = float(np.nanmean(drop_m.to_numpy())), float(np.nanmean(rise_m.to_numpy()))
    stat = lambda a, b: abs(a) - abs(b)
    img_ci = np.percentile(_image_only_bootstrap(drop_m, rise_m, n_boot, seed, stat), [2.5, 97.5])
    crs_ci = np.percentile(_crossed_bootstrap(drop_m, rise_m, n_boot, seed, stat), [2.5, 97.5])
    return {
        "readout": value, "drop": drop, "rise": rise,
        "asymmetry_index": abs(drop) - abs(rise),
        "ratio_abs_drop_over_rise": abs(drop) / abs(rise) if rise else float("inf"),
        "ci95_image_only": [float(x) for x in img_ci],
        "ci95_crossed": [float(x) for x in crs_ci],
        "crossed_clears_zero": bool(crs_ci[0] > 0),
        "n_images": [int(drop_m.shape[0]), int(rise_m.shape[0])],
        "n_sentences": [int(drop_m.shape[1]), int(rise_m.shape[1])],
    }


def override_gap(d: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Override gap with a crossed interval. Cell value is the argmax-category override indicator.

    Identical in definition to `analyze_stage_f._flip_override` (a conflict trial is an override when
    the argmax emotion's valence category follows the text), but the sentence axis is kept so the
    interval can also resample sentences.
    """
    cat = np.array(["pos" if w in _POSITIVE else "neg" if w in _NEGATIVE else "other"
                    for w in EMOTION_LABELS])
    e = d.copy()
    e["_cat"] = cat[e[LP].to_numpy().argmax(axis=1)]

    def indicator(group, cond, win):
        sub = e[(e["image_group"] == group) & (e["condition"] == cond)]
        if sub.empty:
            return None
        return sub.assign(_hit=(sub["_cat"] == win).astype(float)).pivot_table(
            index="_img", columns="context_id", values="_hit", aggfunc="mean")

    neg_m = indicator("positive", "negative", "neg")   # negative sentence overrides a positive image
    pos_m = indicator("negative", "positive", "pos")   # positive sentence overrides a negative image
    if neg_m is None or pos_m is None:
        return {}
    neg_ov, pos_ov = float(np.nanmean(neg_m.to_numpy())), float(np.nanmean(pos_m.to_numpy()))
    stat = lambda a, b: a - b
    img_ci = np.percentile(_image_only_bootstrap(neg_m, pos_m, n_boot, seed, stat), [2.5, 97.5])
    crs_ci = np.percentile(_crossed_bootstrap(neg_m, pos_m, n_boot, seed, stat), [2.5, 97.5])
    return {
        "neg_ctx_overrides_pos_img": neg_ov, "pos_ctx_overrides_neg_img": pos_ov,
        "dominance_gap": neg_ov - pos_ov,
        "ci95_image_only": [float(x) for x in img_ci],
        "ci95_crossed": [float(x) for x in crs_ci],
        "crossed_clears_zero": bool(crs_ci[0] > 0 or crs_ci[1] < 0),
        "n_sentences": [int(neg_m.shape[1]), int(pos_m.shape[1])],
    }


# --------------------------------------------------------------------------- driver
def run(parquets: dict[str, Path], n_boot: int = 2000, seed: int = 0) -> dict:
    ensure_dirs()
    out: dict = {"run": run_stamp(), "git": git_hash(), "n_boot": n_boot, "seed": seed,
                 "readout_note": ("margin = logsumexp(lp_pos) - logsumexp(lp_neg) = log P(pos)/P(neg); "
                                  "unbounded log-odds. Margin contrasts are NOT in valence units."),
                 "models": {}}
    for model, pq in parquets.items():
        if not pq.exists():
            print(f"  [skip] {pq} not found");  continue
        df = pd.read_parquet(pq)
        if not set(LP).issubset(df.columns):
            print(f"  [skip] {pq.name} lacks lp_* columns");  continue
        d = add_readouts(df)
        d = d[d["condition"] != "none"]           # no-context excluded, as in the published analysis
        rec = {"parquet": str(pq),
               "bounded_valence": mirror_contrast(d, "valence", n_boot, seed),
               "unbounded_margin": mirror_contrast(d, "margin", n_boot, seed),
               "override": override_gap(d, n_boot, seed)}
        out["models"][model] = rec

        b, u, o = rec["bounded_valence"], rec["unbounded_margin"], rec["override"]
        print(f"\n  {model}")
        if b:
            print(f"    bounded valence   |drop|-|rise| {b['asymmetry_index']:+.3f}  "
                  f"ratio {b['ratio_abs_drop_over_rise']:.2f}  "
                  f"img-only [{b['ci95_image_only'][0]:+.3f},{b['ci95_image_only'][1]:+.3f}]  "
                  f"crossed [{b['ci95_crossed'][0]:+.3f},{b['ci95_crossed'][1]:+.3f}]")
        if u:
            print(f"    unbounded margin  |drop|-|rise| {u['asymmetry_index']:+.3f} log-odds  "
                  f"ratio {u['ratio_abs_drop_over_rise']:.2f}  "
                  f"crossed [{u['ci95_crossed'][0]:+.3f},{u['ci95_crossed'][1]:+.3f}]  "
                  f"{'CLEARS 0' if u['crossed_clears_zero'] else 'straddles 0'}")
        if o:
            print(f"    override gap      {o['dominance_gap']:+.1%}  "
                  f"img-only [{o['ci95_image_only'][0]:+.1%},{o['ci95_image_only'][1]:+.1%}]  "
                  f"crossed [{o['ci95_crossed'][0]:+.1%},{o['ci95_crossed'][1]:+.1%}]")

    path = STAGE_F_DIR / "unbounded_crossed.json"
    save_json(out, path)
    print(f"\n  data -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gemma", type=Path, default=STAGE_F_DIR / "conflict_pilot.parquet")
    ap.add_argument("--qwen", type=Path, default=STAGE_F_DIR / "conflict_qwen.parquet")
    ap.add_argument("--llava", type=Path, default=STAGE_F_DIR / "conflict_llava.parquet")
    ap.add_argument("--minimal", type=Path, default=None, help="optional minimal-pair bank parquet")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pqs = {"Gemma-3-4B": a.gemma, "Qwen3-VL-8B": a.qwen, "LLaVA-1.5-7B": a.llava}
    if a.minimal:
        pqs["Gemma-3-4B (minimal pairs)"] = a.minimal
    run(pqs, n_boot=a.n_boot, seed=a.seed)


if __name__ == "__main__":
    main()
