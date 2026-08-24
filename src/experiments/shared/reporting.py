"""Pure scientific metrics and reporting transformations shared by experiment runners."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ...data.labels import EMOTION_LABELS, EMOTIC_TO_SHARED


POSITIVE_LABELS = ("joy", "pride", "relief", "trust")
NEGATIVE_LABELS = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")

# Keep the original formula-local names while the public constants describe their meaning.
_POSITIVE = POSITIVE_LABELS
_NEGATIVE = NEGATIVE_LABELS


def correlation(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = np.isfinite(pred) & np.isfinite(target)
    if finite.sum() < 3 or np.std(pred[finite]) == 0 or np.std(target[finite]) == 0:
        return {"n": int(finite.sum()), "pearson": None, "spearman": None}
    return {
        "n": int(finite.sum()),
        "pearson": float(pearsonr(pred[finite], target[finite])[0]),
        "spearman": float(spearmanr(pred[finite], target[finite])[0]),
    }


_corr = correlation


def shared_emotic_label(categories) -> str | None:
    """Row-aligned EMOTIC-26 -> shared-7 collapse (single-label; else None).

    Mirrors data.emotic.to_shared_single_label's logic but keeps row alignment with the
    activation matrix (no reindexing), so labels line up with X_img.
    """
    mapped = set()
    for c in np.atleast_1d(categories):
        v = EMOTIC_TO_SHARED.get(str(c).strip())
        if v is not None:
            mapped.add(v)
    return next(iter(mapped)) if len(mapped) == 1 else None


def polarity_vector(shared_labels, positive, negative):
    """shared-7 label -> +1 (positive emotion) / 0 (negative) / NaN (excluded)."""
    pos, neg = set(positive), set(negative)
    out = np.full(len(shared_labels), np.nan)
    for i, lab in enumerate(shared_labels):
        if lab in pos:
            out[i] = 1.0
        elif lab in neg:
            out[i] = 0.0
    return out


def polarity_auc(pred, polarity):
    """AUC of a read-out separating positive- vs negative-emotion images (scale-free)."""
    from sklearn.metrics import roc_auc_score

    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(polarity)
    y = polarity[m]
    if m.sum() < 10 or y.sum() == 0 or y.sum() == m.sum():
        return {"n": int(m.sum()), "n_pos": int(np.nansum(polarity == 1)),
                "n_neg": int(np.nansum(polarity == 0)), "auc": None}
    return {"n": int(m.sum()), "n_pos": int(y.sum()), "n_neg": int(m.sum() - y.sum()),
            "auc": float(roc_auc_score(y, pred[m]))}


def random_direction_controls(X, y, n_random, seed):
    """Null distribution of |spearman| for random directions vs y (direction specificity).

    A correlation is scale-invariant, so direction norm is irrelevant — we skip
    norm-matching. Gemma's activations are anisotropic, so random directions have a
    non-trivial |spearman| spread; this returns the whole spread so the caller can compute
    an empirical p-value rather than eyeball a single max. `_abs` is the raw per-draw list.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    spears = []
    for _ in range(n_random):
        r = rng.standard_normal(d).astype(np.float32)
        c = _corr(X @ r, y)
        if c["spearman"] is not None:
            spears.append(abs(c["spearman"]))
    a = np.asarray(spears, dtype=float)
    if a.size == 0:
        return {"n_random": n_random, "n_valid": 0, "mean": None, "std": None,
                "max": None, "p95": None, "_abs": []}
    return {"n_random": n_random, "n_valid": int(a.size), "mean": float(a.mean()),
            "std": float(a.std()), "max": float(a.max()),
            "p95": float(np.percentile(a, 95)), "_abs": a.tolist()}


def transfer_verdict(metrics, appraisals):
    """Provisional READ-OUT verdict (human confirms). Requires agreement across three
    independent signals — valence correlation, polarity AUC, and beating the random-
    direction null — and a mirror sign from unpleasantness when scored. Note this is a
    read-out verdict only; shared-geometry vs verbalization needs the caption baseline.
    """
    if "pleasantness" not in appraisals:
        return "inconclusive (pleasantness not scored)"
    p = metrics["image_readout"]["pleasantness"]
    sp = p["vs_valence"]["spearman"]
    auc = p["polarity_auc"]["auc"]
    pval = p.get("vs_control_p")
    if sp is None:
        return "inconclusive (no valence signal computed)"
    beats = pval is not None and pval < 0.05          # beats the random-direction null
    concord = auc is not None and auc >= 0.60          # polarity agrees
    mirror = True
    if "unpleasantness" in appraisals:
        us = metrics["image_readout"]["unpleasantness"]["vs_valence"]["spearman"]
        mirror = us is not None and us < 0             # opposite sign, as theory predicts
    if abs(sp) >= 0.3 and beats and concord and mirror:
        return (f"supports read-out transfer (pleasantness rho={sp:+.2f} vs valence, "
                f"polarity AUC={auc:.2f}, beats random dirs p={pval:.3f}, unpleasantness "
                f"mirrors) — NEXT: caption baseline to separate shared-geometry vs verbalization")
    if abs(sp) >= 0.15 and beats:
        return ("inconclusive (above the random null but modest/mixed — scale to full split "
                "and add the caption baseline)")
    return "fails to support transfer (read-out indistinguishable from random directions)"


def flip_override(df, n_boot: int = 2000, seed: int = 0) -> dict:
    """Cross-model-comparable OVERRIDE rate from the argmax emotion's valence category.

    Calibration-free (uses WHICH emotion the model picks, not the continuous score — Gemma's valence
    is negatively skewed and Qwen's saturates, so a raw sign threshold is not comparable). A conflict
    forward is an OVERRIDE when the context wins the valence category against the image:
      * positive image + negative context → argmax emotion is NEGATIVE-valence
      * negative image + positive context → argmax emotion is POSITIVE-valence
    Aggregated per image (mean over the context bank) and bootstrapped over images (clustered CI).
    Negativity dominance = neg-context override rate > pos-context override rate.
    """
    lp = [f"lp_{w}" for w in EMOTION_LABELS]
    if not set(lp).issubset(df.columns):
        return {}
    cat = {w: ("pos" if w in _POSITIVE else "neg" if w in _NEGATIVE else "other") for w in EMOTION_LABELS}
    d = df.copy()
    d["_cat"] = [cat[EMOTION_LABELS[i]] for i in d[lp].to_numpy().argmax(axis=1)]

    def per_image(group, cond, win):
        g = d[(d["image_group"] == group) & (d["condition"] == cond)]
        return g.groupby("image_path")["_cat"].apply(lambda s: float((s == win).mean())).to_numpy()

    pn = per_image("positive", "negative", "neg")   # neg ctx overrides a positive image
    np_ = per_image("negative", "positive", "pos")  # pos ctx overrides a negative image
    if not len(pn) or not len(np_):
        return {}
    neg_ov, pos_ov = float(pn.mean()), float(np_.mean())
    rng = np.random.default_rng(seed)
    boot = np.array([pn[rng.integers(0, len(pn), len(pn))].mean()
                     - np_[rng.integers(0, len(np_), len(np_))].mean() for _ in range(n_boot)])
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    return {"neg_ctx_overrides_pos_img": neg_ov, "pos_ctx_overrides_neg_img": pos_ov,
            "dominance_gap": neg_ov - pos_ov, "dominance_gap_ci95": ci,
            "n_pos_images": int(len(pn)), "n_neg_images": int(len(np_))}


def minimal_pair_asymmetry(df, n_boot: int = 2000, seed: int = 0) -> dict:
    """Within-item, within-EVENT valence asymmetry for the minimal-pair bank (context_id `mp{i}`).

    The minimal-pair control holds each event constant and flips only the valence word, so for a given
    image a pair's negative and positive members differ ONLY in valence. This is the tightest test of
    negativity dominance available: for each image and each pair, compare the negative member's effect
    to the positive member's effect vs that image's own NEUTRAL baseline —
        Δneg = v(neg member) − v(neutral),  Δpos = v(pos member) − v(neutral),  d = |Δneg| − |Δpos|.
    Reported on the positive-image group (the clean cell, symmetric head-room). The contrast is PAIRED
    (same photo, same event), so we test it with a Wilcoxon signed-rank over images and a clustered
    bootstrap over images — stronger than the across-item aggregate because event and image are held
    fixed. Returns {} for non-minimal banks (no `mp*` context ids).
    """
    if "context_id" not in df.columns:
        return {}
    pairs = sorted({c for c in df["context_id"].dropna().unique()
                    if isinstance(c, str) and c.startswith("mp")})
    if not pairs:
        return {}
    try:  # short "won ↔ lost" labels; optional (analyzer also runs in the probe-free Qwen venv)
        from ...data.conflict_contexts import MINIMAL_PAIRS
        swap = {f"mp{i}": s for i, (_, _, s) in enumerate(MINIMAL_PAIRS)}
    except Exception:
        swap = {}

    g = df[df["image_group"] == "positive"]
    neu = g[g["condition"] == "neutral"].groupby("image_path")["valence"].mean()
    if neu.empty:
        return {}

    per_pair, img_asym = {}, {}   # img_asym: image_path -> [d over pairs]
    for cid in pairs:
        sub = g[g["context_id"] == cid]
        posv = sub[sub["condition"] == "positive"].groupby("image_path")["valence"].mean()
        negv = sub[sub["condition"] == "negative"].groupby("image_path")["valence"].mean()
        dneg_l, dpos_l = [], []
        for img in neu.index:
            if img in posv.index and img in negv.index:
                dn, dp = float(negv[img] - neu[img]), float(posv[img] - neu[img])
                dneg_l.append(dn); dpos_l.append(dp)
                img_asym.setdefault(img, []).append(abs(dn) - abs(dp))
        if dneg_l:
            mn, mp = float(np.mean(dneg_l)), float(np.mean(dpos_l))
            per_pair[cid] = {"swap": swap.get(cid, ""), "neg_effect": mn, "pos_effect": mp,
                             "asymmetry": abs(mn) - abs(mp),
                             "ratio": abs(mn) / abs(mp) if mp != 0 else float("inf"),
                             "n_images": len(dneg_l)}
    imgs = sorted(img_asym)
    per_img = np.array([np.mean(img_asym[i]) for i in imgs])
    if not len(per_img):
        return {}
    rng = np.random.default_rng(seed)
    boot = np.array([per_img[rng.integers(0, len(per_img), len(per_img))].mean()
                     for _ in range(n_boot)])
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]
    wp = None
    try:
        from scipy.stats import wilcoxon
        if np.any(per_img != 0):
            wp = float(wilcoxon(per_img, alternative="greater").pvalue)
    except Exception:
        pass
    return {"image_group": "positive", "n_pairs": len(per_pair), "n_images": len(imgs),
            "paired_asymmetry": float(per_img.mean()), "ci95": ci, "wilcoxon_p_greater": wp,
            "per_pair": per_pair}


def cell_means(df, value):
    """2x2 (image_group x context polarity) means of `value`, plus no-context/neutral references."""
    out = {}
    for grp in ("positive", "negative"):
        for cond in ("none", "positive", "negative", "neutral"):
            cell = df[(df["image_group"] == grp) & (df["condition"] == cond)][value]
            if len(cell):
                out[f"{grp}_img/{cond}_ctx"] = float(cell.mean())
    return out


def asymmetry_vs_floor(df, n_boot: int = 2000, seed: int = 0) -> dict:
    """Is the positive-image + negative-context valence 'drop' a real negativity asymmetry, or just
    ceiling/floor geometry?

    Two competing accounts of the celebrated drop, with a sharp differential prediction:
      * CEILING/FLOOR — positive images sit near the valence ceiling, so a negative context has the
        most ROOM to pull them down; negative images sit near the floor, so a positive context has the
        most room to pull them up. Geometry predicts the two INCONGRUENT effects are equal in
        magnitude: |Δ(pos-img, neg-ctx)| ≈ |Δ(neg-img, pos-ctx)|, and the CONGRUENT (same-polarity)
        contexts barely move (already saturated).
      * NEGATIVITY ASYMMETRY — negative context wins BEYOND geometry: |drop| ≫ |rise|.

    All effects are per-image, vs that image's own NEUTRAL-context valence (the within-structure
    baseline; no-context is excluded upstream), averaged over the context bank. We report:
      - drop  = mean incongruent effect for positive images (negative context)   [expected < 0]
      - rise  = mean incongruent effect for negative images (positive context)   [expected > 0]
      - asymmetry_index = |drop| − |rise| with a bootstrap 95% CI over images (0 ⇒ symmetric geometry)
      - a Mann–Whitney test on the per-image |incongruent effect| between the two groups
      - a HEADROOM-NORMALIZED pull |effect| / (distance from the image's neutral valence to the
        empirical valence bound it is moving toward): if geometry alone drives the drop, the two
        normalized pulls match; a residual gap is the part not explained by the floor.
    """
    d = df.copy()
    # per-image block id: the base pass writes each image as [none, <contexts...>], so cumsum on the
    # `none` row uniquely labels each image (EMOTIC image_path recurs → cannot key on it).
    d["_img"] = (d["condition"] == "none").cumsum()
    # empirical valence bounds (robust) define the ceiling/floor the effects move toward.
    v_lo, v_hi = (float(x) for x in np.percentile(d["valence"].to_numpy(dtype=float), [5, 95]))

    def per_image(group, incong_cond, cong_cond):
        """One row per image in `group`: incongruent- and congruent-context effect vs its neutral."""
        recs = []
        for _, blk in d[d["image_group"] == group].groupby("_img"):
            neu = blk.loc[blk["condition"] == "neutral", "valence"]
            inc = blk.loc[blk["condition"] == incong_cond, "valence"]
            con = blk.loc[blk["condition"] == cong_cond, "valence"]
            if len(neu) and len(inc):
                recs.append({"neutral": float(neu.mean()),
                             "incong": float(inc.mean() - neu.mean()),
                             "cong": float(con.mean() - neu.mean()) if len(con) else np.nan})
        return pd.DataFrame(recs)

    pos = per_image("positive", "negative", "positive")  # drop (incong) + ceiling-saturating cong
    neg = per_image("negative", "positive", "negative")  # rise (incong) + floor-saturating cong
    if pos.empty or neg.empty:
        return {"note": "insufficient positive/negative image cells to test asymmetry"}

    drop = float(pos["incong"].mean())   # ≤ 0
    rise = float(neg["incong"].mean())   # ≥ 0
    asym_index = abs(drop) - abs(rise)

    rng = np.random.default_rng(seed)
    pi, ni = pos["incong"].to_numpy(), neg["incong"].to_numpy()
    boot = np.empty(n_boot)
    for k in range(n_boot):
        bp = pi[rng.integers(0, len(pi), len(pi))]
        bn = ni[rng.integers(0, len(ni), len(ni))]
        boot[k] = abs(bp.mean()) - abs(bn.mean())
    ci = [float(x) for x in np.percentile(boot, [2.5, 97.5])]

    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(np.abs(pi), np.abs(ni), alternative="greater")  # H1: |drop| > |rise|

    # headroom-normalized pull: |effect| / room toward the bound it moves to.
    pos_room = (pos["neutral"] - v_lo).clip(lower=1e-6)
    neg_room = (v_hi - neg["neutral"]).clip(lower=1e-6)
    pull_drop = float((pos["incong"].abs() / pos_room).mean())
    pull_rise = float((neg["incong"].abs() / neg_room).mean())

    # verdict: symmetric (geometry) if the CI on |drop|−|rise| straddles 0.
    symmetric = ci[0] <= 0 <= ci[1]
    if symmetric:
        interp = ("CEILING/FLOOR — the drop and the mirror rise are statistically indistinguishable "
                  "in magnitude (asymmetry CI straddles 0); the 'sharp drop' is saturation geometry, "
                  "not a negativity-specific effect.")
    elif asym_index > 0:
        interp = (f"RESIDUAL NEGATIVITY ASYMMETRY — the drop exceeds the geometry-mirrored rise by "
                  f"{asym_index:+.3f} valence (CI {ci[0]:+.3f},{ci[1]:+.3f}); negative context moves "
                  f"valence more than the floor alone allows.")
    else:
        interp = (f"REVERSE ASYMMETRY — the positive-context rise exceeds the drop by "
                  f"{-asym_index:+.3f}; if anything positive context is stronger here.")

    return {
        "drop_pos_img_neg_ctx": drop, "rise_neg_img_pos_ctx": rise,
        "congruent_pos_img_pos_ctx": float(pos["cong"].mean()),
        "congruent_neg_img_neg_ctx": float(neg["cong"].mean()),
        "asymmetry_index": asym_index, "asymmetry_ci95": ci, "symmetric": bool(symmetric),
        "mannwhitney_u": float(mw.statistic), "mannwhitney_p_greater": float(mw.pvalue),
        "valence_bounds_p5_p95": [v_lo, v_hi],
        "headroom_norm_pull_drop": pull_drop, "headroom_norm_pull_rise": pull_rise,
        "headroom_norm_asymmetry": pull_drop - pull_rise,
        "n_pos_images": int(len(pos)), "n_neg_images": int(len(neg)),
        "interpretation": interp,
    }


def sequence_result_columns(result: dict) -> dict:
    summed = result["sequence_sum"]
    mean = result["content_mean"]
    columns = {"valence": summed["valence"], "valence_content_mean": mean["valence"]}
    for label in EMOTION_LABELS:
        columns[f"lp_{label}"] = summed["logprobs"][label]
        columns[f"lp_content_mean_{label}"] = mean["logprobs"][label]
        columns[f"score_sequence_sum_{label}"] = summed["scores"][label]
        columns[f"score_content_mean_{label}"] = mean["scores"][label]
    return columns


def content_mean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["valence"] = out["valence_content_mean"]
    for label in EMOTION_LABELS:
        out[f"lp_{label}"] = out[f"lp_content_mean_{label}"]
    return out


def image_discriminability(df: pd.DataFrame) -> dict:
    """Can the model still tell the two image groups apart with NO context? The resolution control.

    Uses only the `none` rows, so it is a pure vision read: if this collapses at low resolution the
    image simply became unreadable, and any change in the override gap is confounded with visual
    quality rather than attributable to the token budget.
    """
    d = df[df["condition"] == "none"]
    pos = d[d["image_group"] == "positive"]["valence"].to_numpy(dtype=float)
    neg = d[d["image_group"] == "negative"]["valence"].to_numpy(dtype=float)
    if not len(pos) or not len(neg):
        return {}
    # AUC via the Mann-Whitney U identity (no sklearn dependency in the raw-HF envs).
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return {"mean_valence_positive_images": float(pos.mean()),
            "mean_valence_negative_images": float(neg.mean()),
            "discriminability_gap": float(pos.mean() - neg.mean()),
            "auc": float(auc), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def text_only_readouts(df: pd.DataFrame) -> dict:
    """Both scales for a text-only run: the bounded valence score and an unbounded log-odds margin.

    The bounded score is the paper's primary readout, but with no image some models pin it to a
    bound: Qwen3-VL reads a bare neutral sentence as sadness at -0.996, so every sentence lands
    at +/-1 and the |neg|/|pos| ratio comes out at 1.00 no matter what the sentences do. A ratio
    computed from two saturated constants is not evidence that the sets are balanced; it is evidence
    that the readout cannot see. The margin (best positive-valence label minus best negative-valence
    label, in log-odds) has no bounds and answers the question the control is actually asking.

    Two references are reported per scale. `raw` compares each polarity against zero and is the one
    to trust when the neutral baseline is itself pinned; `vs_neutral` subtracts the neutral-sentence
    baseline and is the one to trust when it is not. `saturation` says which situation you are in.
    """
    d = df.copy()
    d["margin"] = [max(r["lp_" + w] for w in _POSITIVE) - max(r["lp_" + w] for w in _NEGATIVE)
                   for _, r in d.iterrows()]
    out = {"saturation_frac": float((d["valence"].abs() >= 0.999).mean()), "n_rows": int(len(d))}
    for scale, col in (("bounded_valence", "valence"), ("unbounded_margin", "margin")):
        g = lambda c: d[d["condition"] == c][col]
        pos, neg, neu = float(g("positive").mean()), float(g("negative").mean()), float(g("neutral").mean())
        pe, ne = pos - neu, neg - neu
        out[scale] = {
            "pos_raw": pos, "neg_raw": neg, "neutral_baseline": neu,
            "ratio_raw": abs(neg) / abs(pos) if pos else float("nan"),
            "pos_vs_neutral": pe, "neg_vs_neutral": ne,
            "ratio_vs_neutral": abs(ne) / abs(pe) if pe else float("nan"),
            "mirror_contrast_raw": abs(neg) - abs(pos),
        }
    return out


def token_budget_trends(tab: pd.DataFrame) -> dict:
    """Within-model and cross-model evidence on the token-budget hypothesis, kept strictly separate.

    A single correlation over every row is worse than useless here, for three reasons this function
    exists to avoid:

      1. POOLING. A within-model resolution sweep (same weights, budget varied) and a cross-model
         comparison (everything varies at once) answer different questions. Pooling them lets three
         clustered points from one model plus one distant point from another produce r = -0.99 that
         means nothing.
      2. IGNORED UNCERTAINTY. Correlation uses point estimates only. Four gaps that sit inside
         mutually overlapping 95% CIs are a flat line, however neatly ordered they happen to be.
      3. SILENT EXCLUSION. Runs from the older fixed-path runner carry no image-token count and drop
         out of any correlation without comment — and those were exactly the models that break the
         pattern. Missing rows are now named, not dropped quietly.

    Within-model verdict is CI-based: if every sweep CI shares a common intersection, no effect of the
    budget is detectable regardless of the ordering of the point estimates.
    """
    out: dict = {}
    have = tab.dropna(subset=["image_tokens", "override_gap"])
    missing = tab[tab["image_tokens"].isna()]
    if len(missing):
        out["excluded_missing_image_tokens"] = [
            {"model": r["model"], "source": r["source"], "override_gap": r["override_gap"]}
            for _, r in missing.iterrows()]

    # ---- within-model: one entry per model that was swept over >=2 distinct budgets
    within = []
    for model, g in have.groupby("model"):
        if g["image_tokens"].nunique() < 2:
            continue
        lo, hi = float(g["ci_lo"].max()), float(g["ci_hi"].min())   # common intersection of all CIs
        flat = lo <= hi
        within.append({
            "model": model, "n_runs": int(len(g)),
            "tokens_min": float(g["image_tokens"].min()), "tokens_max": float(g["image_tokens"].max()),
            "fold_range": float(g["image_tokens"].max() / g["image_tokens"].min()),
            "gap_min": float(g["override_gap"].min()), "gap_max": float(g["override_gap"].max()),
            "all_cis_overlap": bool(flat),
            "discriminability_auc_range": [float(g["auc"].min()), float(g["auc"].max())]
                                          if g["auc"].notna().any() else None,
            "verdict": ("FLAT — every CI overlaps, so no effect of the token budget is detectable "
                        "over this range" if flat else
                        "MOVES — at least two CIs are disjoint across the budget range")})
    if within:
        out["within_model"] = within

    # ---- cross-model: exactly one representative run per model (the largest budget measured)
    reps = have.sort_values("image_tokens").groupby("model", as_index=False).last()
    if len(reps) >= 3:
        t, gp = reps["image_tokens"].to_numpy(float), reps["override_gap"].to_numpy(float)
        r = float(np.corrcoef(t, gp)[0, 1])
        loo = [float(np.corrcoef(np.delete(t, i), np.delete(gp, i))[0, 1]) for i in range(len(t))]
        out["cross_model"] = {
            "n_models": int(len(reps)), "models": reps["model"].tolist(),
            "pearson_tokens_vs_gap": r, "leave_one_out_range": [min(loo), max(loo)],
            "caveat": (f"n={len(reps)} models: a correlation this small is descriptive only, ignores "
                       f"every CI, and confounds the budget with everything else that differs "
                       f"between checkpoints."
                       + (" Runs are missing image-token counts (see excluded_missing_image_tokens), "
                          "so this omits models that may break the pattern."
                          if "excluded_missing_image_tokens" in out else ""))}
    elif len(reps):
        out["cross_model"] = {"n_models": int(len(reps)),
                              "note": "fewer than 3 models with measured image tokens — no trend reported"}
    return out
