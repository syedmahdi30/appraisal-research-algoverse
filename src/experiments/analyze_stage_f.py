"""Stage F — analysis + decision D2 (pilot-plan-stage-e-f.md).

CPU-only. Consumes results/stage_f/conflict_pilot.parquet (base pass) and, if present,
arbitration_pilot.parquet (steering pass).

Base: standardized OLS of each read-out on the image and text cues —
  probe_readout ~ z(image_valence) + text_code   (text_code = -1/0/+1)
  valence       ~ z(image_valence) + text_code
Reports standardized β_img, β_txt and the dominance ratio |β_txt|/|β_img| for each; any stable
dominance pattern (image-led / text-led / mixed) is a finding.

Arbitration: mean Δ behavioral valence vs β on the incongruent cells (Δ vs the β=0 baseline of
the same cell) and its slope, compared to the Stage D single-direction slope (~0.33). The probe
read-out slope is also reported — expected ~0 because attn_out L18 is upstream of the resid_post
L18 injection (recorded to demonstrate, not to score).

D2: SIGNAL if both β sign-correct and incongruent cells separate from congruent; NULL if the
context leaves both read-outs unchanged (retry a stronger, person-naming context).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.labels import EMOTION_LABELS
from ..paths import FIGURES_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, run_stamp, save_json

STAGE_D_SLOPE = 0.33  # Stage D pleasantness single-direction Δvalence slope (reference)


def _z(x):
    x = np.asarray(x, dtype=float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def _std_ols(y, cols: dict) -> dict:
    """Standardized OLS slopes of y on the given predictor columns (all z-scored)."""
    ys = _z(y)
    X = np.column_stack([np.ones(len(ys))] + [_z(v) for v in cols.values()])
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    return {name: float(b) for name, b in zip(cols, beta[1:])}


def _dominance(y, df) -> dict:
    b = _std_ols(y, {"image": df["image_valence"].to_numpy(), "text": df["text_code"].to_numpy()})
    ratio = abs(b["text"]) / abs(b["image"]) if b["image"] != 0 else float("inf")
    # 0.8-1.25 = the two cues contribute comparably (the shared-representation reading);
    # outside that band one modality clearly leads.
    lead = "text-led" if ratio > 1.25 else "image-led" if ratio < 0.8 else "balanced/integrated"
    return {"beta_img": b["image"], "beta_txt": b["text"], "dominance_ratio": ratio, "lead": lead}


def _condition_breakdown(df) -> dict:
    """RAW per-(image group × condition) means for both read-outs, with per-context_id detail.

    Diagnoses the 'vs no-context' anomaly (e.g. a neutral context appearing to raise valence): shows
    whether it is the `none` baseline that is odd vs `neutral`, whether it appears in the internal
    probe read-out or only behavioral valence, and whether one specific context drives it.
    """
    out = {"by_condition": {}, "by_context_id": {}}
    for grp in ("positive", "negative"):
        g = df[df["image_group"] == grp]
        for cond in ("none", "neutral", "positive", "negative"):
            cell = g[g["condition"] == cond]
            if len(cell):
                out["by_condition"][f"{grp}/{cond}"] = {
                    "n": int(len(cell)), "valence": float(cell["valence"].mean()),
                    "probe": float(cell["probe_readout"].mean())}
    if "context_id" in df.columns:
        for grp in ("positive", "negative"):
            g = df[df["image_group"] == grp]
            for cid, cell in g.groupby("context_id"):
                out["by_context_id"][f"{grp}/{cid}"] = {
                    "n": int(len(cell)), "valence": float(cell["valence"].mean()),
                    "probe": float(cell["probe_readout"].mean()),
                    "context": str(cell["context"].iloc[0])}
    return out


def _cell_means(df, value):
    """2x2 (image_group x context polarity) means of `value`, plus no-context/neutral references."""
    out = {}
    for grp in ("positive", "negative"):
        for cond in ("none", "positive", "negative", "neutral"):
            cell = df[(df["image_group"] == grp) & (df["condition"] == cond)][value]
            if len(cell):
                out[f"{grp}_img/{cond}_ctx"] = float(cell.mean())
    return out


def _asymmetry_vs_floor(df, n_boot: int = 2000, seed: int = 0) -> dict:
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


def _arbitration(df, betas):
    """Mean Δ (valence, probe) vs β on incongruent cells, relative to each cell's β=0 baseline.

    Aligns each steered row to its own baseline by CELL, not image_path: EMOTIC is per-person, so
    the same image_path recurs and cannot key the baseline. Rows are written per cell as
    [β=0, then the betas], so `(beta==0).cumsum()` labels each cell uniquely.
    """
    df = df.copy()
    df["_cell"] = (df["beta"] == 0).cumsum()
    base = df[df["beta"] == 0].set_index("_cell")
    res = {"valence": {}, "probe": {}}
    for b in betas:
        sub = df[df["beta"] == b]
        dv = sub["valence"].to_numpy() - base.loc[sub["_cell"], "valence"].to_numpy()
        dp = sub["probe_readout"].to_numpy() - base.loc[sub["_cell"], "probe_readout"].to_numpy()
        res["valence"][int(b)] = float(np.mean(dv))
        res["probe"][int(b)] = float(np.mean(dp))
    xs = sorted(res["valence"])
    res["valence_slope"] = float(np.polyfit(xs, [res["valence"][b] for b in xs], 1)[0])
    res["probe_slope"] = float(np.polyfit(xs, [res["probe"][b] for b in xs], 1)[0])
    return res


def run(config_path: str | None = None) -> dict:
    ensure_dirs()
    base_pq = STAGE_F_DIR / "conflict_pilot.parquet"
    if not base_pq.exists():
        raise FileNotFoundError(f"{base_pq} missing — run stage_f_conflict (base pass) first.")
    df = pd.read_parquet(base_pq)

    # The no-context prompt is structurally non-comparable: adding ANY framing sentence (even a
    # neutral one) raises both the probe read-out and behavioral valence (see _condition_breakdown).
    # So dominance and the context effect use only the SENTENCE-BEARING conditions, with NEUTRAL as
    # the within-structure baseline; no-context is retained in the breakdown for transparency.
    df_ctx = df[df["condition"] != "none"].copy()
    dom = {"probe_readout": _dominance(df_ctx["probe_readout"], df_ctx),
           "valence": _dominance(df_ctx["valence"], df_ctx)}
    cells = {"probe_readout": _cell_means(df, "probe_readout"),
             "valence": _cell_means(df, "valence")}

    # context effect vs the NEUTRAL context (within-structure baseline), per image group.
    ctx_effect = {}
    for grp in ("positive", "negative"):
        g = df[df["image_group"] == grp]
        base = g[g["condition"] == "neutral"]["valence"].mean()
        ctx_effect[grp] = {c: float(g[g["condition"] == c]["valence"].mean() - base)
                           for c in ("positive", "negative")}

    breakdown = _condition_breakdown(df)
    asymmetry = _asymmetry_vs_floor(df)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "n_rows": int(len(df)),
        "n_images": int(df["image_path"].nunique()),
        "dominance": dom, "dominance_excludes_no_context": True,
        "cell_means": cells, "context_effect_vs_neutral": ctx_effect,
        "asymmetry_vs_floor": asymmetry,
        "condition_breakdown": breakdown,
    }

    arb_pq = STAGE_F_DIR / "arbitration_pilot.parquet"
    arb = None
    if arb_pq.exists():
        adf = pd.read_parquet(arb_pq)
        betas = sorted(int(b) for b in adf["beta"].unique() if b != 0)
        arb = _arbitration(adf, betas)
        arb["within_50pct_of_stage_d"] = bool(
            0.5 * STAGE_D_SLOPE <= abs(arb["valence_slope"]) <= 1.5 * STAGE_D_SLOPE)
        metrics["arbitration"] = arb

    metrics["verdict"] = _verdict(dom, ctx_effect, arb)
    save_json(metrics, STAGE_F_DIR / "conflict_analysis.json")
    _plot(cells, arb)

    print(f"\nStage F analysis — {metrics['n_images']} images, {metrics['n_rows']} base rows "
          f"(dominance on sentence-bearing conditions only; no-context excluded).\n")
    for ro in ("probe_readout", "valence"):
        d = dom[ro]
        print(f"  {ro:14s} β_img={d['beta_img']:+.3f}  β_txt={d['beta_txt']:+.3f}  "
              f"|txt|/|img|={d['dominance_ratio']:.2f}  -> {d['lead']}")
    print("\n  context effect on behavioral valence (vs the NEUTRAL context):")
    for grp, e in ctx_effect.items():
        print(f"    {grp:8s} img: +ctx {e['positive']:+.3f}  -ctx {e['negative']:+.3f}")
    if "drop_pos_img_neg_ctx" in asymmetry:
        a = asymmetry
        print("\n  ASYMMETRY vs FLOOR (is the pos-img+neg-ctx drop real, or ceiling/floor?):")
        print(f"    drop  (pos img, neg ctx) = {a['drop_pos_img_neg_ctx']:+.3f}   "
              f"congruent (pos img, pos ctx) = {a['congruent_pos_img_pos_ctx']:+.3f}")
        print(f"    rise  (neg img, pos ctx) = {a['rise_neg_img_pos_ctx']:+.3f}   "
              f"congruent (neg img, neg ctx) = {a['congruent_neg_img_neg_ctx']:+.3f}")
        print(f"    |drop|-|rise| = {a['asymmetry_index']:+.3f}  "
              f"CI95 [{a['asymmetry_ci95'][0]:+.3f}, {a['asymmetry_ci95'][1]:+.3f}]  "
              f"MW p(|drop|>|rise|)={a['mannwhitney_p_greater']:.3f}")
        print(f"    headroom-normalized pull: drop {a['headroom_norm_pull_drop']:.3f} vs "
              f"rise {a['headroom_norm_pull_rise']:.3f}  (Δ {a['headroom_norm_asymmetry']:+.3f})")
        print(f"    → {a['interpretation']}")
    print("\n  RAW per-condition means (diagnose the vs-none anomaly; is `none` or `neutral` odd?):")
    print(f"    {'cell':18s} {'n':>4s} {'valence':>9s} {'probe':>9s}")
    for k, v in breakdown["by_condition"].items():
        print(f"    {k:18s} {v['n']:>4d} {v['valence']:>+9.3f} {v['probe']:>+9.3f}")
    if breakdown["by_context_id"]:
        print("\n  per-context detail (spot a single outlier context):")
        for k, v in sorted(breakdown["by_context_id"].items()):
            print(f"    {k:14s} val {v['valence']:+.3f}  probe {v['probe']:+.3f}  \"{v['context'][:44]}\"")
    if arb:
        print(f"\n  arbitration: behavioral valence slope {arb['valence_slope']:+.3f} "
              f"(Stage D ~{STAGE_D_SLOPE}; within±50%: {arb['within_50pct_of_stage_d']})  |  "
              f"probe slope {arb['probe_slope']:+.3f} (expected ~0, upstream)")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  figure -> {FIGURES_DIR/'stage_f_conflict.png'}   "
          f"metrics -> {STAGE_F_DIR/'conflict_analysis.json'}")
    return metrics


def _verdict(dom, ctx_effect, arb) -> str:
    b = dom["valence"]
    img_ok = b["beta_img"] > 0
    txt_ok = b["beta_txt"] > 0
    # context must actually move behavioral valence in its own direction (pos>neutral>neg-ish)
    moved = any(abs(v) > 0.05 for e in ctx_effect.values() for v in e.values())
    if not moved and abs(b["beta_txt"]) < 0.05:
        return ("NULL (criterion: context leaves both read-outs unchanged — image dominates, text "
                "context ignored at L18). Re-run one STRONG person-naming context (conflict_contexts."
                "STRONG_CONTEXT); if still flat, that image-dominance is itself the reportable finding.")
    if img_ok and txt_ok:
        base = (f"SIGNAL — both cues sign-correct (β_img={b['beta_img']:+.2f}, "
                f"β_txt={b['beta_txt']:+.2f}; {b['lead']}, ratio {b['dominance_ratio']:.2f}). "
                f"Image and text both write valence into the shared read-out"
                + ("; the two are comparable (shared-representation reading). " if b['lead'].startswith('balanced')
                   else f"; {b['lead']}. "))
        if arb is not None:
            if arb["within_50pct_of_stage_d"]:
                return base + (f"Steering arbitrates the conflict: behavioral-valence slope "
                               f"{arb['valence_slope']:+.2f} (~{abs(arb['valence_slope'])/STAGE_D_SLOPE:.0%} "
                               f"of Stage D's {STAGE_D_SLOPE}).")
            return base + (f"But arbitration is weak (steering slope {arb['valence_slope']:+.2f} vs "
                           f"Stage D ~{STAGE_D_SLOPE}).")
        return base + "Run --arbitrate to test whether steering arbitrates the conflict."
    return (f"INCONCLUSIVE (β_img={b['beta_img']:+.2f}, β_txt={b['beta_txt']:+.2f}; signs mixed) — "
            f"human call; consider stronger contexts or more images.")


def _plot(cells, arb):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = 3 if arb else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    for ax, ro in zip(axes[0], ("probe_readout", "valence")):
        groups = ("positive", "negative")
        conds = ("positive", "negative")
        x = np.arange(len(groups))
        for j, cond in enumerate(conds):
            vals = [cells[ro].get(f"{g}_img/{cond}_ctx", np.nan) for g in groups]
            ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label=f"{cond} ctx")
        ax.set_xticks(x); ax.set_xticklabels([f"{g} img" for g in groups])
        ax.set_title(ro); ax.legend(fontsize=7); ax.axhline(0, color="gray", lw=0.5)
    if arb:
        ax = axes[0][2]
        xs = sorted(arb["valence"])
        ax.plot(xs, [arb["valence"][b] for b in xs], "-o", label="behavioral valence")
        ax.plot(xs, [arb["probe"][b] for b in xs], "k:", label="probe (upstream ~0)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"arbitration (slope {arb['valence_slope']:+.2f})")
        ax.set_xlabel("β  (pleasantness Δμ)"); ax.set_ylabel("Δ vs β=0"); ax.legend(fontsize=7)
    fig.suptitle("Stage F — modality conflict: image vs text cues, and steering arbitration")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_f_conflict.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — analysis + decision D2")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
