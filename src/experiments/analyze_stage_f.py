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
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import FIGURES_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, run_stamp, save_json
from .shared.reporting import (
    NEGATIVE_LABELS as _NEGATIVE,
    POSITIVE_LABELS as _POSITIVE,
    arbitration as _arbitration,
    asymmetry_vs_floor as _asymmetry_vs_floor,
    cell_means as _cell_means,
    flip_override as _flip_override,
    minimal_pair_asymmetry as _minimal_pair_asymmetry,
)

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
        # Group by (context_id, condition), NOT context_id alone: the minimal-pair bank shares one id
        # (mp{i}) across a pair's positive AND negative member, so keying on context_id alone would
        # average the two polarities together into a meaningless row. Splitting by condition keeps each
        # sentence separate; for the full bank (one condition per id) the condition suffix is redundant
        # but harmless.
        for grp in ("positive", "negative"):
            g = df[df["image_group"] == grp]
            for (cid, cond), cell in g.groupby(["context_id", "condition"]):
                out["by_context_id"][f"{grp}/{cid}/{cond}"] = {
                    "n": int(len(cell)), "valence": float(cell["valence"].mean()),
                    "probe": float(cell["probe_readout"].mean()),
                    "context": str(cell["context"].iloc[0])}
    return out


def run(config_path: str | None = None, base_pq=None) -> dict:
    ensure_dirs()
    # default = the full-bank base pass; `base_pq` (name or path) analyses an alternate bank, e.g.
    # conflict_minimal.parquet (T1.1). A custom parquet writes a stem-matched *_analysis.json so it
    # never clobbers the canonical conflict_analysis.json.
    if base_pq is None:
        base_pq = STAGE_F_DIR / "conflict_pilot.parquet"
        out_name, fig_name = "conflict_analysis.json", "stage_f_conflict.png"
    else:
        base_pq = Path(base_pq)
        if not base_pq.is_absolute():
            base_pq = STAGE_F_DIR / base_pq
        out_name, fig_name = f"{base_pq.stem}_analysis.json", f"{base_pq.stem}.png"
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
    flip = _flip_override(df)
    minimal = _minimal_pair_asymmetry(df)   # non-empty only for the minimal-pair bank (mp* ids)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "n_rows": int(len(df)),
        "n_images": int(df["image_path"].nunique()),
        "dominance": dom, "dominance_excludes_no_context": True,
        "cell_means": cells, "context_effect_vs_neutral": ctx_effect,
        "asymmetry_vs_floor": asymmetry, "flip_override": flip,
        "condition_breakdown": breakdown,
    }
    if minimal:
        metrics["minimal_pair_asymmetry"] = minimal

    # Arbitration is a Gemma-only steering sweep at a FIXED path, so attaching it to an arbitrary
    # base parquet silently reports Gemma's steering slope inside (say) a Qwen or LLaVA analysis —
    # and `_verdict` then repeats it as though it belonged to that model. Only attach it to the run
    # it was actually measured alongside.
    arb_pq = STAGE_F_DIR / "arbitration_pilot.parquet"
    arb = None
    if arb_pq.exists() and base_pq.stem in ("conflict_pilot", "conflict_minimal"):
        adf = pd.read_parquet(arb_pq)
        betas = sorted(int(b) for b in adf["beta"].unique() if b != 0)
        arb = _arbitration(adf, betas)
        arb["within_50pct_of_stage_d"] = bool(
            0.5 * STAGE_D_SLOPE <= abs(arb["valence_slope"]) <= 1.5 * STAGE_D_SLOPE)
        metrics["arbitration"] = arb

    metrics["verdict"] = _verdict(dom, ctx_effect, arb)
    metrics["source_parquet"] = base_pq.name
    save_json(metrics, STAGE_F_DIR / out_name)
    _plot(cells, arb, fig_name)

    print(f"\nStage F analysis — {metrics['n_images']} images, {metrics['n_rows']} base rows "
          f"(dominance on sentence-bearing conditions only; no-context excluded).\n")
    for ro in ("probe_readout", "valence"):
        d = dom[ro]
        print(f"  {ro:14s} β_img={d['beta_img']:+.3f}  β_txt={d['beta_txt']:+.3f}  "
              f"|txt|/|img|={d['dominance_ratio']:.2f}  -> {d['lead']}")
    print("\n  context effect on behavioral valence (vs the NEUTRAL context):")
    for grp, e in ctx_effect.items():
        print(f"    {grp:8s} img: +ctx {e['positive']:+.3f}  -ctx {e['negative']:+.3f}")
    if flip:
        print(f"\n  FLIP-RATE OVERRIDE (argmax-emotion category, cross-model comparable): "
              f"neg-ctx overrides positive image {flip['neg_ctx_overrides_pos_img']:.0%}  vs  "
              f"pos-ctx overrides negative image {flip['pos_ctx_overrides_neg_img']:.0%}  "
              f"(gap {flip['dominance_gap']:+.0%}, CI "
              f"[{flip['dominance_gap_ci95'][0]:+.0%},{flip['dominance_gap_ci95'][1]:+.0%}])")
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
    if minimal:
        m = minimal
        print("\n  MINIMAL-PAIR within-item asymmetry (positive images; same event, only the valence "
              "word flipped):")
        print(f"    {'pair':5s} {'swap':26s} {'Δneg':>8s} {'Δpos':>8s} {'|Δn|-|Δp|':>10s} {'ratio':>7s}")
        for cid, pp in sorted(m["per_pair"].items()):
            r = pp["ratio"]
            print(f"    {cid:5s} {pp['swap'][:26]:26s} {pp['neg_effect']:>+8.3f} {pp['pos_effect']:>+8.3f} "
                  f"{pp['asymmetry']:>+10.3f} {r:>7.2f}")
        wp = f", Wilcoxon p={m['wilcoxon_p_greater']:.3f}" if m["wilcoxon_p_greater"] is not None else ""
        print(f"    OVERALL paired |Δneg|-|Δpos| = {m['paired_asymmetry']:+.3f} "
              f"CI [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]{wp} "
              f"(n={m['n_images']} images x {m['n_pairs']} pairs, within-item paired)")
    print("\n  RAW per-condition means (diagnose the vs-none anomaly; is `none` or `neutral` odd?):")
    print(f"    {'cell':18s} {'n':>4s} {'valence':>9s} {'probe':>9s}")
    for k, v in breakdown["by_condition"].items():
        print(f"    {k:18s} {v['n']:>4d} {v['valence']:>+9.3f} {v['probe']:>+9.3f}")
    if breakdown["by_context_id"]:
        print("\n  per-context detail (spot a single outlier context):")
        for k, v in sorted(breakdown["by_context_id"].items()):
            print(f"    {k:24s} val {v['valence']:+.3f}  probe {v['probe']:+.3f}  \"{v['context'][:40]}\"")
    if arb:
        print(f"\n  arbitration: behavioral valence slope {arb['valence_slope']:+.3f} "
              f"(Stage D ~{STAGE_D_SLOPE}; within±50%: {arb['within_50pct_of_stage_d']})  |  "
              f"probe slope {arb['probe_slope']:+.3f} (expected ~0, upstream)")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  figure -> {FIGURES_DIR/fig_name}   metrics -> {STAGE_F_DIR/out_name}")
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


def _plot(cells, arb, fig_name: str = "stage_f_conflict.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = ("positive", "negative")
    conds = ("positive", "negative")

    # Only models whose internal states the probe was fit on carry a probe readout;
    # for the others the column is all-NaN by design (see stage_f_llava.py). Plotting
    # it anyway produced an empty axes box in the figures that shipped to the paper.
    panel_labels = {"probe_readout": "probe readout (a.u.)",
                    "valence": "behavioral valence"}
    panels = [ro for ro in ("probe_readout", "valence")
              if any(np.isfinite(cells.get(ro, {}).get(f"{g}_img/{c}_ctx", np.nan))
                     for g in groups for c in conds)]
    if not panels:
        raise ValueError("no readout has finite cell means; nothing to plot")

    ncol = len(panels) + (1 if arb else 0)
    # Widen past the per-panel size when few panels survive, so the suptitle is not
    # clipped on a single-panel (no-probe) model.
    fig, axes = plt.subplots(1, ncol, figsize=(max(4.2 * ncol, 6.4), 4.0), squeeze=False)
    for ax, ro in zip(axes[0], panels):
        x = np.arange(len(groups))
        for j, cond in enumerate(conds):
            vals = [cells[ro].get(f"{g}_img/{cond}_ctx", np.nan) for g in groups]
            ax.bar(x + (j - 0.5) * 0.35, vals, 0.35, label=f"{cond} context")
        ax.set_xticks(x); ax.set_xticklabels([f"{g} image" for g in groups])
        # With one surviving panel the title would just repeat the y-label.
        if len(panels) > 1:
            ax.set_title(panel_labels[ro])
        ax.set_ylabel(panel_labels[ro])
        ax.legend(fontsize=7); ax.axhline(0, color="gray", lw=0.5)
    if arb:
        ax = axes[0][len(panels)]
        xs = sorted(arb["valence"])
        ax.plot(xs, [arb["valence"][b] for b in xs], "-o", label="behavioral valence")
        ax.plot(xs, [arb["probe"][b] for b in xs], "k:", label="probe (upstream ~0)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"arbitration (slope {arb['valence_slope']:+.2f})")
        ax.set_xlabel("β  (pleasantness Δμ)"); ax.set_ylabel("Δ vs β=0"); ax.legend(fontsize=7)
    # No internal stage name here: these figures ship in the paper, where the LaTeX
    # caption carries the description. Only claim arbitration when it was measured.
    title = "Modality conflict: image vs. context valence"
    if arb:
        title += ", with steering arbitration"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / fig_name, dpi=130)
    fig.savefig((FIGURES_DIR / fig_name).with_suffix(".pdf"))  # vector twin for the paper
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — analysis + decision D2")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--parquet", default=None,
                    help="alternate base-pass parquet (name under results/stage_f or a path), e.g. "
                         "conflict_minimal.parquet for the T1.1 minimal-pair bank")
    args = ap.parse_args()
    run(base_pq=args.parquet)


if __name__ == "__main__":
    main()
