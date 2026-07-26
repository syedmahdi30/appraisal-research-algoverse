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
    lead = "text-led" if ratio > 1.25 else "image-led" if ratio < 0.8 else "mixed"
    return {"beta_img": b["image"], "beta_txt": b["text"], "dominance_ratio": ratio, "lead": lead}


def _cell_means(df, value):
    """2x2 (image_group x context polarity) means of `value`, plus no-context/neutral references."""
    out = {}
    for grp in ("positive", "negative"):
        for cond in ("none", "positive", "negative", "neutral"):
            cell = df[(df["image_group"] == grp) & (df["condition"] == cond)][value]
            if len(cell):
                out[f"{grp}_img/{cond}_ctx"] = float(cell.mean())
    return out


def _arbitration(df, betas):
    """Mean Δ (valence, probe) vs β on incongruent cells, relative to each cell's β=0 baseline."""
    base = df[df["beta"] == 0].set_index("image_path")
    res = {"valence": {}, "probe": {}}
    for b in betas:
        sub = df[df["beta"] == b]
        dv = sub["valence"].to_numpy() - base.loc[sub["image_path"], "valence"].to_numpy()
        dp = sub["probe_readout"].to_numpy() - base.loc[sub["image_path"], "probe_readout"].to_numpy()
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

    dom = {"probe_readout": _dominance(df["probe_readout"], df),
           "valence": _dominance(df["valence"], df)}
    cells = {"probe_readout": _cell_means(df, "probe_readout"),
             "valence": _cell_means(df, "valence")}

    # context effect vs no-context, per image group (does context pull the read-out?)
    ctx_effect = {}
    for grp in ("positive", "negative"):
        g = df[df["image_group"] == grp]
        base = g[g["condition"] == "none"]["valence"].mean()
        ctx_effect[grp] = {c: float(g[g["condition"] == c]["valence"].mean() - base)
                           for c in ("positive", "negative", "neutral")}

    metrics = {
        "run": run_stamp(), "git": git_hash(), "n_rows": int(len(df)),
        "n_images": int(df["image_path"].nunique()),
        "dominance": dom, "cell_means": cells, "context_effect_vs_none": ctx_effect,
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

    print(f"\nStage F analysis — {metrics['n_images']} images, {metrics['n_rows']} base rows.\n")
    for ro in ("probe_readout", "valence"):
        d = dom[ro]
        print(f"  {ro:14s} β_img={d['beta_img']:+.3f}  β_txt={d['beta_txt']:+.3f}  "
              f"|txt|/|img|={d['dominance_ratio']:.2f}  -> {d['lead']}")
    print("\n  context effect on behavioral valence (vs no-context):")
    for grp, e in ctx_effect.items():
        print(f"    {grp:8s} img: +ctx {e['positive']:+.3f}  -ctx {e['negative']:+.3f}  "
              f"0ctx {e['neutral']:+.3f}")
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
        base = (f"SIGNAL (criterion: both cues sign-correct — β_img={b['beta_img']:+.2f}, "
                f"β_txt={b['beta_txt']:+.2f}, pattern is {b['lead']}). ")
        if arb is not None:
            if arb["within_50pct_of_stage_d"]:
                return base + (f"Arbitration works: behavioral-valence steering slope "
                               f"{arb['valence_slope']:+.2f} is within ±50% of Stage D — proceed to "
                               f"the full run.")
            return base + (f"But arbitration is weak (steering slope {arb['valence_slope']:+.2f} vs "
                           f"Stage D ~{STAGE_D_SLOPE}); inspect before the full run.")
        return base + "Run --arbitrate next to test whether steering arbitrates the conflict."
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
