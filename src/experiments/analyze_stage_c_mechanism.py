"""Stage C mechanism analysis — combine the neutral + rich caption baselines (CPU-only).

Loads the persisted per-image read-outs (caption_readout.parquet [neutral] and
caption_readout_rich.parquet) and measures how much UNIQUE valence signal the frozen image
read-out retains after controlling for: the neutral caption, the rich caption, and BOTH
jointly. "Beyond BOTH captions" is the tightest correlational bound on a non-verbalized
residual we can build from these artifacts. Pure re-analysis of saved predictions — no
model, no GPU, runs in seconds. (Still correlational: a surviving residual motivates the
Stage D causal test; it does not by itself prove appraisal-specific shared geometry.)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as tdist

from ..paths import STAGE_C_DIR
from .common import run_stamp, save_json


def _rankz(x):
    r = rankdata(np.asarray(x, dtype=float))
    return (r - r.mean()) / r.std()


def _r2(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    ss_tot = float(((Y - Y.mean()) ** 2).sum())
    return 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0


def semipartial(valence, controls, img):
    """Rank-based semipartial: unique contribution of `img` to `valence` beyond `controls`.

    part_r = sign(beta_img) * sqrt(R2_full - R2_reduced) on rank-standardized data; p-value
    from the t-test on the img coefficient in the full model. `controls` is a list of arrays.
    """
    cols = [valence, img, *controls]
    m = np.all(np.isfinite(np.column_stack([np.asarray(c, float) for c in cols])), axis=1)
    n = int(m.sum())
    if n < 10:
        return {"n": n, "part_r": None, "p": None}
    V = _rankz(valence[m])
    I = _rankz(img[m])
    C = [_rankz(c[m]) for c in controls]
    X_red = np.column_stack([np.ones(n), *C]) if C else np.ones((n, 1))
    X_full = np.column_stack([X_red, I])
    sr2 = max(_r2(V, X_full) - _r2(V, X_red), 0.0)
    beta, *_ = np.linalg.lstsq(X_full, V, rcond=None)
    resid = V - X_full @ beta
    dof = n - X_full.shape[1]
    se = float(np.sqrt((float(resid @ resid) / dof) * np.linalg.inv(X_full.T @ X_full)[-1, -1]))
    t = beta[-1] / se if se > 0 else 0.0
    p = float(2 * tdist.sf(abs(t), dof))
    return {"n": n, "part_r": float(np.sign(beta[-1]) * np.sqrt(sr2)), "p": p}


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m])[0]) if m.sum() >= 3 else None


def run(appraisals=("pleasantness", "unpleasantness")) -> dict:
    neu_path = STAGE_C_DIR / "caption_readout.parquet"
    rich_path = STAGE_C_DIR / "caption_readout_rich.parquet"
    if not neu_path.exists():
        raise FileNotFoundError(f"{neu_path} missing — run the neutral caption baseline first.")
    neu = pd.read_parquet(neu_path).reset_index(drop=True)
    rich = pd.read_parquet(rich_path).reset_index(drop=True) if rich_path.exists() else None

    # EMOTIC rows are per-PERSON and image_path repeats (multi-person images), so merging on
    # image_path would cross-join co-located persons and misalign valence. Both parquets come
    # from the same deterministic subset/order with no skips, so align by ROW POSITION and
    # verify the image_path columns match position-for-position.
    df = neu.rename(columns={c: f"{c}__neu" for c in neu.columns if c.startswith("pred_caption_")})
    if rich is not None:
        if len(rich) != len(neu) or not (neu["image_path"].to_numpy() == rich["image_path"].to_numpy()).all():
            raise ValueError(
                "neutral/rich parquets are not row-aligned (different subset/seed/skips). "
                "Re-run both caption baselines with the same n_images and seed."
            )
        for c in [c for c in rich.columns if c.startswith("pred_caption_")]:
            df[f"{c}__rich"] = rich[c].to_numpy()

    metrics = {"run": run_stamp(), "n": len(df), "has_rich": rich is not None, "appraisals": {}}
    for a in appraisals:
        img = df.get(f"pred_image_{a}")
        if img is None:
            continue
        img = img.to_numpy()
        val = df["valence"].to_numpy()
        neu_cap = df[f"pred_caption_{a}__neu"].to_numpy()
        entry = {
            "r_image_valence": _spearman(img, val),
            "r_neutral_valence": _spearman(neu_cap, val),
            "unique_beyond_neutral": semipartial(val, [neu_cap], img),
        }
        if rich is not None:
            rich_cap = df[f"pred_caption_{a}__rich"].to_numpy()
            entry["r_rich_valence"] = _spearman(rich_cap, val)
            entry["unique_beyond_rich"] = semipartial(val, [rich_cap], img)
            entry["unique_beyond_both"] = semipartial(val, [neu_cap, rich_cap], img)
        metrics["appraisals"][a] = entry

    save_json(metrics, STAGE_C_DIR / "mechanism_summary.json")
    return metrics


def main() -> None:
    argparse.ArgumentParser(description="Stage C mechanism analysis (CPU-only)").parse_args()
    m = run()
    print(f"\nStage C mechanism — n={m['n']}  (rich available: {m['has_rich']})\n")
    for a, e in m["appraisals"].items():
        print(f"  {a}")
        print(f"    image vs valence           rho={e['r_image_valence']:+.3f}")
        print(f"    neutral caption vs valence rho={e['r_neutral_valence']:+.3f}")
        if "r_rich_valence" in e:
            print(f"    rich caption vs valence    rho={e['r_rich_valence']:+.3f}")
        un = e["unique_beyond_neutral"]
        print(f"    unique(image | neutral)      part_r={un['part_r']:+.3f} (p={un['p']:.3f})")
        if "unique_beyond_rich" in e:
            ur, ub = e["unique_beyond_rich"], e["unique_beyond_both"]
            print(f"    unique(image | rich)         part_r={ur['part_r']:+.3f} (p={ur['p']:.3f})")
            print(f"    unique(image | neutral+rich) part_r={ub['part_r']:+.3f} (p={ub['p']:.3f})")
    print(f"\n  summary -> {STAGE_C_DIR/'mechanism_summary.json'}")


if __name__ == "__main__":
    main()
