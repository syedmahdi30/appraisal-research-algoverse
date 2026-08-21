"""Judge / evaluator-choice robustness (TAE P1) — is the override conclusion evaluator-independent?

The Stage F conflict result is scored by ONE evaluator: valence / argmax over the 13 crowd-enVENT
labels partitioned into pos/neg (`analyze_stage_f._flip_override`). This module asks whether the
override-gap conclusion (positive on Gemma/Qwen, null/reversed on LLaVA) is an artefact of that one
partition, by re-scoring the SAME stored runs under a bank of alternative evaluators.

It is a pure re-analysis: `stage_f_conflict.py` and `stage_f_qwen.py` already persist per-label
log-probs (`lp_{label}`) for all 13 labels, every row, every model. Any pos/neg sub-partition of the
13 is therefore exactly recoverable from disk — no forward passes, no GPU. See
`docs/judge-robustness-spec.md`.

Start here: the E0 ANCHOR CHECK proves the parametric estimator reduces to the published one before any
alternative evaluator is trusted — (1) logic-exact: parametric E0 == imported `_flip_override`;
(2) data-exact: == the stored `conflict_analysis.json` gap when it sits beside the parquet.

    python -m src.experiments.analyze_judge_robustness --selftest       # synthetic logic check, no data
    python -m src.experiments.analyze_judge_robustness --anchor-only     # E0 anchor on real parquets
    python -m src.experiments.analyze_judge_robustness                   # full evaluator sweep + table
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.labels import EMOTION_LABELS
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, run_stamp, save_json
from .analyze_stage_f import _POSITIVE, _NEGATIVE, _flip_override, _asymmetry_vs_floor

# Default parquet locations (Gemma full bank is written as conflict_pilot.parquet; see NUMBER-LEDGER).
DEFAULT_PARQUETS = {
    "Gemma-3-4B": STAGE_F_DIR / "conflict_pilot.parquet",
    "Qwen3-VL-8B": STAGE_F_DIR / "conflict_qwen.parquet",
    "LLaVA-1.5-7B": STAGE_F_DIR / "conflict_llava.parquet",
}
LP = [f"lp_{w}" for w in EMOTION_LABELS]


# --------------------------------------------------------------------------- evaluators
@dataclass(frozen=True)
class Evaluator:
    """A scoring judge: a pos/neg partition, the label set the argmax ranges over, and the softmax
    denominator used for the graded valence. Defaults reproduce the paper's E0 evaluator exactly."""
    name: str
    pos: tuple[str, ...] = _POSITIVE
    neg: tuple[str, ...] = _NEGATIVE
    argmax_set: tuple[str, ...] = tuple(EMOTION_LABELS)   # labels the model may 'pick' among
    denom: tuple[str, ...] = tuple(EMOTION_LABELS)        # softmax denominator for graded valence
    note: str = ""

    def __post_init__(self):
        alllab = set(EMOTION_LABELS)
        for fld in ("pos", "neg", "argmax_set", "denom"):
            bad = set(getattr(self, fld)) - alllab
            if bad:
                raise ValueError(f"{self.name}: {fld} has labels outside the 13: {sorted(bad)}")
        if set(self.pos) & set(self.neg):
            raise ValueError(f"{self.name}: pos and neg overlap: {set(self.pos) & set(self.neg)}")
        if not set(self.pos) | set(self.neg) <= set(self.denom):
            raise ValueError(f"{self.name}: pos/neg must be within the softmax denominator")
        if not set(self.pos) | set(self.neg) <= set(self.argmax_set):
            raise ValueError(f"{self.name}: pos/neg must be within the argmax set")


# E0 is the anchor; J1..J5 are the perturbations from docs/judge-robustness-spec.md.
E0 = Evaluator("E0_published", note="the paper's evaluator — anchor")
EVALUATORS: list[Evaluator] = [
    E0,
    Evaluator("J1_sparest", pos=("joy",), neg=("sadness",), argmax_set=("joy", "sadness"),
              denom=("joy", "sadness"), note="cleanest 1-vs-1 antonym contrast"),
    Evaluator("J2_drop_weak_pos", pos=("joy", "pride"),
              note="drop arguably-weak positives relief, trust -> 'other'"),
    Evaluator("J3_boredom_neutral", neg=("anger", "disgust", "fear", "guilt", "sadness", "shame"),
              note="re-file low-arousal boredom out of NEG"),
    Evaluator("J4_denom_swap", denom=tuple(_POSITIVE) + tuple(_NEGATIVE),
              note="renormalise over pos u neg only (drop surprise/neutral sink mass)"),
    Evaluator("J5_rank_only", note="identical partition; argmax-only, magnitudes ignored (== E0 override)"),
]


# --------------------------------------------------------------------------- estimators
def _argmax_category(df: pd.DataFrame, ev: Evaluator) -> np.ndarray:
    """Category ('pos'/'neg'/'other') of the argmax label within `ev.argmax_set`, per row."""
    cols = [f"lp_{w}" for w in ev.argmax_set]
    win = np.asarray(ev.argmax_set)[df[cols].to_numpy().argmax(axis=1)]
    cat = {w: ("pos" if w in ev.pos else "neg" if w in ev.neg else "other") for w in EMOTION_LABELS}
    return np.array([cat[w] for w in win])


def _override_gap(df: pd.DataFrame, ev: Evaluator, n_boot: int = 2000, seed: int = 0) -> dict:
    """Parametric generalisation of `analyze_stage_f._flip_override`. At E0 params it is bit-identical.

    Override = the argmax emotion's valence category follows the CONTEXT against the image:
    positive image + negative context -> 'neg'; negative image + positive context -> 'pos'.
    Per-image mean over the bank; gap bootstrapped over images (clustered CI), seed/n_boot matched to
    `_flip_override` so the CI reproduces too.
    """
    if not set(LP).issubset(df.columns):
        return {}
    d = df.copy()
    d["_cat"] = _argmax_category(d, ev)

    def per_image(group, cond, win):
        g = d[(d["image_group"] == group) & (d["condition"] == cond)]
        return g.groupby("image_path")["_cat"].apply(lambda s: float((s == win).mean())).to_numpy()

    pn = per_image("positive", "negative", "neg")
    np_ = per_image("negative", "positive", "pos")
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


def _rescore_valence(df: pd.DataFrame, ev: Evaluator) -> pd.Series:
    """Graded valence SigmaP(pos) - SigmaP(neg) under `ev`'s partition and softmax denominator, from the
    stored log-probs. Softmax over a subset D = renormalised superset softmax restricted to D:
    P_D(w) = exp(lp_w) / Sigma_{v in D} exp(lp_v)."""
    dcols = [f"lp_{w}" for w in ev.denom]
    p = np.exp(df[dcols].to_numpy())
    p = p / p.sum(axis=1, keepdims=True)
    idx = {w: i for i, w in enumerate(ev.denom)}
    pos = p[:, [idx[w] for w in ev.pos]].sum(axis=1)
    neg = p[:, [idx[w] for w in ev.neg]].sum(axis=1)
    return pd.Series(pos - neg, index=df.index)


def _mirror_contrast(df: pd.DataFrame, ev: Evaluator, n_boot: int = 2000, seed: int = 0) -> dict:
    """Re-score valence under `ev`, then reuse the published asymmetry-vs-floor estimator unchanged.
    Requires the 'none'/'neutral' condition structure the estimator keys on; returns {} if absent."""
    if "condition" not in df.columns or (df["condition"] == "neutral").sum() == 0:
        return {}
    d = df.copy()
    d["valence"] = _rescore_valence(df, ev).to_numpy()
    try:
        res = _asymmetry_vs_floor(d, n_boot=n_boot, seed=seed)
    except Exception as exc:  # noqa: BLE001 — a malformed bank should not kill the sweep
        return {"error": str(exc)}
    return {k: res[k] for k in ("asymmetry_index", "asymmetry_ci95", "headroom_norm_pull_drop",
                                "headroom_norm_pull_rise") if k in res}


# --------------------------------------------------------------------------- anchor
def anchor_check(df: pd.DataFrame, published_gap: float | None = None, tol: float = 1e-9) -> dict:
    """Prove the parametric estimator reduces to the published one at E0 before trusting J1..J5.

    (1) logic-exact: `_override_gap(df, E0)` == imported `_flip_override(df)` on the point estimates.
    (2) data-exact:  == `published_gap` (from a sibling conflict_analysis.json) when available.
    """
    mine = _override_gap(df, E0)
    ref = _flip_override(df)
    if not mine or not ref:
        return {"ok": False, "reason": "no lp_* columns or no conflict cells in this parquet"}
    dgap = abs(mine["dominance_gap"] - ref["dominance_gap"])
    dneg = abs(mine["neg_ctx_overrides_pos_img"] - ref["neg_ctx_overrides_pos_img"])
    dpos = abs(mine["pos_ctx_overrides_neg_img"] - ref["pos_ctx_overrides_neg_img"])
    logic_ok = max(dgap, dneg, dpos) < tol
    out = {"ok": logic_ok, "parametric_gap": mine["dominance_gap"], "reference_gap": ref["dominance_gap"],
           "max_abs_diff": max(dgap, dneg, dpos), "logic_exact": logic_ok}
    if published_gap is not None:
        ddata = abs(mine["dominance_gap"] - published_gap)
        out["published_gap"] = published_gap
        out["data_exact"] = ddata < 1e-4
        out["ok"] = out["ok"] and out["data_exact"]
    return out


def _published_gap_beside(parquet: Path) -> float | None:
    """The stored flip_override/dominance_gap from conflict_analysis.json, if it sits beside the parquet."""
    cand = parquet.parent / "conflict_analysis.json"
    if not cand.exists():
        return None
    try:
        j = json.loads(cand.read_text())
        return float(j["flip_override"]["dominance_gap"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- driver
def _load(parquet: Path) -> pd.DataFrame | None:
    if not parquet.exists():
        print(f"  [skip] {parquet} not found")
        return None
    df = pd.read_parquet(parquet)
    if not set(LP).issubset(df.columns):
        print(f"  [skip] {parquet.name} lacks lp_* columns (cannot re-score)")
        return None
    return df


def run(parquets: dict[str, Path], n_boot: int = 2000, anchor_only: bool = False) -> dict:
    ensure_dirs()
    results: dict[str, dict] = {}
    print("\n=== E0 anchor check (parametric == published) ===")
    for model, pq in parquets.items():
        df = _load(pq)
        if df is None:
            continue
        anc = anchor_check(df, published_gap=_published_gap_beside(pq))
        results.setdefault(model, {})["anchor"] = anc
        tag = "PASS" if anc.get("ok") else "FAIL"
        extra = ""
        if "published_gap" in anc:
            extra = f" | published {anc['published_gap']:+.4f} data_exact={anc['data_exact']}"
        print(f"  {model:14s} [{tag}] parametric {anc.get('parametric_gap', float('nan')):+.4f} "
              f"vs reference {anc.get('reference_gap', float('nan')):+.4f} "
              f"(max|Δ|={anc.get('max_abs_diff', float('nan')):.2e}){extra}")

    if any(not r["anchor"].get("ok") for r in results.values() if "anchor" in r):
        print("\n  !! anchor FAILED — refusing to report alternative evaluators until E0 reproduces.")
        return {"run": run_stamp(), "git": git_hash(), "models": results, "anchor_passed": False}
    if anchor_only:
        return {"run": run_stamp(), "git": git_hash(), "models": results, "anchor_passed": True}

    print("\n=== evaluator sweep (override gap; mirror contrast where computable) ===")
    for model, pq in parquets.items():
        df = _load(pq)
        if df is None:
            continue
        rows = {}
        for ev in EVALUATORS:
            og = _override_gap(df, ev, n_boot=n_boot)
            mc = _mirror_contrast(df, ev, n_boot=n_boot)
            rows[ev.name] = {"override": og, "mirror": mc, "note": ev.note}
        results[model]["evaluators"] = rows
        print(f"\n  {model}")
        for name, r in rows.items():
            og = r["override"]
            if not og:
                print(f"    {name:18s}  (no conflict cells)")
                continue
            ci = og["dominance_gap_ci95"]
            verdict = "positive" if ci[0] > 0 else "null/reversed" if ci[1] < 0 else "null"
            mc = r["mirror"].get("asymmetry_index")
            mcs = f" | mirror {mc:+.3f}" if mc is not None else ""
            print(f"    {name:18s}  gap {og['dominance_gap']:+.0%} "
                  f"[{ci[0]:+.0%},{ci[1]:+.0%}]  {verdict}{mcs}")

    out = {"run": run_stamp(), "git": git_hash(), "n_boot": n_boot,
           "anchor_passed": True, "models": results}
    save_json(out, STAGE_F_DIR / "judge_robustness.json")
    _write_latex(results, STAGE_F_DIR / "judge_robustness_table.tex")
    print(f"\n  data -> {STAGE_F_DIR / 'judge_robustness.json'}")
    print(f"  table -> {STAGE_F_DIR / 'judge_robustness_table.tex'}")
    return out


def _write_latex(results: dict, path: Path) -> None:
    """Evaluator x model override-gap table — the C3 backing / candidate A5 table."""
    models = [m for m in results if "evaluators" in results[m]]
    if not models:
        return
    names = [ev.name for ev in EVALUATORS]
    lines = [r"\begin{tabular}{l" + "r" * len(models) + "}", r"\toprule",
             "Evaluator & " + " & ".join(m.replace("-", "\\text{-}") for m in models) + r" \\",
             r"\midrule"]
    for nm in names:
        cells = []
        for m in models:
            og = results[m]["evaluators"].get(nm, {}).get("override", {})
            if og:
                ci = og["dominance_gap_ci95"]
                cells.append(f"${og['dominance_gap']*100:+.0f}\\%$ $[{ci[0]*100:+.0f},{ci[1]*100:+.0f}]$")
            else:
                cells.append("---")
        lines.append(f"{nm.replace('_', chr(92)+'_')} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- self-test
def _selftest() -> None:
    """Synthetic logic check with no real data: construct rows whose argmax is forced, and confirm
    the parametric estimator and the imported reference agree, and that a hand-computable case lands."""
    rng = np.random.default_rng(0)
    rows = []
    # 10 positive images: under negative context the model always picks 'sadness' (override); under
    # neutral/positive it picks 'joy'. 10 negative images: mirror. Gives a hand-computable gap of +1.0.
    def lp_peaked(winner):
        v = {w: -10.0 + rng.normal(0, 0.01) for w in EMOTION_LABELS}
        v[winner] = 0.0
        return {f"lp_{w}": v[w] for w in EMOTION_LABELS}
    for i in range(10):
        p = f"pos/{i}.jpg"
        rows.append({"image_path": p, "image_group": "positive", "condition": "none", **lp_peaked("joy")})
        rows.append({"image_path": p, "image_group": "positive", "condition": "neutral", **lp_peaked("joy")})
        rows.append({"image_path": p, "image_group": "positive", "condition": "negative", **lp_peaked("sadness")})
        rows.append({"image_path": p, "image_group": "positive", "condition": "positive", **lp_peaked("joy")})
    for i in range(10):
        n = f"neg/{i}.jpg"
        rows.append({"image_path": n, "image_group": "negative", "condition": "none", **lp_peaked("sadness")})
        rows.append({"image_path": n, "image_group": "negative", "condition": "neutral", **lp_peaked("sadness")})
        rows.append({"image_path": n, "image_group": "negative", "condition": "positive", **lp_peaked("joy")})
        rows.append({"image_path": n, "image_group": "negative", "condition": "negative", **lp_peaked("sadness")})
    df = pd.DataFrame(rows)

    anc = anchor_check(df)
    assert anc["logic_exact"], f"parametric != reference at E0: {anc}"
    og = _override_gap(df, E0)
    assert abs(og["neg_ctx_overrides_pos_img"] - 1.0) < 1e-9, og
    assert abs(og["pos_ctx_overrides_neg_img"] - 1.0) < 1e-9, og
    assert abs(og["dominance_gap"] - 0.0) < 1e-9, og  # both directions override -> gap 0 in this toy
    # J1 (joy vs sadness only) must reproduce the same argmax here (peaks are joy/sadness) -> identical.
    assert abs(_override_gap(df, EVALUATORS[1])["dominance_gap"] - og["dominance_gap"]) < 1e-9
    # valence rescore is finite and bounded in [-1, 1].
    v = _rescore_valence(df, E0).to_numpy()
    assert np.isfinite(v).all() and v.min() >= -1 - 1e-9 and v.max() <= 1 + 1e-9
    print("selftest OK — parametric E0 matches reference; override logic and rescore verified.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge/evaluator-choice robustness (TAE P1) — re-analysis only")
    ap.add_argument("--gemma", type=Path, default=DEFAULT_PARQUETS["Gemma-3-4B"])
    ap.add_argument("--qwen", type=Path, default=DEFAULT_PARQUETS["Qwen3-VL-8B"])
    ap.add_argument("--llava", type=Path, default=DEFAULT_PARQUETS["LLaVA-1.5-7B"])
    ap.add_argument("--anchor-only", action="store_true", help="run only the E0 anchor check")
    ap.add_argument("--selftest", action="store_true", help="synthetic logic check, no real data")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    parquets = {"Gemma-3-4B": args.gemma, "Qwen3-VL-8B": args.qwen, "LLaVA-1.5-7B": args.llava}
    run(parquets, n_boot=args.n_boot, anchor_only=args.anchor_only)


if __name__ == "__main__":
    main()
