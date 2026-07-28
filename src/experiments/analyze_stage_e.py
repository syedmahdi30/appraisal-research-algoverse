"""Stage E, step 4 — analysis + decision D1 (pilot-plan-stage-e-f.md).

CPU-only. Consumes results/stage_e/combo_pilot.parquet (+ combo_pilot_metrics.json for arm
targets) and asks whether combined appraisal directions produce the theory-predicted SPECIFIC
emotion under image input. For each arm we compute, relative to the per-image β=0 baseline:
  - mean Δ log-prob per emotion vs β; the TARGET emotion's slope and monotonicity (Spearman over
    the 6 β points);
  - the target's rank among the 13 emotions at β=+3;
  - win-rate = fraction of images where the target is the top gainer at β=+3 (chance ≈ 1/13);
  - combo target-gain at β=+3 vs the best of its two single components (S-arms).

Decision D1 (human confirms):
  SIGNAL  — >=3 of A1-A5 have positive+monotone (ρ>=0.8) target slope, combo > best single at
            β=+3, win-rate >= 15%, and N1 does not raise anger.
  PARTIAL — only valence-linked emotions move (no specificity): inspect A1-raw / β=±4.
  NULL    — combos <= singles <= random.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..data.labels import EMOTION_LABELS
from ..paths import FIGURES_DIR, STAGE_E_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_e_arms import APPRAISALS, ARMS, CONGRUENT_ARMS

LP = [f"lp_{w}" for w in EMOTION_LABELS]
# valence-linked emotion each congruent arm would move if only a valence axis (not specificity)
# were active: +pleasant arms -> joy up, +unpleasant arms -> sadness up.
VALENCE_PROXY = {"A1": "sadness", "A2": "sadness", "A3": "sadness",
                 "A4": "joy", "A5": "joy"}


def _delta_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per (image, arm, beta) Δ log-prob vs that image's β=0 baseline (long-form, one row each).

    Aligns to the baseline by IMAGE CELL, not image_path: EMOTIC is per-person, so the same
    image_path recurs and cannot key the baseline. Rows are written per image as [_base, then the
    arm×β grid], so `(arm=="_base").cumsum()` labels each image's block uniquely.
    """
    df = df.copy()
    df["_img"] = (df["arm"] == "_base").cumsum()
    base = df[df["arm"] == "_base"].set_index("_img")[LP]
    steered = df[df["arm"] != "_base"].copy()
    d = steered[LP].to_numpy() - base.loc[steered["_img"]][LP].to_numpy()
    out = steered[["image_path", "arm", "beta"]].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(d, columns=LP)], axis=1)


def _single_arm_for(appraisal: str) -> str:
    """S-arm name whose single component is `appraisal`."""
    for i, a in enumerate(APPRAISALS, start=1):
        if a == appraisal:
            return f"S{i}"
    raise KeyError(appraisal)


def run(config_path: str | None = None) -> dict:
    ensure_dirs()
    pq = STAGE_E_DIR / "combo_pilot.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run stage_e_combo first.")
    df = pd.read_parquet(pq)
    meta = load_config(STAGE_E_DIR / "combo_pilot_metrics.json")
    targets = meta.get("arm_targets", {})
    arm_meta = meta.get("arm_meta", {})
    # matched-norm single controls, if the run produced them: {parent_arm: {appraisal: arm_name}}
    matched_map: dict = {}
    for nm, mm in arm_meta.items():
        if mm.get("rule") == "matched_norm_single":
            matched_map.setdefault(mm["parent"], {})[mm["appraisal"]] = nm
    betas = sorted(int(b) for b in df.loc[df["arm"] != "_base", "beta"].unique())
    top_b = max(betas)

    delta = _delta_frame(df)
    # mean Δ per (arm, beta) over images: {arm: {beta: {emotion: mean}}}
    grp = delta.groupby(["arm", "beta"])[LP].mean()
    mean_delta = {arm: {int(b): {w: float(grp.loc[(arm, b), f"lp_{w}"]) for w in EMOTION_LABELS}
                        for b in betas if (arm, b) in grp.index}
                  for arm in delta["arm"].unique()}

    def gain(arm, emo, b):
        return mean_delta.get(arm, {}).get(b, {}).get(emo)

    def rank_of(arm, emo, b):
        """Rank of `emo` among the 13 by mean Δ at β=b (1 = biggest gainer); None if absent."""
        d = mean_delta.get(arm, {}).get(b, {})
        if emo not in d:
            return None
        return 1 + sum(1 for w in EMOTION_LABELS if d[w] > d[emo])

    def topk(arm, b, k=3):
        d = mean_delta.get(arm, {}).get(b, {})
        return [(w, float(v)) for w, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    # per-arm target stats
    arm_stats = {}
    for arm in delta["arm"].unique():
        tgt = targets.get(arm) or ARMS.get(arm, {}).get("target")
        if tgt is None:
            arm_stats[arm] = {"target": None}
            continue
        series = [gain(arm, tgt, b) for b in betas]
        slope = float(np.polyfit(betas, series, 1)[0])
        rho = float(spearmanr(betas, series)[0]) if len(set(series)) > 1 else 0.0
        # win-rate: fraction of images where target is the top gainer at β=+3
        cell = delta[(delta["arm"] == arm) & (delta["beta"] == top_b)]
        if len(cell):
            arr = cell[LP].to_numpy()
            winners = np.array(EMOTION_LABELS)[arr.argmax(axis=1)]
            win_rate = float(np.mean(winners == tgt))
        else:
            win_rate = float("nan")
        t3 = topk(arm, top_b)
        arm_stats[arm] = {"target": tgt, "slope": slope, "spearman": rho,
                          "gain_at_top": gain(arm, tgt, top_b), "rank_at_top": rank_of(arm, tgt, top_b),
                          "argmax": t3[0][0] if t3 else None, "top3_at_top": t3,
                          "win_rate": win_rate, "n_images": int(len(cell))}

    # Combination vs its two single components — RANK-based (magnitude-fair): a single arm steers
    # with the full raw Δμ while the combo is rescaled to the mean component norm, so comparing raw
    # target GAINS favours the single by construction. What the specificity claim needs is whether
    # the COMBINATION makes the target a top gainer that NEITHER component alone elevates.
    combo_vs_single = {}
    for arm in [a for a in mean_delta if a in ARMS and len(ARMS[a]["combo"]) == 2]:
        tgt = arm_stats.get(arm, {}).get("target")
        if tgt is None:
            continue
        singles = {}
        for c, _ in ARMS[arm]["combo"]:
            s = _single_arm_for(c)
            singles[c] = {"single_arm": s, "target_gain": gain(s, tgt, top_b),
                          "target_rank": rank_of(s, tgt, top_b),
                          "argmax": (topk(s, top_b, 1)[0][0] if s in mean_delta else None)}
        combo_rank = arm_stats[arm]["rank_at_top"]
        single_ranks = [v["target_rank"] for v in singles.values() if v["target_rank"] is not None]
        single_argmaxes = {v["argmax"] for v in singles.values() if v["argmax"]}
        combo_g = gain(arm, tgt, top_b)
        best_single_g = max((v["target_gain"] for v in singles.values()
                             if v["target_gain"] is not None), default=None)

        # matched-norm control (same norm as the combo, direction only differs): the definitive
        # magnitude-controlled test when the run produced the matched arms.
        matched = {}
        for c, _ in ARMS[arm]["combo"]:
            mnm = matched_map.get(arm, {}).get(c)
            if mnm and mnm in mean_delta:
                matched[c] = {"matched_arm": mnm, "target_gain": gain(mnm, tgt, top_b),
                              "target_rank": rank_of(mnm, tgt, top_b),
                              "argmax": (topk(mnm, top_b, 1)[0][0] if mnm in mean_delta else None)}
        matched_ranks = [v["target_rank"] for v in matched.values() if v["target_rank"] is not None]
        matched_argmaxes = {v["argmax"] for v in matched.values() if v["argmax"]}
        combo_beats_matched = None
        if matched:
            combo_beats_matched = bool(
                combo_rank is not None and matched_ranks and combo_rank <= min(matched_ranks)
                and combo_rank <= 2 and tgt not in matched_argmaxes)

        # specificity: prefer the matched-norm basis; else fall back to the raw-single argmax test.
        raw_spec = combo_rank is not None and combo_rank <= 2 and tgt not in single_argmaxes
        spec = combo_beats_matched if combo_beats_matched is not None else raw_spec
        combo_vs_single[arm] = {
            "combo_target_rank": combo_rank, "combo_target_gain": combo_g,
            "combo_argmax": arm_stats[arm]["argmax"], "singles": singles,
            "min_single_target_rank": min(single_ranks) if single_ranks else None,
            "single_argmaxes": sorted(single_argmaxes),
            "combo_beats_single_gain": (combo_g is not None and best_single_g is not None
                                        and combo_g > best_single_g),
            "combo_sharpens_rank": (combo_rank is not None and single_ranks
                                    and combo_rank <= min(single_ranks) and combo_rank <= 2),
            "matched_norm_singles": matched or None,
            "combo_beats_matched_norm": combo_beats_matched,
            "specificity_from_combo": spec,
            "specificity_basis": "matched_norm" if combo_beats_matched is not None else "raw_single_argmax",
        }

    # N1 must not raise anger; R (random) target-agnostic gains as a null ceiling
    n1_anger_top = gain("N1", "anger", top_b)
    n1_ok = n1_anger_top is not None and n1_anger_top <= 0.05

    # SIGNAL membership among A1-A5 — magnitude-fair: target is a top-2 gainer, monotone up, and
    # wins per-image above chance. (The combination-sharpens / specificity flags are reported
    # separately rather than gating, since they answer a different, mechanism question.)
    def arm_passes(arm):
        s = arm_stats.get(arm, {})
        return (s.get("target") is not None and s.get("spearman", 0) >= 0.8
                and (s.get("rank_at_top") or 99) <= 2 and s.get("win_rate", 0) >= 0.15)

    signal_arms = [a for a in CONGRUENT_ARMS if arm_passes(a)]
    sharpen_arms = [a for a in CONGRUENT_ARMS if combo_vs_single.get(a, {}).get("specificity_from_combo")]
    verdict = _verdict(signal_arms, sharpen_arms, n1_ok, arm_stats, combo_vs_single, mean_delta, betas, top_b)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "betas": betas, "top_beta": top_b,
        "n_images": int(df["image_path"].nunique()),
        "arm_targets": targets, "arm_stats": arm_stats,
        "combo_vs_single": combo_vs_single,
        "n1_anger_at_top": n1_anger_top, "n1_ok": n1_ok,
        "signal_arms": signal_arms, "n_signal_arms": len(signal_arms),
        "sharpen_arms": sharpen_arms, "mean_delta_logprob": mean_delta, "verdict": verdict,
    }
    save_json(metrics, STAGE_E_DIR / "combo_analysis.json")
    _plot(mean_delta, arm_stats, betas)

    print(f"\nStage E analysis — {metrics['n_images']} images, β {betas}.\n")
    print(f"{'arm':5s} {'target':9s} {'slope':>7s} {'ρ':>5s} {'rank':>4s} {'win%':>5s}  "
          f"{'combo argmax':>12s}   singles → argmax (target-rank)")
    for arm in CONGRUENT_ARMS + ("N1", "N2"):
        s = arm_stats.get(arm, {})
        if s.get("target") is None:
            continue
        cvs = combo_vs_single.get(arm, {})
        sing = cvs.get("singles", {})
        sstr = ", ".join(f"{c[:5]}→{sing[c]['argmax']}(r{sing[c]['target_rank']})" for c in sing)
        star = " *" if cvs.get("specificity_from_combo") else ""
        print(f"{arm:5s} {s['target']:9s} {s['slope']:+7.3f} {s['spearman']:+5.2f} "
              f"{(s['rank_at_top'] or 0):>4d} {s['win_rate']*100:>4.0f}%  "
              f"{str(s.get('argmax')):>12s}{star}   {sstr}")

    has_matched = any(combo_vs_single.get(a, {}).get("matched_norm_singles") for a in CONGRUENT_ARMS)
    if has_matched:
        print("\n  matched-norm control (single at the COMBO's norm — direction-only difference):")
        for arm in CONGRUENT_ARMS:
            cvs = combo_vs_single.get(arm, {})
            mm = cvs.get("matched_norm_singles")
            if not mm:
                continue
            mstr = ", ".join(f"{c[:5]}→{v['argmax']}(r{v['target_rank']})" for c, v in mm.items())
            print(f"    {arm}: combo→{cvs['combo_argmax']}(r{cvs['combo_target_rank']}) vs "
                  f"matched {mstr}  -> combo beats matched: {cvs['combo_beats_matched_norm']}")

    basis = "matched-norm" if has_matched else "raw-single argmax (magnitude-confounded — add "\
            "matched_norm_control for the clean test)"
    print(f"\n  N1 anger Δ@+{top_b} = {n1_anger_top:+.3f} ({'ok' if n1_ok else 'RAISES anger'})")
    print(f"  signal arms (rank≤2, monotone, win≥15%): {len(signal_arms)}/5 {signal_arms}")
    print(f"  '*' = combination creates the specificity [basis: {basis}]: {sharpen_arms}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  figure -> {FIGURES_DIR/'stage_e_combo_pilot.png'}   "
          f"metrics -> {STAGE_E_DIR/'combo_analysis.json'}")
    return metrics


def _verdict(signal_arms, sharpen_arms, n1_ok, arm_stats, combo_vs_single, mean_delta, betas, top_b) -> str:
    n, hits = len(signal_arms), ", ".join(
        f"{a}→{arm_stats[a]['argmax']}(win {arm_stats[a]['win_rate']*100:.0f}%)" for a in signal_arms)
    basis = "matched-norm" if any(combo_vs_single.get(a, {}).get("specificity_basis") == "matched_norm"
                                  for a in sharpen_arms) else "raw-single argmax"
    sharp = (f" Combination creates the specificity in {sharpen_arms} (target top-2, argmax of "
             f"neither single; basis: {basis}).") if sharpen_arms else (
             " But no arm's target is top-2 in the combo while absent from both singles — the effect "
             "may ride the dominant valence component (add matched_norm_control to settle it).")
    if n >= 3 and n1_ok:
        return (f"SIGNAL ({n}/5 congruent arms: target is a top-2 gainer, monotone, win-rate≥15% — "
                f"{hits}; N1 does not raise anger).{sharp} Proceed to the full run (150-300 images, "
                f"lexical-frequency control, 3 seeds).")
    if n >= 1 and n1_ok:
        return (f"PARTIAL-POSITIVE ({n}/5 congruent arms show clean specific-emotion steering — "
                f"{hits}; N1 ok).{sharp} Positive/surprise arms flat. This is specific-emotion "
                f"synthesis for some appraisals, not a valence-only null — report it, and for the "
                f"full run add lexical-frequency control + inspect whether the win rides the "
                f"valence component (matched-norm single vs combo).")
    # No arm makes its target a top-2 monotone gainer: valence-only or null.
    val_moves = sum(1 for a, proxy in VALENCE_PROXY.items()
                    if all(mean_delta.get(a, {}).get(b, {}).get(proxy) is not None for b in betas)
                    and abs(np.polyfit(betas, [mean_delta[a][b][proxy] for b in betas], 1)[0]) > 0.02)
    if val_moves >= 3:
        return ("PARTIAL (valence-linked emotions move but no specific target is a top-2 gainer) — "
                "retry β=±4 and inspect A1-raw; if still flat, demote to 'shared valence axis, no "
                "specific-emotion synthesis' and pivot the headline.")
    return ("NULL (no arm elevates its specific target above the other emotions) — first re-verify "
            "the Stage D pleasantness slope on these 30 images as a sanity gate; if it holds, "
            "report the combo-null honestly and stop Stage E.")


def _plot(mean_delta, arm_stats, betas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(CONGRUENT_ARMS), figsize=(3.4 * len(CONGRUENT_ARMS), 3.6),
                             squeeze=False, sharey=True)
    for ax, arm in zip(axes[0], CONGRUENT_ARMS):
        tgt = arm_stats.get(arm, {}).get("target")
        if tgt is None:
            ax.set_title(f"{arm} (no target)")
            continue
        ax.plot(betas, [mean_delta[arm][b][tgt] for b in betas], "-o", lw=2, ms=4,
                label=f"combo→{tgt}")
        for c, _ in ARMS[arm]["combo"]:
            s = _single_arm_for(c)
            if s in mean_delta:
                ax.plot(betas, [mean_delta[s][b][tgt] for b in betas], "--", ms=3,
                        label=f"single {c[:6]}")
        if "R" in mean_delta:
            ax.plot(betas, [mean_delta["R"][b][tgt] for b in betas], "k:", label="random")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{arm}: →{tgt}\nslope {arm_stats[arm]['slope']:+.3f}")
        ax.set_xlabel("β"); ax.legend(fontsize=6)
    axes[0][0].set_ylabel("Δ log-prob(target) vs β=0")
    fig.suptitle("Stage E — appraisal-specific emotion steering (combo vs components vs random)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_e_combo_pilot.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage E step 4 — analysis + decision D1")
    ap.add_argument("--config", default="config/stage_e.yaml")
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
