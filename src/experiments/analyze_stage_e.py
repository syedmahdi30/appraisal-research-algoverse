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
    """Per (image, arm, beta) Δ log-prob vs that image's β=0 baseline (long-form, one row each)."""
    base = df[df["arm"] == "_base"].set_index("image_path")[LP]
    steered = df[df["arm"] != "_base"].copy()
    d = steered[LP].to_numpy() - base.loc[steered["image_path"]][LP].to_numpy()
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
    betas = sorted(int(b) for b in df.loc[df["arm"] != "_base", "beta"].unique())
    top_b = max(betas)

    delta = _delta_frame(df)
    # mean Δ per (arm, beta) over images: {arm: {beta: {emotion: mean}}}
    grp = delta.groupby(["arm", "beta"])[LP].mean()
    mean_delta = {arm: {int(b): {w: float(grp.loc[(arm, b), f"lp_{w}"]) for w in EMOTION_LABELS}
                        for b in betas if (arm, b) in grp.index}
                  for arm in delta["arm"].unique()}

    def target_gain(arm, emo, b):
        return mean_delta.get(arm, {}).get(b, {}).get(emo)

    # per-arm target stats
    arm_stats = {}
    for arm in delta["arm"].unique():
        tgt = targets.get(arm) or ARMS.get(arm, {}).get("target")
        if tgt is None:
            arm_stats[arm] = {"target": None}
            continue
        series = [target_gain(arm, tgt, b) for b in betas]
        slope = float(np.polyfit(betas, series, 1)[0])
        rho = float(spearmanr(betas, series)[0]) if len(set(series)) > 1 else 0.0
        gain_top = target_gain(arm, tgt, top_b)
        # rank of target among 13 emotions by mean Δ at β=+3 (1 = biggest gainer)
        at_top = mean_delta[arm][top_b]
        rank = 1 + sum(1 for w in EMOTION_LABELS if at_top[w] > at_top[tgt])
        # win-rate: fraction of images where target is the top gainer at β=+3
        cell = delta[(delta["arm"] == arm) & (delta["beta"] == top_b)]
        if len(cell):
            arr = cell[LP].to_numpy()
            winners = np.array(EMOTION_LABELS)[arr.argmax(axis=1)]
            win_rate = float(np.mean(winners == tgt))
        else:
            win_rate = float("nan")
        arm_stats[arm] = {"target": tgt, "slope": slope, "spearman": rho,
                          "gain_at_top": gain_top, "rank_at_top": int(rank),
                          "win_rate": win_rate, "n_images": int(len(cell))}

    # combo vs best single component, per congruent/control combo arm
    combo_vs_single = {}
    for arm in [a for a in mean_delta if a in ARMS and len(ARMS[a]["combo"]) == 2]:
        tgt = arm_stats.get(arm, {}).get("target")
        if tgt is None:
            continue
        comps = [c for c, _ in ARMS[arm]["combo"]]
        singles = {c: target_gain(_single_arm_for(c), tgt, top_b) for c in comps}
        best_single = max((v for v in singles.values() if v is not None), default=None)
        combo_g = target_gain(arm, tgt, top_b)
        combo_vs_single[arm] = {
            "combo_gain_at_top": combo_g, "component_single_gains": singles,
            "best_single_gain": best_single,
            "combo_beats_single": (combo_g is not None and best_single is not None
                                   and combo_g > best_single),
        }

    # N1 must not raise anger; R (random) target-agnostic gains as a null ceiling
    n1_anger_top = target_gain("N1", "anger", top_b)
    n1_ok = n1_anger_top is not None and n1_anger_top <= 0.05

    # per-arm SIGNAL membership among A1-A5
    def arm_passes(arm):
        s = arm_stats.get(arm, {})
        cvs = combo_vs_single.get(arm, {})
        return (s.get("target") is not None and s.get("slope", 0) > 0
                and s.get("spearman", 0) >= 0.8 and cvs.get("combo_beats_single", False)
                and s.get("win_rate", 0) >= 0.15)

    signal_arms = [a for a in CONGRUENT_ARMS if arm_passes(a)]
    verdict = _verdict(signal_arms, n1_ok, arm_stats, combo_vs_single, mean_delta, betas, top_b)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "betas": betas, "top_beta": top_b,
        "n_images": int(df["image_path"].nunique()),
        "arm_targets": targets, "arm_stats": arm_stats,
        "combo_vs_single": combo_vs_single,
        "n1_anger_at_top": n1_anger_top, "n1_ok": n1_ok,
        "signal_arms": signal_arms, "n_signal_arms": len(signal_arms),
        "mean_delta_logprob": mean_delta, "verdict": verdict,
    }
    save_json(metrics, STAGE_E_DIR / "combo_analysis.json")
    _plot(mean_delta, arm_stats, betas)

    print(f"\nStage E analysis — {metrics['n_images']} images, β {betas}.\n")
    print(f"{'arm':7s} {'target':9s} {'slope':>8s} {'ρ':>6s} {'rank@+':>7s} "
          f"{'win%':>6s}  {'combo>single':>12s}")
    for arm in CONGRUENT_ARMS + ("N1", "N2"):
        s = arm_stats.get(arm, {})
        if s.get("target") is None:
            continue
        cvs = combo_vs_single.get(arm, {})
        print(f"{arm:7s} {s['target']:9s} {s['slope']:+8.3f} {s['spearman']:+6.2f} "
              f"{s['rank_at_top']:>7d} {s['win_rate']*100:>5.0f}% "
              f"{str(cvs.get('combo_beats_single','')):>12s}")
    print(f"\n  N1 anger Δ@+{top_b} = {n1_anger_top:+.3f} ({'ok' if n1_ok else 'RAISES anger'})")
    print(f"  signal arms ({len(signal_arms)}/5): {signal_arms}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  figure -> {FIGURES_DIR/'stage_e_combo_pilot.png'}   "
          f"metrics -> {STAGE_E_DIR/'combo_analysis.json'}")
    return metrics


def _verdict(signal_arms, n1_ok, arm_stats, combo_vs_single, mean_delta, betas, top_b) -> str:
    if len(signal_arms) >= 3 and n1_ok:
        return (f"SIGNAL (criterion: {len(signal_arms)}/5 congruent arms show positive+monotone "
                f"target slope, combo>best-single at β=+{top_b}, win-rate≥15%; N1 does not raise "
                f"anger) — proceed to the full run (150-300 images, freq control, 3 seeds).")
    # PARTIAL: valence-linked emotion moves monotonically but the specific target does not.
    val_moves = 0
    for arm, proxy in VALENCE_PROXY.items():
        ser = [mean_delta.get(arm, {}).get(b, {}).get(proxy) for b in betas]
        if all(v is not None for v in ser) and abs(np.polyfit(betas, ser, 1)[0]) > 0.02 \
                and arm not in signal_arms:
            val_moves += 1
    if val_moves >= 3 and len(signal_arms) < 3:
        return ("PARTIAL (criterion: valence-linked emotions move but specific targets do not) — "
                "retry β=±4 and inspect A1-raw; if still flat, demote to 'shared valence axis, no "
                "specific-emotion synthesis' and pivot the headline.")
    beats_single = sum(1 for c in combo_vs_single.values() if c.get("combo_beats_single"))
    if beats_single == 0:
        return ("NULL (criterion: combos do not exceed single components) — first re-verify the "
                "Stage D pleasantness slope on these 30 images as a sanity gate; if that holds, "
                "report the combo-null honestly and stop Stage E.")
    return (f"INCONCLUSIVE ({len(signal_arms)}/5 arms pass; {beats_single} combos beat their best "
            f"single; N1 anger {'ok' if n1_ok else 'raised'}) — needs the human call; consider "
            f"β=±4 or the full-run controls.")


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
