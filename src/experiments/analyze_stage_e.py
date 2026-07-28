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


def run(config_path: str | None = None, decorrelate: str = "none") -> dict:
    ensure_dirs()
    suf = "" if decorrelate == "none" else f"_{decorrelate}"
    pq = STAGE_E_DIR / f"combo_pilot{suf}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run stage_e_combo"
                                + (f" --decorrelate {decorrelate}" if suf else "") + " first.")
    df = pd.read_parquet(pq)
    meta = load_config(STAGE_E_DIR / f"combo_pilot_metrics{suf}.json")
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

    # Lexical-frequency control: a more frequent / higher-prior emotion token can gain more Δ
    # log-prob under ANY perturbation. Baseline (β=0) mean log-prob per emotion is the frequency
    # proxy; we regress the 13 Δ log-probs on it and keep the RESIDUAL, so a target that still ranks
    # high after removing the frequency trend is not a frequency artifact. (The random arm R is the
    # backup null.)
    base_lp = {w: float(df[df["arm"] == "_base"][f"lp_{w}"].mean()) for w in EMOTION_LABELS}
    base_vec = np.array([base_lp[w] for w in EMOTION_LABELS])

    def resid_map(arm, b):
        d = mean_delta.get(arm, {}).get(b, {})
        if not d:
            return {}
        y = np.array([d[w] for w in EMOTION_LABELS])
        s, i = np.polyfit(base_vec, y, 1)
        r = y - (s * base_vec + i)
        return {w: float(r[k]) for k, w in enumerate(EMOTION_LABELS)}

    def resid_rank(arm, emo, b):
        r = resid_map(arm, b)
        if emo not in r:
            return None
        return 1 + sum(1 for w in EMOTION_LABELS if r[w] > r[emo])

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
                          "win_rate": win_rate, "n_images": int(len(cell)),
                          "freq_resid_at_top": resid_map(arm, top_b).get(tgt),
                          "freq_resid_rank_at_top": resid_rank(arm, tgt, top_b)}

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

        # Compositional specificity (STRICT): the combination makes the target the WINNING emotion
        # (argmax, not merely top-2 — a rank-2 target while a different emotion wins is not the
        # combination "producing" the target, and would let control arms pass), AND the target
        # beats both matched-norm singles. Falls back to the raw-single argmax test if no matched
        # arms are present.
        combo_argmax_is_target = arm_stats[arm]["argmax"] == tgt
        if combo_beats_matched is not None:
            spec = bool(combo_argmax_is_target and combo_beats_matched)
            basis = "matched_norm"
        else:
            spec = bool(combo_argmax_is_target and tgt not in single_argmaxes)
            basis = "raw_single_argmax"
        # Single-appraisal specificity (the real positive here): a COMPONENT alone (matched norm if
        # available) already makes the target its argmax — i.e. the target emotion is carried by one
        # appraisal direction, not the combination.
        src = matched if matched else singles
        single_hit = sorted({c for c, _ in ARMS[arm]["combo"] if src.get(c, {}).get("argmax") == tgt})
        combo_vs_single[arm] = {
            "combo_target_rank": combo_rank, "combo_target_gain": combo_g,
            "combo_argmax": arm_stats[arm]["argmax"], "combo_argmax_is_target": combo_argmax_is_target,
            "singles": singles,
            "min_single_target_rank": min(single_ranks) if single_ranks else None,
            "single_argmaxes": sorted(single_argmaxes),
            "combo_beats_single_gain": (combo_g is not None and best_single_g is not None
                                        and combo_g > best_single_g),
            "matched_norm_singles": matched or None,
            "combo_beats_matched_norm": combo_beats_matched,
            "specificity_from_combo": spec, "specificity_basis": basis,
            "single_appraisal_hit": single_hit or None,
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
    synthesis_arms = [a for a in CONGRUENT_ARMS if combo_vs_single.get(a, {}).get("specificity_from_combo")]
    single_arms = {a: combo_vs_single[a]["single_appraisal_hit"] for a in CONGRUENT_ARMS
                   if combo_vs_single.get(a, {}).get("single_appraisal_hit")}
    verdict = _verdict(signal_arms, synthesis_arms, single_arms, n1_ok, arm_stats, combo_vs_single, top_b)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "betas": betas, "top_beta": top_b,
        "n_images": int(df["image_path"].nunique()),
        "arm_targets": targets, "arm_stats": arm_stats,
        "combo_vs_single": combo_vs_single,
        "n1_anger_at_top": n1_anger_top, "n1_ok": n1_ok,
        "decorrelate": meta.get("decorrelate", decorrelate),
        "signal_arms": signal_arms, "n_signal_arms": len(signal_arms),
        "synthesis_arms": synthesis_arms, "single_appraisal_arms": single_arms,
        "mean_delta_logprob": mean_delta, "verdict": verdict,
    }
    save_json(metrics, STAGE_E_DIR / f"combo_analysis{suf}.json")
    _plot(mean_delta, arm_stats, betas, suf)

    print(f"\nStage E analysis [decorrelate={metrics['decorrelate']}] — "
          f"{metrics['n_images']} images, β {betas}.\n")
    print(f"{'arm':5s} {'target':9s} {'slope':>7s} {'ρ':>5s} {'rank':>4s} {'fr':>3s} {'win%':>5s}  "
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
              f"{(s['rank_at_top'] or 0):>4d} {(s.get('freq_resid_rank_at_top') or 0):>3d} "
              f"{s['win_rate']*100:>4.0f}%  {str(s.get('argmax')):>12s}{star}   {sstr}")
    print("  (rank = target rank among 13 by Δlogprob@+β; fr = same after removing the baseline-"
          "frequency trend)")

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
    single_str = ", ".join(f"{a}:{'+'.join(h)}→{arm_stats[a]['target']}" for a, h in single_arms.items())
    print(f"\n  N1 anger Δ@+{top_b} = {n1_anger_top:+.3f} ({'ok' if n1_ok else 'RAISES anger'})")
    print(f"  signal arms (target rank≤2, monotone, win≥15%): {len(signal_arms)}/5 {signal_arms}")
    print(f"  COMPOSITIONAL synthesis (target is combo argmax AND beats matched singles) "
          f"[basis: {basis}]: {synthesis_arms or 'NONE'}")
    print(f"  SINGLE-APPRAISAL specificity (one component alone yields the target): "
          f"{single_str or 'none'}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  figure -> {FIGURES_DIR/f'stage_e_combo_pilot{suf}.png'}   "
          f"metrics -> {STAGE_E_DIR/f'combo_analysis{suf}.json'}")
    return metrics


def _verdict(signal_arms, synthesis_arms, single_arms, n1_ok, arm_stats, combo_vs_single, top_b) -> str:
    """Three-way, matched-norm-aware verdict.

    COMPOSITIONAL = the combination makes the target the winning emotion above both matched-norm
    singles (the strong claim). SINGLE-APPRAISAL = the target is produced by one component alone
    (matched-norm control attributes it there) — a real but weaker specificity claim that extends
    Stage D's valence result to specific emotions without supporting compositional synthesis.
    """
    ns = len(synthesis_arms)
    single_str = "; ".join(f"{a}: {'+'.join(h)}→{arm_stats[a]['target']}" for a, h in single_arms.items())
    syn_str = ", ".join(f"{a}→{arm_stats[a]['argmax']}(win {arm_stats[a]['win_rate']*100:.0f}%)"
                        for a in synthesis_arms)
    if ns >= 3 and n1_ok:
        return (f"SIGNAL — compositional synthesis in {ns}/5 arms ({syn_str}): the COMBINATION "
                f"makes the target the winning emotion above both matched-norm singles; N1 ok. "
                f"Proceed to the full run (150-300 images, lexical-frequency control, 3 seeds).")
    # 1-2 arms compose cleanly: a genuine (partial) compositional result, not a null.
    if 1 <= ns < 3 and n1_ok:
        extra = f" Single-appraisal specificity elsewhere: {single_str}." if single_arms else ""
        return (f"PARTIAL COMPOSITIONAL — {ns}/5 arms show genuine compositional synthesis "
                f"({syn_str}): the combination makes the target the WINNING emotion above BOTH "
                f"matched-norm singles, so it is not a magnitude artifact and not carried by either "
                f"component alone.{extra} Remaining targets fail (positive/surprise hit a joy "
                f"attractor). De-correlation was necessary to expose this — with raw directions the "
                f"same arm(s) look single-appraisal because the second component is valence-"
                f"contaminated. Report the passing arm(s) as a compositional result and scale them to "
                f"the full run (150-300 images, 3 seeds, lexical-frequency control).")
    # No compositional arm. Is there single-appraisal specificity?
    if single_arms and n1_ok:
        return (f"SINGLE-APPRAISAL SPECIFICITY, NOT COMPOSITIONAL ({len(single_arms)} arms: "
                f"{single_str}). The matched-norm control attributes each specific emotion to ONE "
                f"appraisal direction, not the combination (combo beats matched singles in "
                f"{ns}/5). This extends Stage D from valence to specific NEGATIVE emotions "
                f"(a real positive) but does NOT support compositional emotion synthesis — likely "
                f"bottlenecked by appraisal entanglement (self_responsblt≈valence; 7/15 |cos|>0.6). "
                f"Honest headline: individual appraisals carry cross-modal emotion specificity; "
                f"combining them does not compose into finer emotions (guilt/pride/surprise fail). "
                f"Next: build de-correlated directions (partial-out valence / orthogonalized probes) "
                f"before claiming any compositionality.")
    if signal_arms and n1_ok:
        return (f"WEAK/AMBIGUOUS — {len(signal_arms)} arms move a specific emotion monotonically "
                f"({signal_arms}) but neither the combination nor a clean single component owns the "
                f"target under the matched-norm control. Treat as inconclusive; de-entangle directions.")
    return ("NULL — no arm elevates its specific target above the other emotions. Re-verify the "
            "Stage D pleasantness slope on these 30 images as a sanity gate; if it holds, report "
            "the combo-null honestly.")


def _plot(mean_delta, arm_stats, betas, suf=""):
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
    fig.suptitle(f"Stage E — appraisal-specific emotion steering (combo vs components vs random)"
                 f"{'  [decorrelated]' if suf else ''}")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"stage_e_combo_pilot{suf}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage E step 4 — analysis + decision D1")
    ap.add_argument("--config", default="config/stage_e.yaml")
    ap.add_argument("--decorrelate", choices=["none", "valence"], default="none",
                    help="analyze the matching variant produced by stage_e_combo --decorrelate")
    args = ap.parse_args()
    run(decorrelate=args.decorrelate)


if __name__ == "__main__":
    main()
