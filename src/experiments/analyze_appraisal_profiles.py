"""Stage E, step 2 — empirical appraisal->emotion prediction matrix (pilot-plan-stage-e-f.md).

CPU-only. Grounds the theory targets in the actual crowd-enVENT data before we spend GPU on
steering: for each emotion, the mean appraisal profile (z-scored across the 13 emotions); then
for each signed appraisal combo, the emotion that maximizes sum(sign_a * z[emotion, a]) is the
empirically-predicted target. If that disagrees with the Smith & Ellsworth theory target in
stage_e_arms, we USE THE EMPIRICAL TARGET downstream (analyze_stage_e) and log the change here —
so a null verdict later can't be blamed on a wrong theory label.

Saves results/stage_e/appraisal_profiles.json (per-emotion z-profiles, per-arm predicted target
+ runner-up, theory-vs-empirical agreement). No model, no images.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.crowd_envent import load_split
from ..paths import STAGE_E_DIR, ensure_dirs
from .common import git_hash, run_stamp, save_json
from .stage_e_arms import APPRAISALS, ARMS, COMBO_ARMS


def emotion_profiles(df: pd.DataFrame, appraisals=APPRAISALS):
    """Return (emotions, z [n_emotion, n_appraisal]) — appraisal means per emotion, z-scored
    down each appraisal column across emotions."""
    cols = [a for a in appraisals if a in df.columns]
    prof = df.groupby("emotion")[cols].mean()
    z = (prof - prof.mean(axis=0)) / prof.std(axis=0, ddof=0)
    return list(prof.index), cols, z


def predict_arm(z: pd.DataFrame, combo) -> tuple[str, str, dict]:
    """(predicted_target, runner_up, per-emotion score) for a signed appraisal combo."""
    score = pd.Series(0.0, index=z.index)
    for a, sign in combo:
        if a in z.columns:
            score = score + sign * z[a]
    ranked = score.sort_values(ascending=False)
    return str(ranked.index[0]), str(ranked.index[1]), {e: float(s) for e, s in score.items()}


def run(config_path: str | None = None) -> dict:
    ensure_dirs()
    seed = 0
    df = load_split("train", seed=seed)
    emotions, cols, z = emotion_profiles(df)

    arms_out, disagreements = {}, []
    for name in COMBO_ARMS:
        spec = ARMS[name]
        target, runner, score = predict_arm(z, spec["combo"])
        theory = spec["target"]
        alt = spec.get("alt")
        # "agrees" if theory unset (pure control) OR empirical argmax hits theory or its alt.
        agrees = theory is None or target == theory or (alt is not None and target == alt)
        final = target if theory is None else (theory if agrees else target)
        if theory is not None and not agrees:
            disagreements.append(
                {"arm": name, "theory": theory, "empirical": target, "runner_up": runner})
        arms_out[name] = {
            "combo": [[a, s] for a, s in spec["combo"]],
            "theory_target": theory, "theory_alt": alt,
            "empirical_target": target, "runner_up": runner,
            "agrees": bool(agrees), "final_target": final,
            "must_not": spec["must_not"], "scores": score,
        }

    metrics = {
        "run": run_stamp(), "git": git_hash(), "seed": seed, "n_train": int(len(df)),
        "emotions": emotions, "appraisals": cols,
        "z_profiles": {e: {a: float(z.loc[e, a]) for a in cols} for e in emotions},
        "arms": arms_out, "disagreements": disagreements,
    }
    save_json(metrics, STAGE_E_DIR / "appraisal_profiles.json")

    print(f"\nStage E appraisal profiles — crowd-enVENT train (n={len(df)}, "
          f"{len(emotions)} emotions).\n")
    print(f"{'arm':5s} {'combo':38s} {'theory':>9s} {'empirical':>10s} {'runner':>9s}  ok")
    for name, r in arms_out.items():
        combo = " ".join(f"{'+' if s > 0 else '-'}{a[:6]}" for a, s in r["combo"])
        print(f"{name:5s} {combo:38s} {str(r['theory_target']):>9s} "
              f"{r['empirical_target']:>10s} {r['runner_up']:>9s}  {'yes' if r['agrees'] else 'NO'}")
    if disagreements:
        print("\n  Theory<->empirical disagreements (using EMPIRICAL target downstream):")
        for d in disagreements:
            print(f"    {d['arm']}: theory={d['theory']} but empirical argmax={d['empirical']}")
    print(f"\n  profiles -> {STAGE_E_DIR/'appraisal_profiles.json'}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage E step 2 — empirical appraisal->emotion matrix")
    ap.add_argument("--config", default="config/stage_e.yaml")
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
