"""Stage E, step 3 — appraisal-specific emotion steering pilot (pilot-plan-stage-e-f.md).

Mirrors stage_d_steering.py, but instead of moving valence with ONE appraisal direction it
COMBINES two text-derived Δμ directions per appraisal theory (e.g. +unpleasant +other-blame ->
anger) and asks whether the predicted SPECIFIC emotion gains probability under image input, above
its two components and above a perturbation null. Directions are frozen from Stage A / E1; the
cross-modal claim stays text -> image; layer 18, resid_post, last prompt token only.

Per forward we record the closed-vocab valence AND all 13 emotion log-probs (log-softmax over the
same 13-label set — see stage_a_steering.emotion_logprobs). The β=0 baseline is computed ONCE per
image and reused; Δ log-prob vs that baseline is the Stage E signal, analyzed in analyze_stage_e.

Arms (15): A1-A5 (congruent theory), N1/N2 (controls), S1-S6 (single appraisals), R (two-
orthogonal-random null), A1-raw (unrescaled Δμ sum, magnitude check). See stage_e_arms.py.

Run on the A100 with HF_TOKEN + EMOTIC. Requires results/stage_e/directions.npz (run
stage_e_directions first). `--limit N` uses the first N images. No autoregressive generation.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import make_steer_hook, resid_post_name
from ..bridge.multimodal import build_image_inputs
from ..data.emotic import load_split as load_emotic_split
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import FIGURES_DIR, STAGE_E_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_a_steering import emotion_logprobs, emotion_token_ids, valence_score
from .stage_e_arms import ARMS, COMBO_ARMS, CONGRUENT_ARMS, arm_order


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


VALENCE_APPRAISALS = ("pleasantness", "unpleasantness")


def _cosmat(dmu: dict, appraisals) -> dict:
    """Nested pairwise cosine matrix of the unit directions."""
    U = {a: _unit(dmu[a]) for a in appraisals}
    return {a: {b: float(U[a] @ U[b]) for b in appraisals} for a in appraisals}


def decorrelate_valence(dmu: dict):
    """Residualize every NON-valence appraisal Δμ against the bipolar valence axis.

    Every Stage E combo is (valence axis) + (a non-valence appraisal); `self_responsblt` etc. are so
    entangled with valence (|cos|~0.8) that their raw Δμ is effectively a valence direction, so the
    pilot could not tell composition from "a bigger valence nudge". Here the valence axis is
    v = unit(unit(Δμ_pleasant) − unit(Δμ_unpleasant)); each non-valence appraisal keeps only its
    component ORTHOGONAL to v. The valence appraisals themselves are left raw (they ARE the axis).
    Returns (new_dmu, v).
    """
    p, u = _unit(dmu["pleasantness"]), _unit(dmu["unpleasantness"])
    v = _unit(p - u)
    out = {}
    for a, vec in dmu.items():
        out[a] = vec.astype(np.float32) if a in VALENCE_APPRAISALS \
            else (vec - float(vec @ v) * v).astype(np.float32)
    return out, v


def build_arm_vectors(dmu: dict, norms: dict, cos: dict, thr: float, seed: int,
                      matched_norm: bool = True):
    """Return {arm: np.float32[d]} steering vectors and per-arm metadata (E3 rule).

    Combo arms: sum the two SIGNED unit Δμ, rescale so the sum's norm equals the mean of the two
    component ‖Δμ‖. Entangled pairs (|cos|>thr) Gram-Schmidt the 2nd unit against the 1st first.
    Singles: the raw Δμ (natural scale, as Stage D). R: two orthogonal random units, scaled to the
    mean congruent-combo norm. A1-raw: raw signed Δμ sum, unrescaled. With `matched_norm`, also
    emit per-congruent-arm single directions rescaled to that combo's norm (`{ARM}~{appraisal}`) —
    the magnitude-controlled combo-vs-single specificity control.
    """
    vecs, meta = {}, {}
    congruent_norms, combo_norm = [], {}
    for name, spec in ((n, ARMS[n]) for n in arm_order() if n in ARMS):
        combo = spec["combo"]
        if len(combo) == 1:  # single appraisal -> raw Δμ (natural units)
            a, s = combo[0]
            vecs[name] = (s * dmu[a]).astype(np.float32)
            meta[name] = {"rule": "single_raw_dmu", "gram_schmidt": False}
            continue
        (a1, s1), (a2, s2) = combo
        u1, u2 = s1 * _unit(dmu[a1]), s2 * _unit(dmu[a2])
        gs = abs(cos[a1][a2]) > thr
        if gs:  # orthogonalize the second against the first before summing
            u2 = u2 - float(u2 @ u1) * u1
        summed = u1 + u2
        target_norm = 0.5 * (norms[a1] + norms[a2])
        vecs[name] = (_unit(summed) * target_norm).astype(np.float32)
        meta[name] = {"rule": "sum_units_rescaled", "gram_schmidt": bool(gs),
                      "target_norm": float(target_norm)}
        combo_norm[name] = float(target_norm)
        if name in CONGRUENT_ARMS:
            congruent_norms.append(target_norm)

    # Matched-norm single controls: each congruent arm's two component directions, scaled to THAT
    # combo's norm. combo-vs-matched-single then differs ONLY in direction, not magnitude — the
    # clean test of whether the COMBINATION (not just a bigger nudge) creates the specific emotion.
    if matched_norm:
        for arm in CONGRUENT_ARMS:
            if arm not in combo_norm:
                continue
            for a, s in ARMS[arm]["combo"]:
                nm = f"{arm}~{a}"
                vecs[nm] = (s * _unit(dmu[a]) * combo_norm[arm]).astype(np.float32)
                meta[nm] = {"rule": "matched_norm_single", "parent": arm, "appraisal": a,
                            "sign": int(s), "target_norm": combo_norm[arm], "gram_schmidt": False}

    ref_norm = float(np.mean(congruent_norms)) if congruent_norms else 1.0
    # R — two orthogonal random unit vectors, scaled to the mean congruent-combo norm.
    rng = np.random.default_rng(seed)
    d = len(next(iter(dmu.values())))
    r1 = _unit(rng.standard_normal(d).astype(np.float32))
    r2 = rng.standard_normal(d).astype(np.float32)
    r2 = _unit(r2 - float(r2 @ r1) * r1)
    vecs["R"] = (_unit(r1 + r2) * ref_norm).astype(np.float32)
    meta["R"] = {"rule": "two_orthogonal_random", "target_norm": ref_norm, "gram_schmidt": False}
    # A1-raw — raw signed Δμ sum for A1, unrescaled (magnitude sensitivity).
    (a1, s1), (a2, s2) = ARMS["A1"]["combo"]
    vecs["A1-raw"] = (s1 * dmu[a1] + s2 * dmu[a2]).astype(np.float32)
    meta["A1-raw"] = {"rule": "raw_dmu_sum_unrescaled", "gram_schmidt": False}
    return vecs, meta


def _readout(bridge, ids, pv, hooks, tok_ids):
    """(valence, {emotion: logprob}) at the last token under image input (+ optional steer)."""
    with torch.no_grad():
        logits = bridge.run_with_hooks(ids, pixel_values=pv, fwd_hooks=hooks)
    last = logits[0, -1]
    return valence_score(last, tok_ids), emotion_logprobs(last, tok_ids)


def run(config_path: str, limit_override: int | None = None,
        decorrelate_override: str | None = None) -> dict:
    import pandas as pd

    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("steering_layers", [18])[0])
    betas = list(cfg.get("betas", [-3, -2, -1, 1, 2, 3]))
    n_images = limit_override or int(cfg.get("n_images", 30))
    seed = int(cfg.get("seed", 0))
    thr = float(cfg.get("cos_flag_threshold", 0.6))
    matched_norm = bool(cfg.get("matched_norm_control", True))
    decorrelate = decorrelate_override or str(cfg.get("decorrelate", "none"))

    dpath = STAGE_E_DIR / "directions.npz"
    if not dpath.exists():
        raise FileNotFoundError(f"{dpath} missing — run `python -m src.experiments.stage_e_directions` first.")
    dz = np.load(dpath, allow_pickle=True)
    ap = [str(a) for a in dz["appraisals"]]
    dmu = {a: dz["dmu"][i].astype(np.float32) for i, a in enumerate(ap)}
    val_cos_before = {a: float(_unit(dmu[a]) @ _unit(_unit(dmu["pleasantness"]) - _unit(dmu["unpleasantness"])))
                      for a in ap if a not in VALENCE_APPRAISALS}
    if decorrelate == "valence":
        dmu, _vax = decorrelate_valence(dmu)
        val_cos_after = {a: float(_unit(dmu[a]) @ _unit(_unit(dmu["pleasantness"]) - _unit(dmu["unpleasantness"])))
                         for a in ap if a not in VALENCE_APPRAISALS}
        print(f"  decorrelate=valence: |cos with valence axis| non-valence appraisals "
              f"max {max(abs(v) for v in val_cos_before.values()):.2f} -> "
              f"{max(abs(v) for v in val_cos_after.values()):.2f}")
    elif decorrelate != "none":
        raise ValueError(f"unknown decorrelate={decorrelate!r}; expected 'none' or 'valence'")
    # (re)compute norms + cosine from the (possibly residualized) directions
    norms = {a: float(np.linalg.norm(dmu[a])) for a in ap}
    cos = _cosmat(dmu, ap)

    # Final target per arm: empirical (E2) if present, else theory (stage_e_arms).
    prof_path = STAGE_E_DIR / "appraisal_profiles.json"
    targets = {}
    if prof_path.exists():
        prof = load_config(prof_path)["arms"]
        for name in COMBO_ARMS:
            targets[name] = prof.get(name, {}).get("final_target") or ARMS[name]["target"]
    else:
        for name in COMBO_ARMS:
            targets[name] = ARMS[name]["target"]

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    tok_report = verify_label_tokenization(bridge.tokenizer)
    multi = {w: r for w, r in tok_report.items() if not r["single_token"]}
    if multi:
        print(f"  WARNING: {len(multi)} emotion label(s) are multi-token on this tokenizer "
              f"(scored by first sub-token): {list(multi)}")
    dev = next(bridge.parameters()).device

    vecs, arm_meta = build_arm_vectors(dmu, norms, cos, thr, seed, matched_norm=matched_norm)
    scaled = {name: torch.tensor(v, dtype=torch.float32, device=dev) for name, v in vecs.items()}

    # Images: reproduce the Stage D selection (sample 150, seed 0) then take the first n_images,
    # so the pilot set is a strict subset of the Stage D 150.
    idf = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    idf = idf.sample(n=min(150, len(idf)), random_state=seed).reset_index(drop=True).head(n_images)

    rows, base_vals, n_ok, n_skip = [], [], 0, 0
    hook_name = resid_post_name(layer)
    for path in tqdm(idf["image_path"].tolist(), desc="stage-e steering"):
        try:
            inputs = build_image_inputs(bridge, Image.open(path).convert("RGB"))
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        ids, pv = inputs["input_ids"], inputs["pixel_values"]
        bval, blp = _readout(bridge, ids, pv, [], tok_ids)
        base_vals.append(bval)
        rows.append({"image_path": path, "arm": "_base", "beta": 0, "valence": bval,
                     **{f"lp_{w}": blp[w] for w in EMOTION_LABELS}})
        for name, z in scaled.items():
            for b in betas:
                val, lp = _readout(bridge, ids, pv, [(hook_name, make_steer_hook(z, b))], tok_ids)
                rows.append({"image_path": path, "arm": name, "beta": b, "valence": val,
                             **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})
        n_ok += 1

    # suffix outputs by variant so the de-correlated run sits beside the raw one (not clobber it).
    suf = "" if decorrelate == "none" else f"_{decorrelate}"
    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_E_DIR / f"combo_pilot{suf}.parquet")

    metrics = {
        "run": run_stamp(), "git": git_hash(), "method": "appraisal_specific_combo_steering",
        "layer": layer, "betas": betas, "n_images": n_ok, "n_skipped": n_skip,
        "seed": seed, "base_valence_mean": float(np.mean(base_vals)) if base_vals else None,
        "logprob_readout": "log_softmax over the 13 closed-vocab emotion labels",
        "decorrelate": decorrelate, "arm_targets": targets,
        "arm_meta": arm_meta,
        "arm_norms": {name: float(np.linalg.norm(v)) for name, v in vecs.items()},
        "cos_matrix": {a: cos[a] for a in ap}, "cos_flag_threshold": thr,
        "tokenization_multi_token": multi,
        "emotion_labels": list(EMOTION_LABELS),
    }
    save_json(metrics, STAGE_E_DIR / f"combo_pilot_metrics{suf}.json")
    print(f"\nStage E combo pilot [decorrelate={decorrelate}] — L{layer}, {n_ok} images "
          f"({n_skip} skipped), {len(scaled)} arms x {len(betas)} betas + baseline.")
    print(f"  targets: " + ", ".join(f"{k}->{v}" for k, v in targets.items()))
    print(f"  data -> {STAGE_E_DIR/f'combo_pilot{suf}.parquet'}   "
          f"metrics -> {STAGE_E_DIR/f'combo_pilot_metrics{suf}.json'}")
    print(f"  NEXT: python -m src.experiments.analyze_stage_e"
          + (f" --decorrelate {decorrelate}" if suf else ""))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage E step 3 — appraisal-specific combo steering")
    ap.add_argument("--config", default="config/stage_e.yaml")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N images (dry run)")
    ap.add_argument("--decorrelate", choices=["none", "valence"], default=None,
                    help="residualize non-valence appraisals against the valence axis (overrides config)")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit, decorrelate_override=args.decorrelate)


if __name__ == "__main__":
    main()
