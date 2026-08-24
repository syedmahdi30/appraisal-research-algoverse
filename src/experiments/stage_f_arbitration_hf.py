"""Stage F conflict steering on raw HuggingFace — re-score the +0.215 arbitration slope.

WHY. This is the paper's causal capstone under conflict: a direction learned only from TEXT, injected
at resid_post L18 while an image and a sentence disagree, still moves the model's output. It is the
last number in §7 measured on the superseded stack.

It is a SLOPE, which is the measurement class that survived everywhere it was checked — Stage D's
no-conflict slopes reproduced to within 2-3% even though the base valence differed between stacks
(+0.0987 raw HF vs +0.1799 published), because a slope subtracts off the offset the bug introduced.
So we expect it to hold. The reason to measure it anyway is that the denominator moved: the paper
normalizes this slope against Stage D's no-conflict slope, which is now +0.336 rather than +0.329, so
even an unchanged numerator changes the reported ratio.

CONVENTIONS, kept identical to the runs this must be comparable with:
  * the steering vector is the RAW text-derived pleasantness Δμ at resid_post L18 (not unit-normalized)
    — `stage_f_conflict.pleasantness_dmu` and `stage_d_steering_hf.build_directions` agree on this, so
    a given β means the same thing across all three runs;
  * β = 0 is measured once per cell and every steered row is differenced against ITS OWN cell, never
    against an image mean — EMOTIC is per-person so `image_path` recurs and cannot key the baseline;
  * rows are written per cell as [β=0, then the βs], which is the layout
    `analyze_stage_f._arbitration` relies on to label cells via `(beta == 0).cumsum()`.

The output parquet matches the bridge schema exactly, so the existing analyzer consumes it unchanged.

    python -m src.experiments.stage_f_arbitration_hf --verify-dir   # check the Δμ site first
    python -m src.experiments.stage_f_arbitration_hf

Published reference (bridge): behavioral-valence slope +0.215, about 65% of the no-conflict slope.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import context_prompt, sample_contexts
from ..data.labels import EMOTION_LABELS
from ..paths import STAGE_A_DIR, STAGE_F_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_c_transfer_hf import last_token_tap, load_hf
from .stage_d_steering_hf import REFERENCE_DMU_NORM, build_directions, steer
from .stage_f_patching_hf import encode
from .shared.readouts import closed_vocab_logprobs, closed_vocab_valence, first_content_token_ids
from .shared.sampling import select_extreme_rows

# Stage D no-conflict slope on raw HF (results/stage_d/steering_metrics_hf.json), the denominator the
# paper normalizes this slope against. Published bridge value was +0.3293.
STAGE_D_SLOPE_HF = 0.33602


def select_conflict_cells(split: str, n_images: int) -> pd.DataFrame:
    """Both valence extremes, each to be paired with the OPPOSING context polarity."""
    from ..data.emotic import load_split as load_emotic_split
    df = load_emotic_split(split).reset_index(drop=True)
    return select_extreme_rows(df, n_images)


def run(config_path: str, limit_override: int | None = None, verify_only: bool = False) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("steering_layers", [18])[0])
    crit = int(cfg.get("critical_layer", 18))
    betas = [int(b) for b in cfg.get("betas", [-3, -2, -1, 1, 2, 3])]
    seed = int(cfg.get("seed", 0))
    n_dir = int(cfg.get("n_dir", 1200))
    n_images = limit_override or int(cfg.get("n_images", 150))

    probes = load_probes(STAGE_A_DIR / "probes.npz")
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    model, processor = load_hf(cfg.get("model", "google/gemma-3-4b-it"))
    tok_ids = first_content_token_ids(processor)

    # Build the text-derived Δμ and check it against the published norms before spending the sweep.
    # A wrong resid site does not error; it just steers with a vector that means nothing.
    dmu, norms, n_used = build_directions(model, processor, layer, n_dir, seed, ("pleasantness",))
    ref = REFERENCE_DMU_NORM["pleasantness"]
    rel = abs(norms["pleasantness"] - ref) / ref
    site_ok = rel < 0.05
    print(f"\n  Δμ(pleasantness) norm {norms['pleasantness']:.3f} vs published {ref:.3f} "
          f"(rel err {rel:.2%}) -> {'OK' if site_ok else 'SUSPECT'}")
    if not site_ok:
        raise RuntimeError("Δμ norm does not match the published site — the hooked module is not the "
                           "residual stream the direction was defined on; refusing to sweep.")
    if verify_only:
        return {"dmu_norm": norms["pleasantness"], "published": ref, "rel_err": rel, "n_dir": n_used}

    z = torch.tensor(dmu["pleasantness"], dtype=torch.float32, device=model.device)
    ctx = sample_contexts(seed)
    opposing = {"positive": ("negative", ctx["negative"]), "negative": ("positive", ctx["positive"])}
    sel = select_conflict_cells(cfg.get("split", "test"), n_images)

    def readout(enc, store):
        with torch.no_grad():
            out = model(**enc)
        last = out.logits[0, -1].float()
        act = np.asarray(store[0], dtype=np.float32)
        return (float(predict(act[None, :], coef, inter)[0]),
                closed_vocab_valence(last, tok_ids), closed_vocab_logprobs(last, tok_ids))

    rows, n_skip, n_cells = [], 0, 0
    with last_token_tap(model, crit, "post_attention_layernorm") as store:
        for _, r in tqdm(list(sel.iterrows()), desc="stage-f arbitrate (raw HF)"):
            ctx_code, sentence = opposing[r["image_group"]]
            try:
                img = Image.open(r["image_path"]).convert("RGB")
            except (FileNotFoundError, OSError):
                n_skip += 1
                continue
            enc = encode(processor, img, context_prompt(sentence), model.device)
            n_cells += 1
            p0, v0, lp0 = readout(enc, store)                       # β = 0 first: cell baseline
            rows.append(_row(r, ctx_code, sentence, 0, p0, v0, lp0))
            for b in betas:
                with steer(model, layer, z, b):
                    p, v, lp = readout(enc, store)
                rows.append(_row(r, ctx_code, sentence, b, p, v, lp))

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "arbitration_hf.parquet")

    from .analyze_stage_f import _arbitration
    arb = _arbitration(df, betas)
    ratio = arb["valence_slope"] / STAGE_D_SLOPE_HF if STAGE_D_SLOPE_HF else float("nan")

    metrics = {
        "run": run_stamp(), "git": git_hash(), "stack": "raw_hf", "layer": layer, "betas": betas,
        "seed": seed, "n_dir": n_used, "n_incongruent_cells": n_cells, "n_skipped": n_skip,
        "n_forwards": int(len(rows)), "dmu_norm": norms["pleasantness"],
        "site_check": {"published": ref, "rel_err": rel, "ok": site_ok},
        "arbitration": arb, "stage_d_slope_hf": STAGE_D_SLOPE_HF,
        "fraction_of_no_conflict_slope": ratio,
        "published_conflict_slope": 0.215,
        "note": ("PRIMARY = behavioral valence, which is downstream of the resid_post injection. The "
                 "probe read-out at attn_out L18 is UPSTREAM of it and is therefore invariant to "
                 "steering by construction — recorded to demonstrate that, not to score."),
    }
    save_json(metrics, STAGE_F_DIR / "arbitration_hf_metrics.json")

    print(f"\nStage F arbitration (RAW HF) — {n_cells} incongruent cells x ({len(betas)}+1) β "
          f"= {len(rows)} forwards ({n_skip} skipped).")
    print(f"  {'β':>4s} {'Δ valence':>11s} {'Δ probe':>10s}")
    for b in sorted(arb["valence"]):
        print(f"  {b:>4d} {arb['valence'][b]:>+11.4f} {arb['probe'][b]:>+10.4f}")
    print(f"\n  behavioral-valence slope {arb['valence_slope']:+.4f}   (published bridge +0.215)")
    print(f"  probe slope {arb['probe_slope']:+.4f}  (expected ~0 — upstream of the injection)")
    print(f"  = {ratio:.0%} of the raw-HF no-conflict slope ({STAGE_D_SLOPE_HF:+.4f}); "
          f"the paper reported 65% of +0.329")
    print(f"  data -> {STAGE_F_DIR/'arbitration_hf.parquet'}")
    return metrics


def _row(r, ctx_code, sentence, beta, probe, val, lp):
    return {"image_path": r["image_path"], "image_valence": float(r["valence"]),
            "image_group": r["image_group"], "context_code": ctx_code, "context": sentence,
            "beta": beta, "probe_readout": probe, "valence": val,
            **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F conflict steering on raw HF")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="incongruent cell count (default 150)")
    ap.add_argument("--verify-dir", action="store_true",
                    help="build the Δμ, check its norm against the published site, and exit")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit, verify_only=args.verify_dir)


if __name__ == "__main__":
    main()
