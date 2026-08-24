"""Stage F — modality conflict pilot (pilot-plan-stage-e-f.md).

Pits a one-sentence TEXT context against an IMAGE's own valence and asks which the shared
appraisal read-out follows, then whether pleasantness Δμ steering can arbitrate.

Base pass (default): the 20 highest- and 20 lowest-EMOTIC-valence test images (true extremes,
taken directly from the EMOTIC test parquet — self-contained). For each: no-context, one positive,
one negative, one neutral context (4 forwards, 160 total). Per forward we record (a) the FROZEN
Stage A pleasantness probe read-out at blocks.18.hook_attn_out (Stage C machinery, unchanged),
(b) behavioral valence from the logits, (c) all 13 emotion log-probs.

Arbitration pass (--arbitrate): on the incongruent cells (positive image + negative context, and
negative image + positive context) sweep the pleasantness Δμ at resid_post L18, β {-3..+3}.
PRIMARY metric = behavioral valence (downstream of the injection, like Stage D). The probe
read-out is ALSO recorded, but note it is captured at hook_attn_out L18, which is UPSTREAM of the
resid_post L18 injection within the same block, so it is invariant to steering by construction —
we record it to demonstrate that, not to score arbitration on it.

Run on the A100 with HF_TOKEN + EMOTIC. Never re-fits probes. `--limit N` sets the image count.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import make_steer_hook, resid_post_name
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      STRONG_CONTEXT, TEXT_CODE, build_conditions, context_prompt,
                                      sample_contexts)
from ..data.emotic import load_split as load_emotic_split
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import STAGE_E_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .shared.patching import stash_activation
from .shared.readouts import bridge_probe_and_logits
from .shared.sampling import select_extreme_rows
from .stage_a_steering import emotion_token_ids


# --------------------------------------------------------------------------- image selection
def select_extreme_images(split: str, n: int):
    """Return the n/2 highest- and n/2 lowest-valence EMOTIC rows, tagged by image_group."""
    df = load_emotic_split(split).reset_index(drop=True)
    return select_extreme_rows(df, n)


# --------------------------------------------------------------------------- read-out
def _stash_hook(store: dict):
    return stash_activation(store)


def _probe_and_logits(bridge, ids, pv, name, coef, inter, tok_ids, steer_hooks=None):
    return bridge_probe_and_logits(
        bridge, ids, pv, name, coef, inter, tok_ids, steering_hooks=steer_hooks
    )


# --------------------------------------------------------------------------- pleasantness Δμ
def pleasantness_dmu(bridge, layer, n_dir, seed):
    """Text-derived pleasantness Δμ at resid_post `layer`. Reuse Stage E directions.npz if present."""
    dpath = STAGE_E_DIR / "directions.npz"
    if dpath.exists():
        dz = np.load(dpath, allow_pickle=True)
        ap = [str(a) for a in dz["appraisals"]]
        if "pleasantness" in ap and int(dz["layer"]) == layer:
            return dz["dmu"][ap.index("pleasantness")].astype(np.float32)
    # fall back to building it (Stage D recipe)
    from ..data.crowd_envent import load_split, sample_tak_subset
    from .stage_a_steering_v2 import diff_of_means, extract_resid
    dtr = sample_tak_subset(load_split("train", seed=seed), seed=seed).head(n_dir)
    acts = extract_resid(bridge, dtr["text"].tolist(), [layer], "build-pleasantness")[layer]
    v = diff_of_means(acts, dtr["pleasantness"].to_numpy())
    if v is None:
        raise ValueError("not enough high/low pleasantness examples — widen n_dir")
    return v.astype(np.float32)


# --------------------------------------------------------------------------- base pass (F2)
def run_base(config_path: str, limit_override: int | None = None, bank: str | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("critical_layer", 18))
    tap = cfg.get("tap", "hook_attn_out")
    seed = int(cfg.get("seed", 0))
    n_images = limit_override or int(cfg.get("n_images", 40))
    bank = bank or cfg.get("bank", "full")
    full_bank = bool(cfg.get("full_context_bank", False)) or bank == "minimal"

    probes = load_probes(STAGE_A_DIR_probes())
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    # conditions: (polarity, ctx_id, sentence). Full run sweeps the whole bank (averaged in analysis);
    # pilot uses one sampled context per polarity.
    ctx = sample_contexts(seed)
    if bank == "minimal":
        conditions = build_conditions("minimal")
    elif full_bank:
        conditions = build_conditions("full")
    else:
        conditions = [("none", "none", None), ("positive", "p", ctx["positive"]),
                      ("negative", "n", ctx["negative"]), ("neutral", "z", ctx["neutral"])]

    sel = select_extreme_images(cfg.get("split", "test"), n_images)
    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    multi = {w: r for w, r in verify_label_tokenization(bridge.tokenizer).items() if not r["single_token"]}
    name = f"blocks.{layer}.{tap}"

    rows, n_skip = [], 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f base"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for cond, ctx_id, sentence in conditions:
            inputs = build_image_inputs(bridge, img, prompt=context_prompt(sentence))
            probe, val, lp = _probe_and_logits(
                bridge, inputs["input_ids"], inputs["pixel_values"], name, coef, inter, tok_ids)
            rows.append({"image_path": r["image_path"], "image_valence": float(r["valence"]),
                         "image_group": r["image_group"], "condition": cond, "context_id": ctx_id,
                         "context": sentence or "", "text_code": TEXT_CODE[cond],
                         "probe_readout": probe, "valence": val,
                         **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    stem = "conflict_minimal" if bank == "minimal" else "conflict_pilot"
    out_pq = STAGE_F_DIR / f"{stem}.parquet"
    df.to_parquet(out_pq)
    metrics = {
        "run": run_stamp(), "git": git_hash(), "layer": layer, "tap": tap, "seed": seed,
        "bank": bank, "n_images": int(sel.shape[0] - n_skip), "n_skipped": n_skip,
        "n_forwards": int(len(rows)), "full_context_bank": full_bank, "contexts": ctx,
        "n_conditions": len(conditions),
        "text_code": TEXT_CODE, "tokenization_multi_token": multi,
        "image_ids": [{"image_path": r["image_path"], "valence": float(r["valence"]),
                       "group": r["image_group"]} for _, r in sel.iterrows()],
    }
    save_json(metrics, STAGE_F_DIR / f"{stem}_metrics.json")
    print(f"\nStage F base [bank={bank}] — {metrics['n_images']} images x "
          f"{len(conditions)} conditions = {len(rows)} forwards ({n_skip} skipped).")
    if bank not in ("full", "minimal"):
        print(f"  contexts: +[{ctx['positive']}]  -[{ctx['negative']}]  0[{ctx['neutral']}]")
    if multi:
        print(f"  WARNING multi-token labels (first sub-token scored): {list(multi)}")
    print(f"  data -> {out_pq}")
    if bank == "minimal":
        print(f"  NEXT: python -m src.experiments.analyze_stage_f --parquet {out_pq.name}")
    else:
        print("  NEXT: python -m src.experiments.stage_f_conflict --arbitrate")
    return metrics


# --------------------------------------------------------------------------- arbitration (F3)
def run_arbitrate(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("steering_layers", [cfg.get("critical_layer", 18)])[0])
    tap = cfg.get("tap", "hook_attn_out")
    betas = list(cfg.get("betas", [-3, -2, -1, 1, 2, 3]))
    seed = int(cfg.get("seed", 0))
    n_dir = int(cfg.get("n_dir", 1200))
    n_images = limit_override or int(cfg.get("n_images", 40))

    probes = load_probes(STAGE_A_DIR_probes())
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    ctx = sample_contexts(seed)
    sel = select_extreme_images(cfg.get("split", "test"), n_images)
    # incongruent context = the polarity that OPPOSES the image's valence group.
    opposing = {"positive": ("negative", ctx["negative"]), "negative": ("positive", ctx["positive"])}

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    name = f"blocks.{layer}.{tap}"
    dmu = pleasantness_dmu(bridge, layer, n_dir, seed)
    dev = next(bridge.parameters()).device
    z = torch.tensor(dmu, dtype=torch.float32, device=dev)
    hook_name = resid_post_name(layer)

    rows, n_skip, n_cells = [], 0, 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f arbitrate"):
        ctx_code, sentence = opposing[r["image_group"]]
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        inputs = build_image_inputs(bridge, img, prompt=context_prompt(sentence))
        ids, pv = inputs["input_ids"], inputs["pixel_values"]
        n_cells += 1
        # β=0 baseline (no steering) once per cell
        p0, v0, lp0 = _probe_and_logits(bridge, ids, pv, name, coef, inter, tok_ids)
        rows.append(_arb_row(r, ctx_code, sentence, 0, p0, v0, lp0))
        for b in betas:
            p, v, lp = _probe_and_logits(bridge, ids, pv, name, coef, inter, tok_ids,
                                         steer_hooks=[(hook_name, make_steer_hook(z, b))])
            rows.append(_arb_row(r, ctx_code, sentence, b, p, v, lp))

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "arbitration_pilot.parquet")
    metrics = {
        "run": run_stamp(), "git": git_hash(), "layer": layer, "tap": tap, "seed": seed,
        "betas": betas, "n_incongruent_cells": n_cells, "n_skipped": n_skip,
        "n_forwards": int(len(rows)), "dmu_norm": float(np.linalg.norm(dmu)),
        "note": ("PRIMARY = behavioral valence (downstream of resid_post injection). probe_readout "
                 "at attn_out L18 is UPSTREAM of the injection -> invariant to steering by "
                 "construction; recorded to demonstrate, not to score."),
    }
    save_json(metrics, STAGE_F_DIR / "arbitration_metrics.json")
    print(f"\nStage F arbitration — {n_cells} incongruent cells x ({len(betas)}+1) β "
          f"= {len(rows)} forwards ({n_skip} skipped).")
    print(f"  data -> {STAGE_F_DIR/'arbitration_pilot.parquet'}")
    print("  NEXT: python -m src.experiments.analyze_stage_f")
    return metrics


def _arb_row(r, ctx_code, sentence, beta, probe, val, lp):
    return {"image_path": r["image_path"], "image_valence": float(r["valence"]),
            "image_group": r["image_group"], "context_code": ctx_code, "context": sentence,
            "beta": beta, "probe_readout": probe, "valence": val,
            **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}}


def STAGE_A_DIR_probes():
    from ..paths import STAGE_A_DIR
    p = STAGE_A_DIR / "probes.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — Stage A must have saved frozen probes.")
    return p


def main() -> None:  # noqa: D401
    ap = argparse.ArgumentParser(description="Stage F — modality conflict (base + arbitration)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--arbitrate", action="store_true", help="run the steering arbitration pass (F3)")
    ap.add_argument("--bank", choices=["full", "minimal"], default=None,
                    help="context bank for the base pass: 'full' (default) or 'minimal' (T1.1 "
                         "matched valence-only pairs → conflict_minimal.parquet)")
    ap.add_argument("--limit", type=int, default=None, help="use only N images (dry run)")
    args = ap.parse_args()
    if args.arbitrate:
        run_arbitrate(args.config, limit_override=args.limit)
    else:
        run_base(args.config, limit_override=args.limit, bank=args.bank)


if __name__ == "__main__":
    main()
