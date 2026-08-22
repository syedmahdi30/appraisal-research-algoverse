"""Stage F CROSS-IMAGE patching on raw HuggingFace — re-score where visual valence lives.

WHY THIS ONE IS THE RISKY RE-SCORE. The same-image re-score (`stage_f_patching_hf`) is safe on paper:
it is a ratio of differences read off the L18 probe, at a site whose cosine to raw HF is 0.980. This
experiment is not. `_probe_valid` in the bridge module is False for any band reaching L18 or beyond
(the probe tap is upstream of the injection, so probe recovery is invariant-by-construction there),
which is why the published §6.3 table is scored on BEHAVIOURAL VALENCE across all three bands --- and
behaviour is exactly what the bridge corrupts (6.15 nats, argmax flips, no-context image separation
AUC 0.788 vs 0.982; see docs/bridge-bug-2026-08-22.md).

A recovery ratio on behavioural valence sits between the two categories the re-runs established.
It is a ratio of differences, which is the property that saved Stage C's correlation and Stage D's
slope --- but it is computed on the bounded, corrupted output scale, which is what killed the override
rates. There is no principled way to call it from the armchair, so it gets measured. The 18--28 band
in particular is the one carrying the paper's "visual valence moves from image tokens into text
states over depth" claim (image-token recovery 9%, text 68%), and it has no valid probe column to
fall back on.

The 0--12 and 13--17 bands DO have a valid probe column. This module reports probe and valence side by
side for every band, so a divergence between them localises the damage instead of just flagging it.

Shared with the bridge version by import, never reimplemented: `_cross_groups`, `_recovery` (with its
clustered bootstrap and shared resamples), `_metrics`, `_print`, `_probe_valid`. Aggregation must be
identical for a re-score to mean anything. Model plumbing comes from `stage_f_patching_hf`.

    python -m src.experiments.stage_f_cross_patching_hf --layers 0-12
    python -m src.experiments.stage_f_cross_patching_hf --layers 13-17
    python -m src.experiments.stage_f_cross_patching_hf --layers 18-28
    python -m src.experiments.stage_f_cross_patching_hf --reanalyze --layers 18-28   # CPU

Published reference (bridge, behavioural valence): image tokens 80% / 66% / 9% and all-non-image-text
10% / 65% / 68% for bands 0--12, 13--17, 18--28.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import context_prompt
from ..data.emotic import load_split as load_emotic_split
from ..paths import STAGE_A_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_c_transfer_hf import CANDIDATE_TAPS, last_token_tap, load_hf
from .stage_f_attribution import segment_positions
from .stage_f_cross_patching import (CONTEXT_BANKS, GROUPS, _cross_groups, _metrics, _print,
                                     _probe_valid, _recovery)
from .stage_f_patching_hf import _TokShim, encode, patch_resid, readout, resid_capture_full
from .stage_f_qwen import emotion_token_ids

PARQUET = STAGE_F_DIR / "cross_patching_hf.parquet"
METRICS = STAGE_F_DIR / "cross_patching_hf_metrics.json"


def select_pairs(split: str, n_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank-matched (highest-valence, lowest-valence) EMOTIC pairs, maximising the image-valence gap.

    Mirrors `stage_f_conflict.select_extreme_images` + the bridge module's rank pairing, inlined so
    this module has no path back to the bridge stack.
    """
    df = load_emotic_split(split).reset_index(drop=True)
    df = df[np.isfinite(df["valence"].to_numpy(dtype=float))].sort_values("valence")
    pos = df.tail(n_pairs).sort_values("valence", ascending=False).reset_index(drop=True)
    neg = df.head(n_pairs).sort_values("valence").reset_index(drop=True)
    return pos, neg


def run(config_path: str, limit_override: int | None = None, layers_override: str | None = None,
        context_polarity: str = "neutral", context_idx: int = 0,
        tap: str = "post_attention_layernorm") -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    crit = int(cfg.get("critical_layer", 18))
    onset = int(cfg.get("patch_onset_layer", 13))
    if layers_override:
        a, b = (int(x) for x in layers_override.split("-"))
        patch_layers = list(range(a, b + 1))
    else:
        patch_layers = list(range(onset, crit))
    n_pairs = limit_override or int(cfg.get("patch_n_images", 60))
    ctx = CONTEXT_BANKS[context_polarity][context_idx]

    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    pos, neg = select_pairs(cfg.get("split", "test"), n_pairs)
    n_pairs = min(n_pairs, len(pos), len(neg))

    model, processor = load_hf(cfg.get("model", "google/gemma-3-4b-it"))
    tok_ids = emotion_token_ids(processor)
    shim = _TokShim(processor.tokenizer)
    if not _probe_valid(patch_layers, crit):
        print(f"  NOTE: band {patch_layers[0]}-{patch_layers[-1]} reaches the L{crit} probe tap, so the "
              f"probe column is invariant-by-construction. This band is scored on BEHAVIOURAL VALENCE "
              f"— the readout the bridge corrupts, and the reason this re-run exists.")

    rows, n_skip, seg_bad = [], 0, 0
    with last_token_tap(model, crit, tap) as store:
        for k in tqdm(range(n_pairs), desc="stage-f cross-patching (raw HF)"):
            try:
                dimg = Image.open(pos.loc[k, "image_path"]).convert("RGB")
                rimg = Image.open(neg.loc[k, "image_path"]).convert("RGB")
            except (FileNotFoundError, OSError):
                n_skip += 1
                continue
            enc_d = encode(processor, dimg, context_prompt(ctx), model.device)   # donor  = POSITIVE img
            enc_r = encode(processor, rimg, context_prompt(ctx), model.device)   # recip. = NEGATIVE img
            if enc_d["input_ids"].shape[-1] != enc_r["input_ids"].shape[-1]:
                seg_bad += 1                       # identical text ⇒ should never differ
                continue
            seg = segment_positions(shim, enc_r["input_ids"].cpu())   # identical for both runs
            groups, ok = _cross_groups(seg)
            if not all(ok.get(g) for g in GROUPS):
                seg_bad += 1
                continue

            with resid_capture_full(model, patch_layers) as cap:
                pos_probe, pos_val = readout(model, enc_d, store, coef, inter, tok_ids)
                donor = dict(cap)
            neg_probe, neg_val = readout(model, enc_r, store, coef, inter, tok_ids)

            row = {"donor_path": pos.loc[k, "image_path"], "recipient_path": neg.loc[k, "image_path"],
                   "donor_valence": float(pos.loc[k, "valence"]),
                   "recipient_valence": float(neg.loc[k, "valence"]),
                   "pos_probe": pos_probe, "neg_probe": neg_probe,
                   "pos_val": pos_val, "neg_val": neg_val}
            for g in GROUPS:
                idx = groups[g]
                with patch_resid(model, donor, idx, idx):   # identical input_ids ⇒ donor idx == recip idx
                    p_probe, p_val = readout(model, enc_r, store, coef, inter, tok_ids)
                row[f"patch_{g}_probe"] = p_probe
                row[f"patch_{g}_val"] = p_val
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(PARQUET)
    rec = _recovery(df)
    metrics = _metrics(rec, patch_layers, crit, ctx, context_polarity, len(rows), n_skip, seg_bad)
    metrics["stack"] = "raw_hf"
    metrics["tap"] = tap
    save_json(metrics, METRICS)
    _print(rec, metrics, patch_layers)
    print(f"  (raw HF re-score; data -> {PARQUET}   metrics -> {METRICS})")
    return metrics


def reanalyze(config_path: str, layers_override: str | None = None) -> dict:
    """Recompute recovery + bootstrap CIs from the saved parquet. CPU, no model."""
    load_config(config_path)
    if not PARQUET.exists():
        raise FileNotFoundError(f"{PARQUET} missing — run the raw-HF cross-patching pass first.")
    df = pd.read_parquet(PARQUET)
    old = load_config(METRICS) if METRICS.exists() else {}
    if layers_override:
        a, b = (int(x) for x in layers_override.split("-"))
        patch_layers = list(range(a, b + 1))
    else:
        patch_layers = old.get("patch_layers", []) or [0, 0]
    rec = _recovery(df)
    metrics = _metrics(rec, patch_layers, old.get("critical_layer", 18), old.get("context", ""),
                       old.get("context_polarity", "neutral"), int(len(df)),
                       old.get("n_skipped", 0), old.get("n_segmentation_dropped", 0))
    metrics["stack"] = "raw_hf"
    save_json(metrics, METRICS)
    _print(rec, metrics, patch_layers)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage F cross-image patching on raw HF (bridge-free re-score of §6.3)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="donor/recipient pair count (default 60)")
    ap.add_argument("--layers", type=str, default=None, help="patch band 'a-b' (default onset..L17)")
    ap.add_argument("--context", choices=list(CONTEXT_BANKS), default="neutral",
                    help="fixed context polarity held across donor/recipient (default neutral)")
    ap.add_argument("--context-idx", type=int, default=0, help="index within the chosen context bank")
    ap.add_argument("--tap", default="post_attention_layernorm", choices=CANDIDATE_TAPS,
                    help="raw-HF equivalent of the bridge's blocks.L.hook_attn_out")
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute recovery + CIs from the saved parquet (CPU, no model)")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze(args.config, layers_override=args.layers)
    else:
        run(args.config, limit_override=args.limit, layers_override=args.layers,
            context_polarity=args.context, context_idx=args.context_idx, tap=args.tap)


if __name__ == "__main__":
    main()
