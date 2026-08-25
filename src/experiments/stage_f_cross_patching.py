"""Stage F — CROSS-IMAGE activation patching: WHERE does the image's own valence live? (T1.2)

The same-image experiment (`stage_f_patching.py`) showed the TEXT context delta is carried by the
text token stream and the image tokens are causally INERT for it. That leaves the mirror question open:
where does the IMAGE's own valence live, and is it read out from the image tokens? This experiment
answers it with the complementary design.

    donor    = a POSITIVE-valence image  + a fixed (neutral) context
    recipient= a NEGATIVE-valence image  + the SAME fixed context

Because the context and prompt are identical, the two runs have BYTE-IDENTICAL `input_ids` — only the
`pixel_values` differ — so every token position (image, context, question, turn scaffold) is trivially
1:1 aligned and even the context tokens are patchable (unlike the same-image design). We overwrite the
recipient's `resid_post` across the band for ONE token group at a time with the donor's values, and
measure how far the L18 pleasantness read-out (and behavioral valence) recovers toward the positive
(donor) image:

    recovery(G) = (patched(G) − neg_img_baseline) / (pos_img_baseline − neg_img_baseline)

Expected two-sided result: if visual valence lives in and is read out from the image tokens, patching
`image` recovers a LARGE share (the mirror of its inertness for text context) while text groups recover
little — i.e. context delta rides the text stream, image valence rides the image tokens. If `image`
recovers little even here, the "image valence lives in image tokens" wording must weaken. `all` (every
token bar the read-out query) is a sanity check that should recover ~100%.

Run on the A100 with HF_TOKEN + EMOTIC. Frozen probe; never re-fit. `--reanalyze` (CPU) recomputes
recovery + bootstrap CIs from the saved parquet. `--limit N`, `--layers a-b`, `--context {neutral,
positive,negative}`.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import context_prompt
from ..data.emotic import load_split as load_emotic_split
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .shared.patching import (CROSS_IMAGE_CONTEXT_BANKS, bridge_patch_hook, cross_image_groups,
                              cross_image_recovery, probe_recovery_valid,
                              segment_prompt_positions)
from .shared.readouts import QUESTION, bridge_probe_readout
from .shared.reporting import (CROSS_IMAGE_GROUPS, cross_image_metrics, cross_image_verdict,
                               print_cross_image_report)
from .shared.sampling import select_ranked_pairs
from .stage_a_steering import emotion_token_ids

# image = the visual-valence test; context/question/structure = did image valence broadcast into text?;
# text_all = all non-image alignable tokens; all = everything but the read-out query (≈100% sanity).
GROUPS = CROSS_IMAGE_GROUPS
CONTEXT_BANKS = CROSS_IMAGE_CONTEXT_BANKS


def _cross_groups(seg: dict) -> tuple[dict, dict]:
    return cross_image_groups(seg)


def run(config_path: str, limit_override: int | None = None, layers_override: str | None = None,
        context_polarity: str = "neutral", context_idx: int = 0) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    crit = int(cfg.get("critical_layer", 18))
    onset = int(cfg.get("patch_onset_layer", 13))
    if layers_override:
        a, b = (int(x) for x in layers_override.split("-"))
        patch_layers = list(range(a, b + 1))
    else:
        patch_layers = list(range(onset, crit))   # [onset .. L18) — same band as same-image, for comparability
    n_pairs = limit_override or int(cfg.get("patch_n_images", 60))
    ctx = CONTEXT_BANKS[context_polarity][context_idx]

    from ..paths import STAGE_A_DIR
    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    # donor = highest-valence (positive group), recipient = lowest-valence (negative group), paired by
    # rank so each pair has the largest possible image-valence gap.
    frame = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    pos, neg = select_ranked_pairs(frame, n_pairs)
    n_pairs = min(n_pairs, len(pos), len(neg))

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    name = f"blocks.{crit}.hook_attn_out"
    resid_names = {f"blocks.{L}.hook_resid_post" for L in patch_layers}

    rows, n_skip, seg_bad = [], 0, 0
    for k in tqdm(range(n_pairs), desc="stage-f cross-patching"):
        try:
            dimg = Image.open(pos.loc[k, "image_path"]).convert("RGB")
            rimg = Image.open(neg.loc[k, "image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        din = build_image_inputs(bridge, dimg, prompt=context_prompt(ctx))   # donor = POSITIVE image
        rin = build_image_inputs(bridge, rimg, prompt=context_prompt(ctx))   # recipient = NEGATIVE image
        if din["input_ids"].shape[-1] != rin["input_ids"].shape[-1]:
            seg_bad += 1                                                     # identical text ⇒ should never differ
            continue
        seg = segment_prompt_positions(
            bridge.tokenizer, rin["input_ids"], QUESTION, expected_image_tokens=256
        )                                                                   # identical for both runs
        groups, ok = cross_image_groups(seg)
        if not all(ok.get(g) for g in GROUPS):
            seg_bad += 1
            continue

        with torch.no_grad():
            _, dcache = bridge.run_with_cache(din["input_ids"], pixel_values=din["pixel_values"],
                                              names_filter=lambda nm: nm in resid_names)
        pos_probe, pos_val = bridge_probe_readout(
            bridge, din["input_ids"], din["pixel_values"], name, tok_ids, coef, inter
        )
        neg_probe, neg_val = bridge_probe_readout(
            bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids, coef, inter
        )

        row = {"donor_path": pos.loc[k, "image_path"], "recipient_path": neg.loc[k, "image_path"],
               "donor_valence": float(pos.loc[k, "valence"]), "recipient_valence": float(neg.loc[k, "valence"]),
               "pos_probe": pos_probe, "neg_probe": neg_probe, "pos_val": pos_val, "neg_val": neg_val}
        for g in GROUPS:
            idx = groups[g]
            hooks = []
            for L in patch_layers:
                donor_vals = dcache[f"blocks.{L}.hook_resid_post"][0, torch.as_tensor(idx)]
                hooks.append((f"blocks.{L}.hook_resid_post", bridge_patch_hook(idx, donor_vals)))
            p_probe, p_val = bridge_probe_readout(
                bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids,
                coef, inter, extra_hooks=hooks
            )
            row[f"patch_{g}_probe"] = p_probe
            row[f"patch_{g}_val"] = p_val
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "cross_patching.parquet")
    rec = cross_image_recovery(df, GROUPS)
    metrics = cross_image_metrics(
        rec, patch_layers, crit, ctx, context_polarity, len(rows), n_skip, seg_bad,
        run_stamp=run_stamp(), git_hash=git_hash(),
    )
    data_path = STAGE_F_DIR / "cross_patching.parquet"
    metrics_path = STAGE_F_DIR / "cross_patching_metrics.json"
    save_json(metrics, metrics_path)
    print_cross_image_report(rec, metrics, patch_layers, data_path, metrics_path)
    return metrics


def _recovery(df, n_boot: int = 2000, seed: int = 0) -> dict:
    return cross_image_recovery(df, GROUPS, n_boot=n_boot, seed=seed)


def _probe_valid(patch_layers, crit) -> bool:
    return probe_recovery_valid(patch_layers, crit)


def _verdict(rec, patch_layers, crit) -> str:
    return cross_image_verdict(rec, patch_layers, crit)


def _metrics(rec, patch_layers, crit, ctx, pol, n_ok, n_skip, seg_bad) -> dict:
    return cross_image_metrics(
        rec, patch_layers, crit, ctx, pol, n_ok, n_skip, seg_bad,
        run_stamp=run_stamp(), git_hash=git_hash(),
    )


def _print(rec, metrics, patch_layers) -> None:
    print_cross_image_report(
        rec, metrics, patch_layers,
        STAGE_F_DIR / "cross_patching.parquet",
        STAGE_F_DIR / "cross_patching_metrics.json",
    )


def reanalyze(config_path: str) -> dict:
    load_config(config_path)
    pq = STAGE_F_DIR / "cross_patching.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run the cross-image patching pass on the A100 first.")
    df = pd.read_parquet(pq)
    rec = cross_image_recovery(df, GROUPS)
    mpath = STAGE_F_DIR / "cross_patching_metrics.json"
    old = load_config(mpath) if mpath.exists() else {}
    metrics = cross_image_metrics(
        rec, old.get("patch_layers", []), old.get("critical_layer", 18),
        old.get("context", ""), old.get("context_polarity", "neutral"),
        int(len(df)), old.get("n_skipped", 0), old.get("n_segmentation_dropped", 0),
        run_stamp=run_stamp(), git_hash=git_hash(),
    )
    save_json(metrics, mpath)
    print_cross_image_report(
        rec, metrics, old.get("patch_layers", [0, 0]) or [0, 0],
        STAGE_F_DIR / "cross_patching.parquet", mpath,
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — cross-image activation patching (T1.2: where does visual valence live?)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="donor/recipient pair count (default 60)")
    ap.add_argument("--layers", type=str, default=None, help="patch band 'a-b' (default onset..L17)")
    ap.add_argument("--context", choices=list(CONTEXT_BANKS), default="neutral",
                    help="fixed context polarity held constant across donor/recipient (default neutral)")
    ap.add_argument("--context-idx", type=int, default=0, help="index within the chosen context bank")
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute recovery + bootstrap CIs from the saved parquet (CPU, no model)")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze(args.config)
    else:
        run(args.config, limit_override=args.limit, layers_override=args.layers,
            context_polarity=args.context, context_idx=args.context_idx)


if __name__ == "__main__":
    main()
