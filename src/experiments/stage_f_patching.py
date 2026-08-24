"""Stage F — activation patching: WHICH token group carries the context effect into the read-out?

Layerwise localization put the onset at layer ~13; the attribution showed the last token reads the
context INDIRECTLY, via the high-attention image and question positions. This names the carrier
causally: on each positive image, take the POSITIVE-context run as donor and the NEGATIVE-context run
as recipient, then overwrite the recipient's resid_post across the band [onset .. L18) for ONE token
group at a time — image tokens, question tokens, or both — with the donor's values, and measure how
far the L18 pleasantness read-out (and behavioral valence) recovers toward the positive-context value.

    recovery(G) = (patched(G) − neg_baseline) / (pos_baseline − neg_baseline)

Image and question tokens are position-aligned across the two runs (image = same 256-token block
before the context; the question string is identical), so their resid can be swapped 1:1; the context
tokens differ in length and are NOT patched (they are the source being attributed away). The band is
patched at every layer so the group's pathway is pinned to the donor through the read-out (a single
layer would be re-mixed by the still-negative context downstream — a lower bound). Whichever group
recovers most is the carrier.

Run on the A100 with HF_TOKEN + EMOTIC. Frozen probe; never re-fit. `--limit N`, `--layers a-b`.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS, context_prompt
from ..data.emotic import load_split as load_emotic_split
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .shared.patching import (SAME_IMAGE_GROUPS, aligned_patch_groups, bridge_patch_hook,
                              same_image_recovery, segment_prompt_positions)
from .shared.readouts import QUESTION, bridge_probe_readout
from .shared.reporting import same_image_verdict
from .shared.sampling import select_extreme_rows
from .stage_a_steering import emotion_token_ids

GROUPS = SAME_IMAGE_GROUPS


def _patch_hook(recip_idx, donor_vals):
    return bridge_patch_hook(recip_idx, donor_vals)


def _readout(bridge, ids, pv, name, tok_ids, coef, inter, extra_hooks=None):
    return bridge_probe_readout(
        bridge, ids, pv, name, tok_ids, coef, inter, extra_hooks=extra_hooks
    )


def _aligned_groups(seg_d, seg_r):
    return aligned_patch_groups(seg_d, seg_r)


def run(config_path: str, limit_override: int | None = None,
        layers_override: str | None = None, pos_idx: int | None = None,
        neg_idx: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    crit = int(cfg.get("critical_layer", 18))
    onset = int(cfg.get("patch_onset_layer", 13))
    if layers_override:
        a, b = (int(x) for x in layers_override.split("-"))
        patch_layers = list(range(a, b + 1))
    else:
        patch_layers = list(range(onset, crit))   # [onset .. L18)
    n_images = limit_override or int(cfg.get("patch_n_images", 60))
    ni = neg_idx if neg_idx is not None else int(cfg.get("patch_neg_idx", 2))
    pj = pos_idx if pos_idx is not None else int(cfg.get("patch_pos_idx", 0))
    neg_ctx = NEGATIVE_CONTEXTS[ni]   # default "funeral of a close friend"
    pos_ctx = POSITIVE_CONTEXTS[pj]   # default "won the championship"

    from ..paths import STAGE_A_DIR
    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    frame = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    sel = select_extreme_rows(frame, n_images * 2)
    sel = sel[sel["image_group"] == "positive"].head(n_images).reset_index(drop=True)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    name = f"blocks.{crit}.hook_attn_out"
    resid_names = {f"blocks.{L}.hook_resid_post" for L in patch_layers}

    rows, n_skip, n_ok, seg_bad = [], 0, 0, 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f patching"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        din = build_image_inputs(bridge, img, prompt=context_prompt(pos_ctx))
        rin = build_image_inputs(bridge, img, prompt=context_prompt(neg_ctx))
        seg_d = segment_prompt_positions(
            bridge.tokenizer, din["input_ids"], QUESTION, expected_image_tokens=256
        )
        seg_r = segment_prompt_positions(
            bridge.tokenizer, rin["input_ids"], QUESTION, expected_image_tokens=256
        )
        groups, ok = aligned_patch_groups(seg_d, seg_r)
        if not all(ok.get(g) for g in GROUPS):
            seg_bad += 1
            continue

        # donor (positive) cache at the patch layers + its read-out baseline
        dstore = {}
        with torch.no_grad():
            _, dcache = bridge.run_with_cache(din["input_ids"], pixel_values=din["pixel_values"],
                                              names_filter=lambda n: n in resid_names)
        pos_probe, pos_val = bridge_probe_readout(
            bridge, din["input_ids"], din["pixel_values"], name, tok_ids, coef, inter
        )
        neg_probe, neg_val = bridge_probe_readout(
            bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids, coef, inter
        )

        row = {"image_path": r["image_path"], "pos_probe": pos_probe, "neg_probe": neg_probe,
               "pos_val": pos_val, "neg_val": neg_val}
        for g in GROUPS:
            di, ri = groups[g]
            hooks = []
            for L in patch_layers:
                donor_vals = dcache[f"blocks.{L}.hook_resid_post"][0, torch.as_tensor(di)]
                hooks.append((f"blocks.{L}.hook_resid_post", bridge_patch_hook(ri, donor_vals)))
            p_probe, p_val = bridge_probe_readout(
                bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids,
                coef, inter, extra_hooks=hooks
            )
            row[f"patch_{g}_probe"] = p_probe
            row[f"patch_{g}_val"] = p_val
        rows.append(row)
        n_ok += 1

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "patching.parquet")
    rec = same_image_recovery(df, GROUPS)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "critical_layer": crit, "patch_layers": patch_layers,
        "n_images": n_ok, "n_skipped": n_skip, "n_segmentation_dropped": seg_bad,
        "donor_pos_context": pos_ctx, "recipient_neg_context": neg_ctx,
        "pos_idx": pj, "neg_idx": ni,
        "recovery": rec, "verdict": same_image_verdict(rec),
        "note": ("recovery = (patched-neg)/(pos-neg) at L{c} read-out; resid_post patched over band "
                 "{b} so the group's pathway is pinned to the donor through the read-out (lower bound: "
                 "context tokens stay negative and re-mix downstream).").format(c=crit, b=patch_layers),
    }
    save_json(metrics, STAGE_F_DIR / "patching_metrics.json")

    print(f"\nStage F patching — {n_ok} positive images ({n_skip} skipped, {seg_bad} seg-dropped); "
          f"patch resid_post layers {patch_layers[0]}-{patch_layers[-1]}.")
    print(f"  donor +ctx: \"{pos_ctx[:40]}\"   recipient -ctx: \"{neg_ctx[:40]}\"")
    print(f"  baselines: probe pos {rec['pos_probe']:+.3f} / neg {rec['neg_probe']:+.3f}  |  "
          f"valence pos {rec['pos_val']:+.3f} / neg {rec['neg_val']:+.3f}")
    print(f"\n  {'group':13s} {'recovery(probe)':>16s} {'recovery(valence)':>18s}")
    for g in GROUPS:
        print(f"  {g:13s} {rec[g]['probe']*100:>15.0f}% {rec[g]['val']*100:>17.0f}%")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  data -> {STAGE_F_DIR/'patching.parquet'}   metrics -> {STAGE_F_DIR/'patching_metrics.json'}")
    return metrics


def _recovery(df) -> dict:
    return same_image_recovery(df, GROUPS)


def _verdict(rec) -> str:
    return same_image_verdict(rec)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — activation patching (carrier identification)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="positive-group image count (default 60)")
    ap.add_argument("--layers", type=str, default=None, help="patch band 'a-b' (default onset..L17)")
    ap.add_argument("--pos-idx", type=int, default=None, help="POSITIVE_CONTEXTS index (donor)")
    ap.add_argument("--neg-idx", type=int, default=None, help="NEGATIVE_CONTEXTS index (recipient)")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit, layers_override=args.layers,
        pos_idx=args.pos_idx, neg_idx=args.neg_idx)


if __name__ == "__main__":
    main()
