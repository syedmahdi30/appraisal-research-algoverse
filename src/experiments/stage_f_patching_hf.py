"""Stage F same-image patching on raw HuggingFace — re-score the carrier result without the bridge.

WHY. TransformerBridge computes a different multimodal forward from raw HF for Gemma-3 on
byte-identical inputs (docs/bridge-bug-2026-08-22.md): internal representations are sound
(resid_post cosine >= 0.978 through L32, probe site 0.980) but the OUTPUT is corrupted (6.15 nats,
argmax flips, no-context image separation AUC 0.788 vs 0.982). The organising principle from the
Stage C and Stage D re-runs is that DIFFERENTIAL measures survive and ABSOLUTE/categorical ones die.

Patching recovery is differential twice over --- a ratio of differences,

    recovery(G) = (patched(G) - neg_baseline) / (pos_baseline - neg_baseline)

scored on the L18 probe, at a site whose cosine to raw HF is 0.980. So the published 65%/57%
turn-boundary result is EXPECTED to survive. That is precisely why it has to be measured rather than
assumed: it is the mechanism claim the paper's §6 rests on, and "expected to survive" is not a
result. This module re-runs it end to end on raw HF.

WHAT IS DELIBERATELY SHARED WITH THE BRIDGE VERSION. Alignment, recovery, verdict, and prompt
segmentation come from `experiments.shared`; they are pure numpy/pandas/token operations and touch no
model. Both runners therefore aggregate identically without either backend importing the other.

THE TAP. The probe was fit on the bridge's `blocks.18.hook_attn_out`. The raw-HF equivalent is
`post_attention_layernorm`, NOT `self_attn`: Gemma post-norms the attention output before the
residual add and TransformerLens folds that into its attn-block output. The Stage C port measured
the difference --- `self_attn` scores r^2 -6.26 where `post_attention_layernorm` scores +0.634
against Stage A's 0.641. A wrong tap here would not error; it would silently report that no token
group carries anything. `--verify-tap` checks the site before spending a sweep on it.

    python -m src.experiments.stage_f_patching_hf --verify-tap    # do this first
    python -m src.experiments.stage_f_patching_hf                 # full sweep (pair 1)
    python -m src.experiments.stage_f_patching_hf --pos-idx 1 --neg-idx 0   # pair 2

Published reference (bridge): assistant-turn boundary 65% / 57% across the two pairs, image tokens
~0%, all-aligned-text 85% / 87%. NOTE the pair indices --- Pair 1 is (pos 0, neg 2) and Pair 2 is
(pos 4, neg 0), NOT (1, 0). Only Pair 2 is artifact-backed; Pair 1's run was overwritten.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS, context_prompt
from ..data.emotic import load_split as load_emotic_split
from ..paths import STAGE_A_DIR, STAGE_F_DIR, ensure_dirs
from .common import load_config, load_probes, metrics_provenance, save_json
from .shared.hf_runtime import (
    capture_residuals,
    encode_image_prompt,
    last_token_tap,
    load_gemma_hf,
    patch_residuals,
)
from .shared.patching import (
    SAME_IMAGE_GROUPS,
    aligned_patch_groups,
    same_image_recovery,
    same_image_resampling_metadata,
    segment_prompt_positions,
)
from .shared.readouts import QUESTION, closed_vocab_valence, first_content_token_ids
from .shared.reporting import same_image_verdict
from .shared.sampling import select_extreme_rows

GROUPS = SAME_IMAGE_GROUPS
CANDIDATE_TAPS = ("self_attn", "self_attn.o_proj", "post_attention_layernorm")

# Published bridge results, keyed by the (pos_idx, neg_idx) context pair they were actually run on.
# Keyed explicitly because the two published pairs are NOT (0,2) and (1,0): pair 2 is (4,0), the
# "wonderful news"/"devastating news" minimal pair. Any other pair has no published comparator and
# must print "--" rather than borrowing a number from a different context pair.
#
#   (0, 2)  championship / funeral      — paper Pair 1. DOC-ONLY: the run that produced it was
#                                         overwritten by the old fixed-path clobbering, so it
#                                         survives only in docs/stage-f-mechanism.md.
#   (4, 0)  wonderful / devastating     — paper Pair 2. Artifact-backed:
#                                         results/stage_f/patching_metrics.json.
PUBLISHED_BY_PAIR = {
    (0, 2): {"image": -0.01, "bos": 0.00, "prefix_delim": 0.00,
             "question": 0.22, "suffix_delim": 0.65, "text_all": 0.85},
    (4, 0): {"image": 0.007, "bos": -0.008, "prefix_delim": -0.011,
             "question": 0.320, "suffix_delim": 0.566, "text_all": 0.867},
}
PAIR_LABEL = {(0, 2): "paper Pair 1 (doc-only provenance)", (4, 0): "paper Pair 2 (artifact-backed)"}


class _TokShim:
    """`segment_positions` takes a bridge but only ever reads `.tokenizer`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


# --------------------------------------------------------------------------- raw-HF compatibility façades
resid_capture_full = capture_residuals
encode = encode_image_prompt


def patch_resid(model, donor: dict[int, torch.Tensor], donor_idx, recip_idx):
    return patch_residuals(model, donor, donor_idx, recip_idx)


def readout(model, enc, store, coef, inter, tok_ids) -> tuple[float, float]:
    """One forward -> (frozen-probe read-out at the tap, behavioural valence). `store` is the tap."""
    from ..probes.evaluate import predict

    with torch.no_grad():
        out = model(**enc)
    act = np.asarray(store[0], dtype=np.float32)
    probe = float(predict(act[None, :], coef, inter)[0])
    return probe, closed_vocab_valence(out.logits[0, -1].float(), tok_ids)


# --------------------------------------------------------------------------- data
def select_positive_images(split: str, n: int) -> pd.DataFrame:
    """The n highest-valence EMOTIC rows, tagged as the positive group."""
    df = load_emotic_split(split).reset_index(drop=True)
    selected = select_extreme_rows(df, n * 2)
    return selected[selected["image_group"] == "positive"].reset_index(drop=True)


# --------------------------------------------------------------------------- tap verification
def verify_tap(cfg, tap: str) -> dict:
    """Check the read-out site before spending a sweep on it.

    Two cheap checks on a handful of images, both of which a wrong tap fails loudly:
      1. the frozen probe separates positive- from negative-context runs in the right direction,
      2. segmentation finds a 256-token image block, the question, and all alignable groups.
    """
    crit = int(cfg.get("critical_layer", 18))
    probes = load_probes(STAGE_A_DIR / "probes.npz")
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    model, processor = load_gemma_hf(cfg.get("model", "google/gemma-3-4b-it"))
    tok_ids = first_content_token_ids(processor)
    sel = select_positive_images(cfg.get("split", "test"), 8)
    pos_ctx, neg_ctx = POSITIVE_CONTEXTS[0], NEGATIVE_CONTEXTS[2]

    gaps, seg_ok, n = [], 0, 0
    with last_token_tap(model, crit, tap) as store:
        for _, r in tqdm(list(sel.iterrows()), desc=f"verify-tap [{tap}]"):
            try:
                img = Image.open(r["image_path"]).convert("RGB")
            except (FileNotFoundError, OSError):
                continue
            enc_p = encode_image_prompt(processor, img, context_prompt(pos_ctx), model.device)
            enc_n = encode_image_prompt(processor, img, context_prompt(neg_ctx), model.device)
            p_probe, _ = readout(model, enc_p, store, coef, inter, tok_ids)
            n_probe, _ = readout(model, enc_n, store, coef, inter, tok_ids)
            gaps.append(p_probe - n_probe)
            shim = _TokShim(processor.tokenizer)
            sd = segment_prompt_positions(
                shim.tokenizer, enc_p["input_ids"].cpu(), QUESTION, expected_image_tokens=256
            )
            sr = segment_prompt_positions(
                shim.tokenizer, enc_n["input_ids"].cpu(), QUESTION, expected_image_tokens=256
            )
            _, ok = aligned_patch_groups(sd, sr)
            seg_ok += int(all(ok.get(g) for g in GROUPS) and sd["image_ok"] and sd["question_ok"])
            n += 1

    mean_gap = float(np.mean(gaps)) if gaps else float("nan")
    out = {"tap": tap, "layer": crit, "n": n, "mean_probe_gap_pos_minus_neg": mean_gap,
           "n_segmentation_ok": seg_ok,
           "verdict": ("OK" if mean_gap > 0.05 and seg_ok == n else "SUSPECT")}
    print(f"\n  tap {tap} @ L{crit}: mean probe gap (pos-ctx - neg-ctx) = {mean_gap:+.3f} "
          f"over {n} images; segmentation ok {seg_ok}/{n}  -> {out['verdict']}")
    if out["verdict"] != "OK":
        print("  A near-zero or negative gap means the probe is not reading the site it was fit on.\n"
              f"  Try the other candidates: {', '.join(CANDIDATE_TAPS)} (see the Stage C port).")
    return out


# --------------------------------------------------------------------------- main sweep
def run(config_path: str, limit_override: int | None = None, layers_override: str | None = None,
        pos_idx: int | None = None, neg_idx: int | None = None,
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
    n_images = limit_override or int(cfg.get("patch_n_images", 60))
    ni = neg_idx if neg_idx is not None else int(cfg.get("patch_neg_idx", 2))
    pj = pos_idx if pos_idx is not None else int(cfg.get("patch_pos_idx", 0))
    neg_ctx, pos_ctx = NEGATIVE_CONTEXTS[ni], POSITIVE_CONTEXTS[pj]

    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    sel = select_positive_images(cfg.get("split", "test"), n_images)
    model, processor = load_gemma_hf(cfg.get("model", "google/gemma-3-4b-it"))
    tok_ids = first_content_token_ids(processor)
    shim = _TokShim(processor.tokenizer)

    rows, n_skip, seg_bad = [], 0, 0
    with last_token_tap(model, crit, tap) as store:
        for _, r in tqdm(list(sel.iterrows()), desc="stage-f patching (raw HF)"):
            try:
                img = Image.open(r["image_path"]).convert("RGB")
            except (FileNotFoundError, OSError):
                n_skip += 1
                continue
            enc_d = encode_image_prompt(
                processor, img, context_prompt(pos_ctx), model.device
            )   # donor: +ctx
            enc_r = encode_image_prompt(
                processor, img, context_prompt(neg_ctx), model.device
            )   # recipient: -ctx
            seg_d = segment_prompt_positions(
                shim.tokenizer, enc_d["input_ids"].cpu(), QUESTION, expected_image_tokens=256
            )
            seg_r = segment_prompt_positions(
                shim.tokenizer, enc_r["input_ids"].cpu(), QUESTION, expected_image_tokens=256
            )
            groups, ok = aligned_patch_groups(seg_d, seg_r)
            if not all(ok.get(g) for g in GROUPS):
                seg_bad += 1
                continue

            with capture_residuals(model, patch_layers) as donor:
                pos_probe, pos_val = readout(model, enc_d, store, coef, inter, tok_ids)
                donor = dict(donor)                      # detach from the context manager's store
            neg_probe, neg_val = readout(model, enc_r, store, coef, inter, tok_ids)

            row = {"image_path": r["image_path"], "pos_probe": pos_probe, "neg_probe": neg_probe,
                   "pos_val": pos_val, "neg_val": neg_val}
            for g in GROUPS:
                di, ri = groups[g]
                with patch_residuals(model, donor, di, ri):
                    p_probe, p_val = readout(model, enc_r, store, coef, inter, tok_ids)
                row[f"patch_{g}_probe"] = p_probe
                row[f"patch_{g}_val"] = p_val
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "patching_hf.parquet")
    meta = {
        "stack": "raw_hf", "tap": tap,
        "critical_layer": crit, "patch_layers": patch_layers,
        "n_skipped": n_skip, "n_segmentation_dropped": seg_bad,
        "donor_pos_context": pos_ctx, "recipient_neg_context": neg_ctx,
        "pos_idx": pj, "neg_idx": ni,
        "note": ("raw-HF re-score of the bridge result. recovery = (patched-neg)/(pos-neg) at the "
                 f"L{crit} {tap} read-out; decoder-layer outputs (= resid_post) patched over "
                 f"{patch_layers}. Estimates and paired confidence intervals weight each unique "
                 "image_path once."),
    }
    return _analyze_and_print(df, meta)


def _analyze_and_print(
    df: pd.DataFrame,
    meta: dict,
    *,
    n_boot: int = 2000,
    source_provenance: dict | None = None,
) -> dict:
    """Aggregate a completed raw-HF sweep without loading the model."""
    rec = same_image_recovery(df, GROUPS, n_boot=n_boot)
    metrics = {
        **metrics_provenance(source_provenance),
        **meta,
        **same_image_resampling_metadata(df),
        "bootstrap_samples": n_boot,
        "bootstrap_seed": 0,
        "recovery": rec,
        "verdict": same_image_verdict(rec),
    }
    save_json(metrics, STAGE_F_DIR / "patching_hf_metrics.json")

    pj, ni = int(meta.get("pos_idx", 0)), int(meta.get("neg_idx", 2))
    pos_ctx = meta.get("donor_pos_context", "")
    neg_ctx = meta.get("recipient_neg_context", "")
    patch_layers = meta.get("patch_layers", [])
    n_skip = int(meta.get("n_skipped", 0))
    seg_bad = int(meta.get("n_segmentation_dropped", 0))
    pub = PUBLISHED_BY_PAIR.get((pj, ni))
    pair = PAIR_LABEL.get((pj, ni), f"pos{pj}/neg{ni} — NO published comparator")
    print(f"\nStage F patching (RAW HF) — {metrics['n_images']} unique positive images from "
          f"{metrics['n_rows']} rows "
          f"({n_skip} skipped, {seg_bad} seg-dropped); patch resid_post {patch_layers[0]}-{patch_layers[-1]}.")
    print(f"  donor +ctx: \"{pos_ctx[:40]}\"   recipient -ctx: \"{neg_ctx[:40]}\"   [{pair}]")
    print(f"  baselines: probe pos {rec['pos_probe']:+.3f} / neg {rec['neg_probe']:+.3f}  |  "
          f"valence pos {rec['pos_val']:+.3f} / neg {rec['neg_val']:+.3f}")
    if pub is None:
        print(f"\n  NOTE: ({pj}, {ni}) is not one of the two published context pairs "
              f"((0,2) and (4,0)), so there is nothing to compare against — the published column "
              f"is blank by design rather than borrowed from a different pair.")
    print(f"\n  {'group':14s} {'raw HF (probe)':>15s} {'published (bridge)':>20s} {'valence':>10s}")
    for g in GROUPS:
        pub_s = f"{pub[g]*100:.0f}%" if pub and g in pub else "--"
        print(f"  {g:14s} {rec[g]['probe']*100:>14.0f}% {pub_s:>20s} {rec[g]['val']*100:>9.0f}%")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  data -> {STAGE_F_DIR/'patching_hf.parquet'}   "
          f"metrics -> {STAGE_F_DIR/'patching_hf_metrics.json'}")
    return metrics


def reanalyze(n_boot: int = 2000) -> dict:
    """Recompute unique-image recovery and confidence intervals from parquet on CPU."""
    ensure_dirs()
    data_path = STAGE_F_DIR / "patching_hf.parquet"
    metrics_path = STAGE_F_DIR / "patching_hf_metrics.json"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} missing — run the raw-HF patching pass first.")
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} missing — patch metadata is required for reanalysis."
        )
    previous = load_config(metrics_path)
    meta = {
        key: previous.get(key)
        for key in (
            "stack", "tap", "critical_layer", "patch_layers", "n_skipped",
            "n_segmentation_dropped", "donor_pos_context", "recipient_neg_context",
            "pos_idx", "neg_idx", "note",
        )
    }
    if not meta.get("patch_layers"):
        raise ValueError(f"{metrics_path} does not record a patch layer band")
    return _analyze_and_print(
        pd.read_parquet(data_path), meta, n_boot=n_boot, source_provenance=previous
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage F same-image patching on raw HF (bridge-free re-score)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="positive image count (default 60)")
    ap.add_argument("--layers", type=str, default=None, help="patch band 'a-b' (default onset..L17)")
    ap.add_argument("--pos-idx", type=int, default=None, help="POSITIVE_CONTEXTS index (donor)")
    ap.add_argument("--neg-idx", type=int, default=None, help="NEGATIVE_CONTEXTS index (recipient)")
    ap.add_argument("--tap", default="post_attention_layernorm", choices=CANDIDATE_TAPS,
                    help="raw-HF equivalent of the bridge's blocks.L.hook_attn_out")
    ap.add_argument("--verify-tap", action="store_true",
                    help="check the read-out site and segmentation, then exit")
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute unique-image recovery + CIs from parquet on CPU")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze()
        return
    if args.verify_tap:
        verify_tap(load_config(args.config), args.tap)
        return
    run(args.config, limit_override=args.limit, layers_override=args.layers,
        pos_idx=args.pos_idx, neg_idx=args.neg_idx, tap=args.tap)


if __name__ == "__main__":
    main()
