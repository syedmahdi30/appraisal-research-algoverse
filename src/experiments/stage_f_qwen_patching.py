"""Stage F — Qwen activation patching: does the MECHANISM replicate on a second VLM?

Ports the Gemma carrier experiment (`stage_f_patching.py`) to Qwen-VL via raw HuggingFace forward
hooks — NO TransformerBridge, NO probe (recovery read on BEHAVIORAL VALENCE). Question: on Qwen too,
(a) are the IMAGE tokens causally inert, and (b) is the negative context carried by the ASSISTANT-TURN
scaffold (Qwen's `<|im_end|><|im_start|>assistant`, the analog of Gemma's `<end_of_turn>`
`<start_of_turn>model`)?

Method (identical to Gemma): donor = positive-context run, recipient = negative-context run (SAME
image). Overwrite the recipient's residual (decoder-layer output) over a layer band for one
position-aligned token group, and measure recovery toward the positive read-out:
    recovery(G) = (patched − neg) / (pos − neg).
Groups (aligned across runs, read-out/query token excluded): image, question, bos, prefix_delim,
suffix_delim (turn scaffold), structure (= their union), text_all (question ∪ structure). Context
tokens differ in length and are not patched (the 1 − text_all remainder is their share).

Differences from Gemma, handled here:
  * variable image-token count (dynamic resolution) — detect the image span each time, never hardcode.
  * raw forward hooks on the decoder layers (path resolved at boot).
  * behavioral valence saturates on Qwen → recovery is coarse per image but graded in the mean; a
    calibration-free ARGMAX recovery is reported alongside.
  * onset layer unknown on Qwen (no layerwise yet) → patch a broad mid+late band by default (--layers).

Run in the `requirements-qwen.txt` env on the A100. `--limit N`, `--layers a-b`, `--pos-idx/--neg-idx`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS
from ..data.labels import EMOTION_LABELS
from ..paths import PROCESSED_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .shared.patching import (SAME_IMAGE_GROUPS, aligned_patch_groups, find_subsequence,
                              segment_prompt_positions)
from .shared.readouts import closed_vocab_valence, first_content_token_ids
from .shared.sampling import select_extreme_rows
from .stage_f_qwen import DEFAULT_MODEL, QUESTION, build_inputs, load_qwen

GROUPS = SAME_IMAGE_GROUPS


# --------------------------------------------------------------------------- decoder layers
def decoder_layers(model):
    """Return the LM decoder-layer ModuleList, trying the paths that vary across HF versions."""
    cands = [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
    ]
    for f in cands:
        try:
            layers = f(model)
            if hasattr(layers, "__len__") and len(layers) > 1:
                return layers
        except AttributeError:
            continue
    raise RuntimeError("could not locate the decoder layers — inspect model.named_modules() and add "
                       "the path to decoder_layers().")


# --------------------------------------------------------------------------- segmentation (Qwen)
def _find_subseq(hay, needle):
    return find_subsequence(hay, needle)


def segment_positions(tokenizer, input_ids) -> dict:
    """Partition key positions into image / context / question / template for one Qwen prompt.

    Image span = the maximal contiguous run of a single repeated id (the `<|image_pad|>` block;
    VARIABLE length on Qwen — do not assert a fixed count). Question = the identical question string.
    Context = between the image block and the question. Template = the rest (system/user/assistant
    turn scaffold + BOS-equivalent). Mirrors the Gemma segmenter minus the 256-token assertion.
    """
    segment = segment_prompt_positions(
        tokenizer, input_ids, QUESTION, expected_image_tokens=None
    )
    return {
        "image": np.array(segment["image"].tolist()),
        "question": np.array(segment["question"].tolist()),
        "question_ok": segment["question_ok"], "img_len": segment["img_len"],
        "n": segment["n"],
    }


def _aligned_groups(seg_d, seg_r):
    return aligned_patch_groups(seg_d, seg_r)


# --------------------------------------------------------------------------- hooks
def _cache_hook(store, layer_i):
    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        store[layer_i] = h.detach().clone()
    return hook


def _patch_hook(recip_idx, donor_vals):
    idx = torch.as_tensor(recip_idx, dtype=torch.long)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, idx, :] = donor_vals.to(h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return hook


def _readout(model, inputs, tok_ids, hooks=None):
    """(behavioral_valence, {emotion: logprob}) at the last token, with optional forward hooks."""
    handles = list(hooks or [])
    inp = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inp)
    for h in handles:
        h.remove()
    last = out.logits[0, -1].float()
    lp = {w: float(v) for w, v in zip(EMOTION_LABELS,
          torch.log_softmax(last[[tok_ids[w] for w in EMOTION_LABELS]], dim=-1))}
    return closed_vocab_valence(last, tok_ids), lp


# --------------------------------------------------------------------------- run
def run(config_path: str, model_name: str, limit_override: int | None = None,
        layers_override: str | None = None, pos_idx: int | None = None,
        neg_idx: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    n_images = limit_override or int(cfg.get("patch_n_images", 60))
    ni = neg_idx if neg_idx is not None else int(cfg.get("patch_neg_idx", 2))
    pj = pos_idx if pos_idx is not None else int(cfg.get("patch_pos_idx", 0))
    neg_ctx, pos_ctx = NEGATIVE_CONTEXTS[ni], POSITIVE_CONTEXTS[pj]

    model, processor = load_qwen(model_name)
    layers = decoder_layers(model)
    n_layers = len(layers)
    if layers_override:
        a, b = (int(x) for x in layers_override.split("-"))
        band = list(range(a, b + 1))
    else:  # onset unknown on Qwen (no layerwise yet) → broad mid+late band, proportional to Gemma's
        band = list(range(round(0.35 * n_layers), n_layers - 2))
    tok_ids = first_content_token_ids(processor)

    frame = pd.read_parquet(PROCESSED_DIR / "emotic_test.parquet").reset_index(drop=True)
    sel = select_extreme_rows(frame, n_images * 2)
    sel = sel[sel["image_group"] == "positive"].head(n_images).reset_index(drop=True)

    rows, n_skip, seg_bad, n_ok = [], 0, 0, 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f qwen patching"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        din = build_inputs(processor, img, pos_ctx)
        rin = build_inputs(processor, img, neg_ctx)
        seg_d = segment_prompt_positions(
            processor.tokenizer, din["input_ids"], QUESTION, expected_image_tokens=None
        )
        seg_r = segment_prompt_positions(
            processor.tokenizer, rin["input_ids"], QUESTION, expected_image_tokens=None
        )
        groups, ok = aligned_patch_groups(seg_d, seg_r)
        if not all(ok.get(g) for g in GROUPS):
            seg_bad += 1
            continue

        # donor (positive) cache at the band layers + its read-out baseline
        store = {}
        chandles = [layers[L].register_forward_hook(_cache_hook(store, L)) for L in band]
        pos_val, _ = _readout(model, din, tok_ids, hooks=chandles)   # hooks removed inside _readout
        neg_val, _ = _readout(model, rin, tok_ids)

        row = {"image_path": r["image_path"], "pos_val": pos_val, "neg_val": neg_val}
        for g in GROUPS:
            di_g, ri_g = groups[g]
            phandles = [layers[L].register_forward_hook(
                _patch_hook(ri_g, store[L][0, torch.as_tensor(di_g)])) for L in band]
            p_val, _ = _readout(model, rin, tok_ids, hooks=phandles)
            row[f"patch_{g}_val"] = p_val
        rows.append(row)
        n_ok += 1

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "patching_qwen.parquet")
    meta = {"model": model_name, "n_layers": n_layers, "patch_band": [band[0], band[-1]] if band else [],
            "n_skipped": n_skip, "n_segmentation_dropped": seg_bad,
            "donor_pos_context": pos_ctx, "recipient_neg_context": neg_ctx}
    return _analyze_and_print(df, meta)


def _analyze_and_print(df, meta) -> dict:
    rec = _recovery(df)
    metrics = {"run": run_stamp(), "git": git_hash(), "read_out": "behavioral_valence",
               "n_images": int(len(df)), "recovery": rec, "verdict": _verdict(rec), **meta}
    save_json(metrics, STAGE_F_DIR / "patching_qwen_metrics.json")
    band = meta.get("patch_band", [])
    print(f"\nStage F [Qwen: {meta.get('model')}] patching — {len(df)} positive images "
          f"({meta.get('n_skipped', 0)} skipped, {meta.get('n_segmentation_dropped', 0)} seg-dropped); "
          f"patch decoder layers {band[0] if band else '?'}-{band[-1] if band else '?'} of "
          f"{meta.get('n_layers', '?')}.")
    print(f"  baselines (behavioral valence): pos {rec['pos_val']:+.3f}  neg {rec['neg_val']:+.3f}")
    print(f"  {'group':13s} {'recovery':>9s}   95% CI")
    for g in GROUPS:
        print(f"  {g:13s} {rec[g]['val'] * 100:>8.0f}%   [{rec[g]['ci95'][0] * 100:+.0f}%, "
              f"{rec[g]['ci95'][1] * 100:+.0f}%]")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  data -> {STAGE_F_DIR/'patching_qwen.parquet'}")
    return metrics


def reanalyze(config_path: str) -> dict:
    """Recompute recovery + CIs from the saved parquet — CPU only, no model load."""
    ensure_dirs()
    pq = STAGE_F_DIR / "patching_qwen.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run the Qwen patching pass first.")
    mpath = STAGE_F_DIR / "patching_qwen_metrics.json"
    meta = {k: load_config(mpath).get(k) for k in
            ("model", "n_layers", "patch_band", "n_skipped", "n_segmentation_dropped")} if mpath.exists() else {}
    return _analyze_and_print(pd.read_parquet(pq), meta)


def _recovery(df, n_boot: int = 2000, seed: int = 0) -> dict:
    """Recovery per group with a 95% CI bootstrapped over images (clustered)."""
    pos, neg = df["pos_val"].to_numpy(), df["neg_val"].to_numpy()
    out = {"pos_val": float(pos.mean()), "neg_val": float(neg.mean())}
    d = out["pos_val"] - out["neg_val"]
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(df), (n_boot, len(df))) if len(df) else None
    for g in GROUPS:
        pv = df[f"patch_{g}_val"].to_numpy()
        r = float((pv.mean() - out["neg_val"]) / d) if d else float("nan")
        ci = [float("nan"), float("nan")]
        if boots is not None and d:
            bs = [(pv[b].mean() - neg[b].mean()) / (pos[b].mean() - neg[b].mean()) for b in boots]
            ci = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
        out[g] = {"val": r, "ci95": ci}
    return out


def _verdict(rec) -> str:
    g = lambda k: rec[k]["val"]  # noqa: E731
    if max(abs(g(k)) for k in GROUPS) < 0.05:
        return ("NO RECOVERY — every group patch recovers <5%; the patch likely is not propagating "
                "(check the decoder_layers path and the forward-hook output-tuple format). Not "
                "interpretable — fix before trusting.")
    ri, rq, rs, rt = g("image"), g("question"), g("structure"), g("text_all")
    sinks = {"BOS": g("bos"), "prefix-delims": g("prefix_delim"), "suffix-delims": g("suffix_delim")}
    dom = max(sinks, key=lambda k: sinks[k])
    parts = (f"(valence recovery) image {ri:.0%}, question {rq:.0%} | BOS {sinks['BOS']:.0%}, "
             f"prefix {sinks['prefix-delims']:.0%}, suffix {sinks['suffix-delims']:.0%} → structure "
             f"{rs:.0%}, all-text {rt:.0%}")
    concl = ["IMAGE tokens causally INERT (matches Gemma)" if abs(ri) < 0.08 else f"IMAGE tokens carry {ri:.0%}"]
    if rs > 1.5 * max(rq, 1e-3):
        concl.append(f"carried by the turn scaffold; dominant = {dom} ({sinks[dom]:.0%}) — "
                     f"{'MATCHES Gemma (suffix/assistant-turn preamble)' if dom == 'suffix-delims' else 'DIFFERS from Gemma (was suffix)'}")
    else:
        concl.append(f"question and structure comparable ({rq:.0%} / {rs:.0%})")
    return parts + ". " + "; ".join(concl) + "."


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F Qwen — activation patching (mechanism replication)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="positive-group image count (default 60)")
    ap.add_argument("--layers", type=str, default=None, help="patch band 'a-b' (default mid+late)")
    ap.add_argument("--pos-idx", type=int, default=None)
    ap.add_argument("--neg-idx", type=int, default=None)
    ap.add_argument("--reanalyze", action="store_true", help="recompute recovery + CIs from the saved parquet (CPU)")
    args = ap.parse_args()
    if args.reanalyze:
        reanalyze(args.config)
    else:
        run(args.config, args.model, limit_override=args.limit, layers_override=args.layers,
            pos_idx=args.pos_idx, neg_idx=args.neg_idx)


if __name__ == "__main__":
    main()
