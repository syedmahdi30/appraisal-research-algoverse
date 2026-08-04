"""Stage F — text-only context baseline (the negativity-asymmetry confound control).

The Stage F base pass found that a NEGATIVE context sentence moves the shared L18 valence read-out
much more than a POSITIVE one (on positive images, where head-room is symmetric: -0.598 vs +0.167),
and this survives the ceiling/floor (head-room) normalization. That leaves ONE confound: the
negative context bank (funeral, accident, "lost their job") may simply be more affectively extreme
than the positive bank (birthday, celebration) — a property of the STIMULI, not a model-level
cross-modal negativity amplification.

This script isolates that by running each context sentence with NO IMAGE through the SAME frozen
Stage A pleasantness probe (blocks.18.hook_attn_out) + behavioral valence. The prompt is the base
pass's `context_prompt` with the `<start_of_image>` token removed — token-for-token identical except
the image splice is gone — so the ONLY difference from the image condition is the image's presence.

Decision:
  * text-only |neg|/|pos| ratio ≈ the image-conditioned ratio  -> the asymmetry is a STIMULUS
    property (negative sentences are just stronger); fix by intensity-matching the banks and re-run.
  * image-conditioned ratio > text-only ratio                  -> the image AMPLIFIES negativity
    beyond the text stimulus = a genuine cross-modal effect worth the L18 attribution follow-up.

Cheap: one forward per context (6 neg + 6 pos + 2 neutral + 1 none = 15 forwards, no images). Run on
the A100 with HF_TOKEN. No EMOTIC needed. Never re-fits the probe.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from ..bridge.boot import boot_gemma
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE, context_prompt)
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import STAGE_F_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_a_steering import emotion_logprobs, emotion_token_ids, valence_score


def text_only_prompt(ctx: str | None) -> str:
    """The base-pass context prompt with the image slot removed (image-ablated, else identical)."""
    return context_prompt(ctx).replace("<start_of_image>", "")


def _stash_hook(store: dict):
    def hook(act, hook):  # noqa: ARG001 - TL contract
        store["act"] = act.detach()
        return act
    return hook


def _text_readout(bridge, ids, name, coef, inter, tok_ids):
    """(probe_readout, behavioral_valence, {emotion: logprob}) for a text-only forward (no image)."""
    store: dict = {}
    with torch.no_grad():
        logits = bridge.run_with_hooks(ids, fwd_hooks=[(name, _stash_hook(store))])
    last = ids.shape[-1] - 1
    act = store["act"][0, last].float().cpu().numpy()
    probe = float(predict(act[None, :], coef, inter)[0])
    return probe, valence_score(logits[0, -1], tok_ids), emotion_logprobs(logits[0, -1], tok_ids)


def _asymmetry(df, value: str) -> dict:
    """Per-polarity effect vs the neutral-context baseline for `value`, and the |neg|/|pos| ratio.

    Unit of analysis = the SENTENCE (6 per polarity); this asks whether the negative sentences carry
    more affect than the positive ones on their own, with no image in play.
    """
    neu = float(df[df["condition"] == "neutral"][value].mean())
    pos = (df[df["condition"] == "positive"][value] - neu)
    neg = (df[df["condition"] == "negative"][value] - neu)
    pe, ne = float(pos.mean()), float(neg.mean())
    ratio = abs(ne) / abs(pe) if pe != 0 else float("inf")
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(neg.abs().to_numpy(), pos.abs().to_numpy(), alternative="greater")
    return {"neutral_baseline": neu, "pos_effect": pe, "neg_effect": ne,
            "abs_pos": abs(pe), "abs_neg": abs(ne), "neg_over_pos_ratio": ratio,
            "asymmetry_index": abs(ne) - abs(pe),
            "mannwhitney_p_greater": float(mw.pvalue),
            "per_pos": [float(x) for x in pos], "per_neg": [float(x) for x in neg]}


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("critical_layer", 18))
    tap = cfg.get("tap", "hook_attn_out")

    from ..paths import STAGE_A_DIR
    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    conditions = [("none", "none", None)]
    conditions += [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)]
    conditions += [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
    conditions += [("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    multi = {w: r for w, r in verify_label_tokenization(bridge.tokenizer).items() if not r["single_token"]}
    name = f"blocks.{layer}.{tap}"

    rows = []
    for cond, cid, sentence in conditions:
        ids = bridge.to_tokens(text_only_prompt(sentence))
        probe, val, lp = _text_readout(bridge, ids, name, coef, inter, tok_ids)
        rows.append({"condition": cond, "context_id": cid, "context": sentence or "",
                     "text_code": TEXT_CODE[cond], "probe_readout": probe, "valence": val,
                     **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "text_only.parquet")

    asym = {"valence": _asymmetry(df, "valence"), "probe_readout": _asymmetry(df, "probe_readout")}

    # decisive comparison vs the image-conditioned run, if analyze_stage_f has been run.
    compare = _compare_to_image(asym)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "layer": layer, "tap": tap,
        "n_forwards": int(len(rows)), "asymmetry": asym, "image_comparison": compare,
        "tokenization_multi_token": multi,
        "note": ("image-ablated context baseline: same prompt as the base pass minus <start_of_image>. "
                 "Isolates whether the negativity asymmetry is a stimulus property of the context "
                 "bank (matched here) or cross-modal amplification (needs the image)."),
    }
    save_json(metrics, STAGE_F_DIR / "text_only_metrics.json")

    av = asym["valence"]
    print(f"\nStage F text-only baseline — {len(rows)} forwards (no images), L{layer} {tap}.\n")
    print("  per-context (no image) behavioral valence & probe:")
    print(f"    {'ctx':6s} {'valence':>8s} {'probe':>8s}  context")
    for _, r in df.iterrows():
        print(f"    {r['context_id']:6s} {r['valence']:>+8.3f} {r['probe_readout']:>+8.3f}  "
              f"\"{r['context'][:44]}\"")
    print(f"\n  TEXT-ONLY asymmetry (vs neutral text baseline {av['neutral_baseline']:+.3f}):")
    print(f"    pos-context effect {av['pos_effect']:+.3f}   neg-context effect {av['neg_effect']:+.3f}"
          f"   |neg|/|pos| = {av['neg_over_pos_ratio']:.2f}   MW p={av['mannwhitney_p_greater']:.3f}")
    if compare:
        print(f"\n  DECISIVE COMPARISON (context asymmetry, image present vs ablated):")
        print(f"    image-conditioned |neg|/|pos| (positive-image group) = {compare['image_ratio']:.2f}"
              f"  (neg {compare['image_neg']:+.3f} / pos {compare['image_pos']:+.3f})")
        print(f"    text-only        |neg|/|pos|                        = {compare['text_ratio']:.2f}"
              f"  (neg {av['neg_effect']:+.3f} / pos {av['pos_effect']:+.3f})")
        print(f"    → {compare['verdict']}")
    else:
        print("\n  (run analyze_stage_f first to auto-compare against the image-conditioned ratio)")
    if multi:
        print(f"  WARNING multi-token labels (first sub-token scored): {list(multi)}")
    print(f"\n  data -> {STAGE_F_DIR/'text_only.parquet'}   metrics -> {STAGE_F_DIR/'text_only_metrics.json'}")
    return metrics


def _compare_to_image(asym) -> dict | None:
    """Compare text-only context asymmetry to the image-conditioned one (positive-image group).

    The positive-image group is the clean image-side reference: its neutral valence is mid-scale, so
    head-room is roughly symmetric and the neg-vs-pos context ratio is not a floor artifact. If the
    text-only ratio matches it, the asymmetry lives in the stimuli; if the image ratio is larger, the
    image amplifies negativity.
    """
    ap = STAGE_F_DIR / "conflict_analysis.json"
    if not ap.exists():
        return None
    a = load_config(ap).get("asymmetry_vs_floor", {})
    img_neg = a.get("drop_pos_img_neg_ctx")            # neg context on positive images
    img_pos = a.get("congruent_pos_img_pos_ctx")       # pos context on positive images
    if img_neg is None or img_pos is None or img_pos == 0:
        return None
    image_ratio = abs(img_neg) / abs(img_pos)
    text_ratio = asym["valence"]["neg_over_pos_ratio"]
    if not np.isfinite(text_ratio):
        return None
    # amplification if the image ratio meaningfully exceeds text-only (>25% larger); stimulus if within.
    if image_ratio > 1.25 * text_ratio:
        verdict = (f"CROSS-MODAL AMPLIFICATION — the image inflates the neg/pos context ratio from "
                   f"{text_ratio:.2f} (text alone) to {image_ratio:.2f} (with image). The negativity "
                   f"asymmetry is NOT just stronger negative sentences; the image amplifies it. Real "
                   f"cross-modal effect — proceed to the L18 image-vs-context-token attribution.")
    elif text_ratio > 1.25 * image_ratio:
        verdict = (f"REVERSED — negative sentences are even MORE dominant text-only ({text_ratio:.2f}) "
                   f"than with the image ({image_ratio:.2f}); the image if anything DAMPENS the "
                   f"asymmetry. The effect is a stimulus property; intensity-match the banks.")
    else:
        verdict = (f"STIMULUS CONFOUND — text-only ({text_ratio:.2f}) and image-conditioned "
                   f"({image_ratio:.2f}) neg/pos ratios match: the asymmetry is a property of the "
                   f"context bank (negative sentences are stronger), not cross-modal amplification. "
                   f"Intensity-match the positive/negative banks (normed |valence|) and re-run before "
                   f"claiming a model-level negativity effect.")
    return {"image_ratio": float(image_ratio), "text_ratio": float(text_ratio),
            "image_neg": float(img_neg), "image_pos": float(img_pos), "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — text-only context baseline (asymmetry confound)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
