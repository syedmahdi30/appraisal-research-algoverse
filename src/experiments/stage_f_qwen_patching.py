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
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_f_qwen import (DEFAULT_MODEL, QUESTION, build_inputs, emotion_token_ids, load_qwen,
                           select_extreme_images, valence_score)

GROUPS = ("image", "question", "bos", "prefix_delim", "suffix_delim", "structure", "text_all")


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
    if not needle:
        return None
    for s in range(len(hay) - len(needle) + 1):
        if hay[s:s + len(needle)] == needle:
            return s
    return None


def segment_positions(tokenizer, input_ids) -> dict:
    """Partition key positions into image / context / question / template for one Qwen prompt.

    Image span = the maximal contiguous run of a single repeated id (the `<|image_pad|>` block;
    VARIABLE length on Qwen — do not assert a fixed count). Question = the identical question string.
    Context = between the image block and the question. Template = the rest (system/user/assistant
    turn scaffold + BOS-equivalent). Mirrors the Gemma segmenter minus the 256-token assertion.
    """
    ids = input_ids[0].tolist()
    n = len(ids)
    best = (0, 0)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if (j - i + 1) > best[1]:
            best = (i, j - i + 1)
        i = j + 1
    img_start, img_len = best
    img_end = img_start + img_len
    image_idx = list(range(img_start, img_end))

    q_start, q_len = None, 0
    for anchor in (" " + QUESTION, QUESTION):
        toks = tokenizer.encode(anchor, add_special_tokens=False)
        q_start = _find_subseq(ids, toks)
        if q_start is not None:
            q_len = len(toks)
            break
    if q_start is None or q_start <= img_end:
        q_start, q_len = max(img_end, n - 12), 0
    question_idx = list(range(q_start, min(q_start + q_len, n))) if q_len else []
    return {"image": np.array(image_idx), "question": np.array(question_idx),
            "question_ok": bool(q_len), "img_len": int(img_len), "n": n}


def _aligned_groups(seg_d, seg_r):
    """Map donor→recipient positions per patchable group (equal-length, position-aligned).

    image aligned (same photo → same image-pad count → same indices); question identical (shifted by
    the context-length difference); structure = prefix turn tokens + suffix turn scaffold, split into
    bos / prefix_delim / suffix_delim, EXCLUDING the read-out/query token; text_all = question ∪
    structure. Context tokens differ in length and are never patched.
    """
    out, ok = {}, {}
    di, ri = seg_d["image"], seg_r["image"]
    ok["image"] = bool(len(di) and len(di) == len(ri) and int(di[0]) == int(ri[0]))
    if ok["image"]:
        out["image"] = (di, ri)
    qd, qr = seg_d["question"], seg_r["question"]
    ok["question"] = bool(len(qd) and len(qd) == len(qr))
    if ok["question"]:
        out["question"] = (qd, qr)
    ok["structure"] = ok["bos"] = ok["prefix_delim"] = ok["suffix_delim"] = False
    if ok["image"] and ok["question"]:
        pre = np.arange(0, int(seg_d["image"][0]))
        qe_d, qe_r = int(qd[-1]) + 1, int(qr[-1]) + 1
        suf_d, suf_r = np.arange(qe_d, seg_d["n"] - 1), np.arange(qe_r, seg_r["n"] - 1)  # drop last tok
        if len(suf_d) == len(suf_r) and len(pre) >= 2 and len(suf_d) >= 1:
            out["structure"] = (np.concatenate([pre, suf_d]), np.concatenate([pre, suf_r]))
            out["bos"] = (pre[:1], pre[:1])
            out["prefix_delim"] = (pre[1:], pre[1:])
            out["suffix_delim"] = (suf_d, suf_r)
            ok["structure"] = ok["bos"] = ok["prefix_delim"] = ok["suffix_delim"] = True
    if ok["question"] and ok["structure"]:
        out["text_all"] = (np.concatenate([out["question"][0], out["structure"][0]]),
                           np.concatenate([out["question"][1], out["structure"][1]]))
        ok["text_all"] = True
    return out, ok


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
    return valence_score(last, tok_ids), lp


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
    tok_ids = emotion_token_ids(processor)

    sel = select_extreme_images(n_images * 2)
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
        seg_d = segment_positions(processor.tokenizer, din["input_ids"])
        seg_r = segment_positions(processor.tokenizer, rin["input_ids"])
        groups, ok = _aligned_groups(seg_d, seg_r)
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
    rec = _recovery(df)

    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name, "read_out": "behavioral_valence",
               "n_layers": n_layers, "patch_band": [band[0], band[-1]] if band else [],
               "n_images": n_ok, "n_skipped": n_skip, "n_segmentation_dropped": seg_bad,
               "donor_pos_context": pos_ctx, "recipient_neg_context": neg_ctx,
               "recovery": rec, "verdict": _verdict(rec)}
    save_json(metrics, STAGE_F_DIR / "patching_qwen_metrics.json")

    print(f"\nStage F [Qwen: {model_name}] patching — {n_ok} positive images ({n_skip} skipped, "
          f"{seg_bad} seg-dropped); patch decoder layers {band[0]}-{band[-1]} of {n_layers}.")
    print(f"  donor +ctx: \"{pos_ctx[:38]}\"   recipient -ctx: \"{neg_ctx[:38]}\"")
    print(f"  baselines (behavioral valence): pos {rec['pos_val']:+.3f}  neg {rec['neg_val']:+.3f}")
    print(f"  {'group':13s} {'recovery(valence)':>18s}")
    for g in GROUPS:
        print(f"  {g:13s} {rec[g]['val'] * 100:>17.0f}%")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  data -> {STAGE_F_DIR/'patching_qwen.parquet'}")
    return metrics


def _recovery(df) -> dict:
    out = {k: float(df[k].mean()) for k in ("pos_val", "neg_val")}
    d = out["pos_val"] - out["neg_val"]
    for g in GROUPS:
        out[g] = {"val": float((df[f"patch_{g}_val"].mean() - out["neg_val"]) / d) if d else float("nan")}
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
    args = ap.parse_args()
    run(args.config, args.model, limit_override=args.limit, layers_override=args.layers,
        pos_idx=args.pos_idx, neg_idx=args.neg_idx)


if __name__ == "__main__":
    main()
