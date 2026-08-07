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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS, context_prompt
from ..paths import STAGE_F_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_a_steering import emotion_token_ids, valence_score
from .stage_f_attribution import _stash_hook, segment_positions
from .stage_f_conflict import select_extreme_images

GROUPS = ("image", "question", "bos", "prefix_delim", "suffix_delim", "structure", "text_all")


def _patch_hook(recip_idx, donor_vals):
    """Overwrite resid_post at `recip_idx` positions with donor_vals ([len(idx), d])."""
    idx = torch.as_tensor(recip_idx, dtype=torch.long)

    def hook(act, hook):  # noqa: ARG001 - resid_post [batch, seq, d]
        act[0, idx, :] = donor_vals.to(act.dtype)
        return act
    return hook


def _readout(bridge, ids, pv, name, tok_ids, coef, inter, extra_hooks=None):
    store = {}
    hooks = [(name, _stash_hook(store))] + list(extra_hooks or [])
    with torch.no_grad():
        logits = bridge.run_with_hooks(ids, pixel_values=pv, fwd_hooks=hooks)
    act = store["act"][0, ids.shape[-1] - 1].float().cpu().numpy()
    return float(predict(act[None, :], coef, inter)[0]), valence_score(logits[0, -1], tok_ids)


def _aligned_groups(seg_d, seg_r):
    """Map donor→recipient positions for each patchable token group (equal-length, position-aligned).

    Alignable groups (identical tokens across the two runs, so their resid can be swapped 1:1):
      image     — the 256-token block before the context (same indices in both runs)
      question  — the identical question string (shifted by the context-length difference)
      structure — prefix turn/BOS tokens (before the image, same indices) + suffix turn tokens (after
                  the question), EXCLUDING the last token (the read-out query position — patching it
                  would import the donor's read-out state instead of attributing to source tokens)
      text_all  — question ∪ structure (every alignable non-image text position); 1 − recovery(text_all)
                  is the share left in the UNPATCHABLE context tokens (the differing sentence).
    The context tokens themselves differ in length and are never patched.
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
    # structure: prefix (aligned indices) + suffix (aligned, last token dropped), and its split into
    # bos / prefix-delimiters / suffix-delimiters — the "which sink token" decomposition.
    ok["structure"] = ok["bos"] = ok["prefix_delim"] = ok["suffix_delim"] = False
    if ok["image"] and ok["question"]:
        pre = np.arange(0, int(seg_d["image"][0]))                       # same indices both runs
        qe_d, qe_r = int(qd[-1]) + 1, int(qr[-1]) + 1
        suf_d, suf_r = np.arange(qe_d, seg_d["n"] - 1), np.arange(qe_r, seg_r["n"] - 1)  # drop last tok
        if len(suf_d) == len(suf_r) and len(pre) >= 2 and len(suf_d) >= 1:
            out["structure"] = (np.concatenate([pre, suf_d]), np.concatenate([pre, suf_r]))
            out["bos"] = (pre[:1], pre[:1])                   # first token (<bos>) — canonical sink
            out["prefix_delim"] = (pre[1:], pre[1:])          # <start_of_turn>user\n(<start_of_image>)
            out["suffix_delim"] = (suf_d, suf_r)              # <end_of_turn>\n<start_of_turn>model (no last)
            ok["structure"] = ok["bos"] = ok["prefix_delim"] = ok["suffix_delim"] = True
    if ok["question"] and ok["structure"]:
        out["text_all"] = (np.concatenate([out["question"][0], out["structure"][0]]),
                           np.concatenate([out["question"][1], out["structure"][1]]))
        ok["text_all"] = True
    return out, ok


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

    sel = select_extreme_images(cfg.get("split", "test"), n_images * 2)
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
        seg_d, seg_r = segment_positions(bridge, din["input_ids"]), segment_positions(bridge, rin["input_ids"])
        groups, ok = _aligned_groups(seg_d, seg_r)
        if not all(ok.get(g) for g in GROUPS):
            seg_bad += 1
            continue

        # donor (positive) cache at the patch layers + its read-out baseline
        dstore = {}
        with torch.no_grad():
            _, dcache = bridge.run_with_cache(din["input_ids"], pixel_values=din["pixel_values"],
                                              names_filter=lambda n: n in resid_names)
        pos_probe, pos_val = _readout(bridge, din["input_ids"], din["pixel_values"], name, tok_ids,
                                      coef, inter)
        neg_probe, neg_val = _readout(bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids,
                                      coef, inter)

        row = {"image_path": r["image_path"], "pos_probe": pos_probe, "neg_probe": neg_probe,
               "pos_val": pos_val, "neg_val": neg_val}
        for g in GROUPS:
            di, ri = groups[g]
            hooks = []
            for L in patch_layers:
                donor_vals = dcache[f"blocks.{L}.hook_resid_post"][0, torch.as_tensor(di)]
                hooks.append((f"blocks.{L}.hook_resid_post", _patch_hook(ri, donor_vals)))
            p_probe, p_val = _readout(bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids,
                                      coef, inter, extra_hooks=hooks)
            row[f"patch_{g}_probe"] = p_probe
            row[f"patch_{g}_val"] = p_val
        rows.append(row)
        n_ok += 1

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "patching.parquet")
    rec = _recovery(df)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "critical_layer": crit, "patch_layers": patch_layers,
        "n_images": n_ok, "n_skipped": n_skip, "n_segmentation_dropped": seg_bad,
        "donor_pos_context": pos_ctx, "recipient_neg_context": neg_ctx,
        "pos_idx": pj, "neg_idx": ni,
        "recovery": rec, "verdict": _verdict(rec),
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
    """Aggregate recovery = (mean_patched − mean_neg) / (mean_pos − mean_neg) per group & read-out."""
    out = {k: float(df[k].mean()) for k in ("pos_probe", "neg_probe", "pos_val", "neg_val")}
    dprobe = out["pos_probe"] - out["neg_probe"]
    dval = out["pos_val"] - out["neg_val"]
    for g in GROUPS:
        out[g] = {
            "probe": float((df[f"patch_{g}_probe"].mean() - out["neg_probe"]) / dprobe) if dprobe else float("nan"),
            "val": float((df[f"patch_{g}_val"].mean() - out["neg_val"]) / dval) if dval else float("nan"),
        }
    return out


def _verdict(rec) -> str:
    g = lambda k: rec[k]["probe"]  # noqa: E731
    ri, rq, rs, rt = g("image"), g("question"), g("structure"), g("text_all")
    sinks = {"BOS": g("bos"), "prefix-delims": g("prefix_delim"), "suffix-delims": g("suffix_delim")}
    dom = max(sinks, key=lambda k: sinks[k])
    parts = (f"(probe recovery) image {ri:.0%}, question {rq:.0%} | sinks: BOS {sinks['BOS']:.0%}, "
             f"prefix-delims {sinks['prefix-delims']:.0%}, suffix-delims {sinks['suffix-delims']:.0%} "
             f"→ structure {rs:.0%}, all-text {rt:.0%}")
    concl = ["IMAGE tokens causally INERT" if abs(ri) < 0.05 else f"IMAGE tokens carry {ri:.0%}"]
    if rs > 1.5 * max(rq, 1e-3):
        concl.append(f"the context is BROADCAST into sink/turn tokens (structure {rs:.0%} > question "
                     f"{rq:.0%}); dominant sink = {dom} ({sinks[dom]:.0%})")
    else:
        concl.append(f"question and structure carry it comparably ({rq:.0%} / {rs:.0%}); "
                     f"dominant sink = {dom} ({sinks[dom]:.0%})")
    concl.append(f"sink parts sum {sum(sinks.values()):.0%} vs structure {rs:.0%} (additivity check)")
    concl.append(f"~{1.0 - rt:.0%} remains in the unpatched CONTEXT tokens")
    return parts + ". " + "; ".join(concl) + "."


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
