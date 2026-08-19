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

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      context_prompt)
from ..paths import STAGE_F_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_a_steering import emotion_token_ids, valence_score
from .stage_f_attribution import _stash_hook, segment_positions
from .stage_f_conflict import select_extreme_images
from .stage_f_patching import _patch_hook, _readout

# image = the visual-valence test; context/question/structure = did image valence broadcast into text?;
# text_all = all non-image alignable tokens; all = everything but the read-out query (≈100% sanity).
GROUPS = ("image", "context", "question", "structure", "text_all", "all")
CONTEXT_BANKS = {"neutral": NEUTRAL_CONTEXTS, "positive": POSITIVE_CONTEXTS, "negative": NEGATIVE_CONTEXTS}


def _cross_groups(seg: dict) -> tuple[dict, dict]:
    """Token-group index arrays for the identical-sequence cross-image design (donor idx == recipient).

    Every group excludes the final token (the read-out query position): patching it would import the
    donor's read-out state directly instead of attributing to source tokens. Because `input_ids` are
    identical across donor/recipient, one segmentation drives both runs.
    """
    n = int(seg["n"])
    img = np.asarray(seg["image"], dtype=int)
    ctx = np.asarray(seg["context"], dtype=int)
    q = np.asarray(seg["question"], dtype=int)
    ok = {}
    ok["image"] = bool(len(img) == 256)
    ok["question"] = bool(len(q))
    ok["context"] = bool(len(ctx))
    img_start = int(img[0]) if len(img) else 0
    q_end = int(q[-1]) + 1 if len(q) else (int(img[-1]) + 1 if len(img) else 0)
    pre = np.arange(0, img_start)                 # <bos><start_of_turn>user\n … up to the image
    suf = np.arange(q_end, n - 1)                 # <end_of_turn>\n<start_of_turn>model (drop last token)
    structure = np.concatenate([pre, suf]) if (len(pre) or len(suf)) else np.array([], dtype=int)
    ok["structure"] = bool(len(structure))
    # exclude the last token everywhere (it is n-1; none of img/ctx/q/suf include it by construction)
    text_all = np.unique(np.concatenate([a for a in (ctx, q, structure) if len(a)])) if (len(ctx) or len(q) or len(structure)) else np.array([], dtype=int)
    all_ = np.unique(np.concatenate([a for a in (img, text_all) if len(a)])) if (len(img) or len(text_all)) else np.array([], dtype=int)
    ok["text_all"] = bool(len(text_all))
    ok["all"] = bool(len(all_))
    out = {"image": img, "context": ctx, "question": q, "structure": structure,
           "text_all": text_all, "all": all_}
    return out, ok


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
    sel = select_extreme_images(cfg.get("split", "test"), n_pairs * 2)
    pos = sel[sel["image_group"] == "positive"].sort_values("valence", ascending=False).reset_index(drop=True)
    neg = sel[sel["image_group"] == "negative"].sort_values("valence").reset_index(drop=True)
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
        seg = segment_positions(bridge, rin["input_ids"])                   # identical for both runs
        groups, ok = _cross_groups(seg)
        if not all(ok.get(g) for g in GROUPS):
            seg_bad += 1
            continue

        with torch.no_grad():
            _, dcache = bridge.run_with_cache(din["input_ids"], pixel_values=din["pixel_values"],
                                              names_filter=lambda nm: nm in resid_names)
        pos_probe, pos_val = _readout(bridge, din["input_ids"], din["pixel_values"], name, tok_ids, coef, inter)
        neg_probe, neg_val = _readout(bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids, coef, inter)

        row = {"donor_path": pos.loc[k, "image_path"], "recipient_path": neg.loc[k, "image_path"],
               "donor_valence": float(pos.loc[k, "valence"]), "recipient_valence": float(neg.loc[k, "valence"]),
               "pos_probe": pos_probe, "neg_probe": neg_probe, "pos_val": pos_val, "neg_val": neg_val}
        for g in GROUPS:
            idx = groups[g]
            hooks = []
            for L in patch_layers:
                donor_vals = dcache[f"blocks.{L}.hook_resid_post"][0, torch.as_tensor(idx)]
                hooks.append((f"blocks.{L}.hook_resid_post", _patch_hook(idx, donor_vals)))
            p_probe, p_val = _readout(bridge, rin["input_ids"], rin["pixel_values"], name, tok_ids,
                                      coef, inter, extra_hooks=hooks)
            row[f"patch_{g}_probe"] = p_probe
            row[f"patch_{g}_val"] = p_val
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "cross_patching.parquet")
    rec = _recovery(df)
    metrics = _metrics(rec, patch_layers, crit, ctx, context_polarity, len(rows), n_skip, seg_bad)
    save_json(metrics, STAGE_F_DIR / "cross_patching_metrics.json")
    _print(rec, metrics, patch_layers)
    return metrics


# --------------------------------------------------------------------------- recovery + bootstrap CI
def _boot(patch, pos, neg, boot_idx) -> tuple[float, list]:
    """recovery = (mean_patch − mean_neg)/(mean_pos − mean_neg), with a clustered bootstrap over pairs."""
    num, den = patch - neg, pos - neg
    d = den.mean()
    est = float(num.mean() / d) if d else float("nan")
    vals = []
    for ix in boot_idx:
        dd = den[ix].mean()
        if dd:
            vals.append(num[ix].mean() / dd)
    ci = [float(x) for x in np.percentile(vals, [2.5, 97.5])] if vals else [float("nan"), float("nan")]
    return est, ci


def _recovery(df, n_boot: int = 2000, seed: int = 0) -> dict:
    out = {k: float(df[k].mean()) for k in ("pos_probe", "neg_probe", "pos_val", "neg_val")}
    if df.empty:
        return out
    n = len(df)
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]   # shared resamples across groups
    for g in GROUPS:
        est_p, ci_p = _boot(df[f"patch_{g}_probe"].to_numpy(), df["pos_probe"].to_numpy(),
                            df["neg_probe"].to_numpy(), boot_idx)
        est_v, ci_v = _boot(df[f"patch_{g}_val"].to_numpy(), df["pos_val"].to_numpy(),
                            df["neg_val"].to_numpy(), boot_idx)
        out[g] = {"probe": est_p, "probe_ci95": ci_p, "val": est_v, "val_ci95": ci_v}
    out["n_pairs"] = int(n)
    return out


def _probe_valid(patch_layers, crit) -> bool:
    """The probe tap is attn_out L{crit}, computed from L{crit-1}'s output — UPSTREAM of resid_post at
    L>=crit. So probe recovery is invariant-by-construction (identically 0) once any patched layer is
    >= crit; only behavioral valence is meaningful for such a band."""
    return bool(patch_layers) and max(patch_layers) < crit


def _verdict(rec, patch_layers, crit) -> str:
    if "image" not in rec:
        return "no pairs analysed"
    pv = _probe_valid(patch_layers, crit)
    m = "probe" if pv else "val"
    ri, rt, ra = rec["image"][m], rec["text_all"][m], rec["all"][m]
    ci = rec["image"][f"{m}_ci95"]
    note = "" if pv else (f" [NOTE: patched at/after the L{crit} probe tap, so probe recovery is "
                          f"invariant-by-construction — verdict uses behavioral VALENCE]")
    lead = ("VISUAL VALENCE LIVES IN THE IMAGE TOKENS" if ri > 0.5 else
            "image tokens carry a MODERATE share" if ri > 0.2 else
            "image tokens carry LITTLE in this band")
    return (f"{lead}{note}: patching image tokens recovers {ri:.0%} [{ci[0]:.0%},{ci[1]:.0%}] of the "
            f"image-driven read-out gap, vs all-text {rt:.0%}. Sanity: patching every token bar the "
            f"read-out query recovers {ra:.0%} (expect ~100%). Mirror of the same-image result, where "
            f"image tokens were inert for the TEXT context delta.")


def _metrics(rec, patch_layers, crit, ctx, pol, n_ok, n_skip, seg_bad) -> dict:
    return {"run": run_stamp(), "git": git_hash(), "critical_layer": crit, "patch_layers": patch_layers,
            "n_pairs": n_ok, "n_skipped": n_skip, "n_segmentation_dropped": seg_bad,
            "context_polarity": pol, "context": ctx, "recovery": rec,
            "probe_valid": _probe_valid(patch_layers, crit), "verdict": _verdict(rec, patch_layers, crit),
            "design": ("CROSS-IMAGE: donor=positive image, recipient=negative image, SAME context → "
                       "identical input_ids, all positions (incl. context) patchable. recovery = "
                       "(patched-neg_img)/(pos_img-neg_img) at L{c} read-out, resid_post over band "
                       "{b}.").format(c=crit, b=patch_layers)}


def _print(rec, metrics, patch_layers) -> None:
    print(f"\nStage F CROSS-IMAGE patching — {metrics['n_pairs']} donor/recipient pairs "
          f"({metrics['n_skipped']} skipped, {metrics['n_segmentation_dropped']} seg-dropped); "
          f"context={metrics['context_polarity']} \"{metrics['context'][:34]}\"; "
          f"patch resid_post {patch_layers[0]}-{patch_layers[-1]}.")
    if "image" not in rec:
        print("  no pairs analysed."); return
    print(f"  baselines: probe pos-img {rec['pos_probe']:+.3f} / neg-img {rec['neg_probe']:+.3f}  |  "
          f"valence pos-img {rec['pos_val']:+.3f} / neg-img {rec['neg_val']:+.3f}")
    if not metrics.get("probe_valid", True):
        print(f"  NOTE: patched at/after the L{metrics['critical_layer']} probe tap → the probe column "
              f"is invariant-by-construction (all 0); read the VALENCE column for this band.")
    print(f"\n  {'group':10s} {'recovery(probe)':>22s} {'recovery(valence)':>22s}")
    for g in GROUPS:
        p, v = rec[g]["probe"], rec[g]["val"]
        pc, vc = rec[g]["probe_ci95"], rec[g]["val_ci95"]
        print(f"  {g:10s} {p*100:>7.0f}% [{pc[0]*100:>4.0f},{pc[1]*100:>4.0f}]     "
              f"{v*100:>7.0f}% [{vc[0]*100:>4.0f},{vc[1]*100:>4.0f}]")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  data -> {STAGE_F_DIR/'cross_patching.parquet'}   "
          f"metrics -> {STAGE_F_DIR/'cross_patching_metrics.json'}")
    print("  NEXT (band sweep — does visual valence read out earlier/later?): "
          "--layers 0-12 and --layers 18-28")


def reanalyze(config_path: str) -> dict:
    load_config(config_path)
    pq = STAGE_F_DIR / "cross_patching.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run the cross-image patching pass on the A100 first.")
    df = pd.read_parquet(pq)
    rec = _recovery(df)
    mpath = STAGE_F_DIR / "cross_patching_metrics.json"
    old = load_config(mpath) if mpath.exists() else {}
    metrics = _metrics(rec, old.get("patch_layers", []), old.get("critical_layer", 18),
                       old.get("context", ""), old.get("context_polarity", "neutral"),
                       int(len(df)), old.get("n_skipped", 0), old.get("n_segmentation_dropped", 0))
    save_json(metrics, mpath)
    _print(rec, metrics, old.get("patch_layers", [0, 0]) or [0, 0])
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
