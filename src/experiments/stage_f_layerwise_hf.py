"""Stage F layer-resolved localization on raw HuggingFace — re-score §6.1 without the bridge.

WHY. This is the last un-re-measured piece of the mechanism section (docs/paper-retraction-audit.md).
Both statistics it feeds are probe-projected DIFFERENCES between two runs --- the class of measurement
that reproduced everywhere it has been checked (Stage C rho +0.507 -> +0.510, Stage D slope +0.329 ->
+0.336, patching recovery within a few points) --- so we expect it to hold. It gets measured anyway,
because "expected to survive" is what we said about the turn-boundary concentration, and that did not.

WHAT IS MEASURED. Two lenses on the LAST token, using the frozen pleasantness probe's weight as a
fixed direction w = unit(coef). This is a logit-lens diagnostic read off every layer, not a re-fit
probe:

  * RESID lens  w . resid_post[L]  --- the running pleasantness read-out. Where the neutral / negative
    / positive curves diverge is where the context signal has entered the read-out stream.
  * ATTN lens   w . attn_out[L]    --- how much layer L's attention output projects onto that
    direction, i.e. where the differentiating signal is WRITTEN.

THE TAPS. `resid_post[L]` is the decoder layer's own output, the identification `stage_d_steering_hf`
relies on. `attn_out[L]` is `post_attention_layernorm`, NOT `self_attn` --- Gemma post-norms the
attention output before the residual add and TransformerLens folds that into its attn-block output.
The Stage C port measured the difference (r^2 +0.634 vs -6.26); getting it wrong here would not error,
it would just flatten the attn lens into noise.

The headline numbers in the paper come from the scale-free re-analysis, not from this file's raw
projections, which grow with residual-stream norm. Run `analyze_stage_f_layerwise --parquet
layerwise_hf.parquet` afterwards for the paired effect size d(L).

    python -m src.experiments.stage_f_layerwise_hf
    python -m src.experiments.analyze_stage_f_layerwise --parquet layerwise_hf.parquet

Published reference (bridge, layerwise_normalized.json): onset near L13 (|d| = 0.21 against a 0.125
noise level), amplifying ~7x to a peak near L28 (|d| = 1.47).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE, context_prompt)
from ..paths import STAGE_A_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .shared.hf_runtime import encode_image_prompt, find_language_layers, load_gemma_hf
from .stage_f_patching_hf import select_positive_images

CONDS = ("neutral", "negative", "positive")


@contextmanager
def dual_lens(model, tap: str = "post_attention_layernorm"):
    """Capture the LAST-token `resid_post` and `attn_out` of every decoder layer in one forward.

    Two hooks per layer rather than one full-sequence cache: the lens only ever reads the final
    position, and caching every layer at full sequence length for 840 forwards would be pointless
    memory traffic.
    """
    layers = find_language_layers(model, verbose=False)
    resid: dict[int, np.ndarray] = {}
    attn: dict[int, np.ndarray] = {}
    handles = []

    def make(store, i):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            store[i] = t[0, -1].detach().float().cpu().numpy()
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make(resid, i)))
        handles.append(getattr(layer, tap).register_forward_hook(make(attn, i)))
    try:
        yield resid, attn
    finally:
        for h in handles:
            h.remove()


def run(config_path: str, limit_override: int | None = None,
        tap: str = "post_attention_layernorm") -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    crit = int(cfg.get("critical_layer", 18))
    n_images = limit_override or int(cfg.get("layerwise_n_images", 60))

    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    w = np.asarray(probes.coef[probes.index("pleasantness")], dtype=np.float32)
    w = w / (np.linalg.norm(w) or 1.0)

    conditions = ([("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]
                  + [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
                  + [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)])

    sel = select_positive_images(cfg.get("split", "test"), n_images)
    model, processor = load_gemma_hf(cfg.get("model", "google/gemma-3-4b-it"))
    n_layers = len(find_language_layers(model, verbose=False))

    rows, n_skip, n_ok = [], 0, 0
    with dual_lens(model, tap) as (resid, attn):
        for _, r in tqdm(list(sel.iterrows()), desc="stage-f layerwise (raw HF)"):
            try:
                img = Image.open(r["image_path"]).convert("RGB")
            except (FileNotFoundError, OSError):
                n_skip += 1
                continue
            for cond, cid, sentence in conditions:
                enc = encode_image_prompt(processor, img, context_prompt(sentence), model.device)
                with torch.no_grad():
                    model(**enc)
                for L in range(n_layers):
                    rows.append({"image_path": r["image_path"], "condition": cond,
                                 "context_id": cid, "text_code": TEXT_CODE[cond], "layer": L,
                                 "resid_proj": float(resid[L] @ w),
                                 "attn_proj": float(attn[L] @ w)})
            n_ok += 1

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "layerwise_hf.parquet")

    resid_m = {c: [float(df[(df.condition == c) & (df.layer == L)]["resid_proj"].mean())
                   for L in range(n_layers)] for c in CONDS}
    attn_m = {c: [float(df[(df.condition == c) & (df.layer == L)]["attn_proj"].mean())
                  for L in range(n_layers)] for c in CONDS}

    metrics = {
        "run": run_stamp(), "git": git_hash(), "stack": "raw_hf", "tap": tap,
        "n_layers": n_layers, "critical_layer": crit, "n_images": n_ok, "n_skipped": n_skip,
        "image_group": "positive", "resid_lens": resid_m, "attn_lens": attn_m,
        "lens": "w = unit(frozen pleasantness probe coef); projected off-layer (logit-lens diagnostic)",
        "note": ("raw-HF re-score. Raw projections grow with residual-stream norm — read the paper's "
                 "numbers off analyze_stage_f_layerwise --parquet layerwise_hf.parquet, which reports "
                 "the scale-free paired effect size d(L)."),
    }
    save_json(metrics, STAGE_F_DIR / "layerwise_hf_metrics.json")

    print(f"\nStage F layerwise (RAW HF) — {n_ok} positive-group images x {len(conditions)} contexts "
          f"({n_skip} skipped), {n_layers} layers.\n")
    print(f"  {'L':>3s} {'resid neu':>9s} {'resid neg':>9s} {'resid pos':>9s} "
          f"{'neg-pos':>8s} | {'attn neg-pos':>12s}")
    for L in range(n_layers):
        print(f"  {L:>3d} {resid_m['neutral'][L]:>+9.3f} {resid_m['negative'][L]:>+9.3f} "
              f"{resid_m['positive'][L]:>+9.3f} "
              f"{resid_m['negative'][L] - resid_m['positive'][L]:>+8.3f} | "
              f"{attn_m['negative'][L] - attn_m['positive'][L]:>+12.3f}")
    print(f"\n  data -> {STAGE_F_DIR/'layerwise_hf.parquet'}")
    print("  NEXT: python -m src.experiments.analyze_stage_f_layerwise --parquet layerwise_hf.parquet")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F layerwise localization on raw HF")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="positive-group image count (default 60)")
    ap.add_argument("--tap", default="post_attention_layernorm",
                    help="raw-HF equivalent of the bridge's blocks.L.hook_attn_out")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit, tap=args.tap)


if __name__ == "__main__":
    main()
