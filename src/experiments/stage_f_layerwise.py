"""Stage F — layer-resolved localization: WHERE does the context effect enter the read-out?

The L18 last-token attribution (`stage_f_attribution.py`) ruled OUT the read-out-layer mechanisms:
attention shares are invariant to context polarity (no re-routing) and the direct last-token→context
contribution is ~6% of the effect. So the negativity dominance is established UPSTREAM — the context
reshapes the high-attention (image/question) token representations before L18. This script finds the
depth band where that happens.

Two probe-lenses on the LAST token, one multi-layer cache per forward, using the frozen pleasantness
probe's weight as a fixed direction w = unit(coef) (logit-lens: an L18 attn_out direction read off
every layer — a diagnostic, not a re-fit probe):
  * RESID lens  — w · resid_post[L]  = the running pleasantness read-out at layer L. The layer where
    the neutral/negative/positive curves DIVERGE is where the context signal has entered the read-out
    stream.
  * ATTN lens   — w · attn_out[L]    = how much layer L's attention output projects onto the read-out
    direction. The layer maximizing |neg − pos| is where the differentiating signal is WRITTEN.

Positive-image group only (the clean cell where the positive channel collapses). Sweeps the full
context bank, averaged per polarity. Cheap: one forward per (image, context), no knockouts.

Run on the A100 with HF_TOKEN + EMOTIC. Frozen probe; never re-fit. `--limit N` sets image count.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import keep_language_taps
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE, context_prompt)
from ..paths import FIGURES_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .stage_f_conflict import select_extreme_images

CONDS = ("neutral", "negative", "positive")


def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    crit = int(cfg.get("critical_layer", 18))
    n_images = limit_override or int(cfg.get("layerwise_n_images", 60))

    from ..paths import STAGE_A_DIR
    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    w = np.asarray(probes.coef[pi], dtype=np.float32)
    w = w / (np.linalg.norm(w) or 1.0)   # unit pleasantness direction (the lens)

    conditions = ([("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]
                  + [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
                  + [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)])

    sel = select_extreme_images(cfg.get("split", "test"), n_images * 2)
    sel = sel[sel["image_group"] == "positive"].head(n_images).reset_index(drop=True)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    n_layers = int(bridge.cfg.n_layers)
    keep = keep_language_taps(("hook_attn_out", "hook_resid_post"))
    wt = torch.tensor(w, dtype=torch.float32, device=next(bridge.parameters()).device)

    rows, n_skip, n_ok = [], 0, 0
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f layerwise"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for cond, cid, sentence in conditions:
            inputs = build_image_inputs(bridge, img, prompt=context_prompt(sentence))
            ids, pv = inputs["input_ids"], inputs["pixel_values"]
            last = ids.shape[-1] - 1
            with torch.no_grad():
                _, cache = bridge.run_with_cache(ids, pixel_values=pv, names_filter=keep)
            for L in range(n_layers):
                a = cache[f"blocks.{L}.hook_attn_out"][0, last].float()
                h = cache[f"blocks.{L}.hook_resid_post"][0, last].float()
                rows.append({"image_path": r["image_path"], "condition": cond, "context_id": cid,
                             "text_code": TEXT_CODE[cond], "layer": L,
                             "resid_proj": float(h @ wt), "attn_proj": float(a @ wt)})
        n_ok += 1

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "layerwise.parquet")

    # mean projections per (condition, layer): {cond: [n_layers]}
    resid = {c: [float(df[(df.condition == c) & (df.layer == L)]["resid_proj"].mean())
                 for L in range(n_layers)] for c in CONDS}
    attn = {c: [float(df[(df.condition == c) & (df.layer == L)]["attn_proj"].mean())
                for L in range(n_layers)] for c in CONDS}

    loc = _localize(resid, attn, crit, n_layers)
    metrics = {
        "run": run_stamp(), "git": git_hash(), "n_layers": n_layers, "critical_layer": crit,
        "n_images": n_ok, "n_skipped": n_skip, "image_group": "positive",
        "resid_lens": resid, "attn_lens": attn, "localization": loc,
        "lens": "w = unit(frozen pleasantness probe coef); projected off-layer (logit-lens diagnostic)",
    }
    save_json(metrics, STAGE_F_DIR / "layerwise_metrics.json")
    _plot(resid, attn, crit, loc, n_layers)

    print(f"\nStage F layerwise — {n_ok} positive-group images x {len(conditions)} contexts "
          f"({n_skip} skipped), {n_layers} layers.\n")
    print(f"  {'L':>3s} {'resid neu':>9s} {'resid neg':>9s} {'resid pos':>9s} "
          f"{'neg-pos':>8s} | {'attn neg-pos':>12s}")
    for L in range(n_layers):
        sep = resid["negative"][L] - resid["positive"][L]
        asep = attn["negative"][L] - attn["positive"][L]
        mark = "  <- onset" if L == loc["resid_onset_layer"] else \
               "  <- attn-write peak" if L == loc["attn_write_peak_layer"] else ""
        print(f"  {L:>3d} {resid['neutral'][L]:>+9.3f} {resid['negative'][L]:>+9.3f} "
              f"{resid['positive'][L]:>+9.3f} {sep:>+8.3f} | {asep:>+12.3f}{mark}")
    print(f"\n  {loc['summary']}")
    print(f"  data -> {STAGE_F_DIR/'layerwise.parquet'}   figure -> {FIGURES_DIR/'stage_f_layerwise.png'}")
    return metrics


def _localize(resid, attn, crit, n_layers) -> dict:
    """Onset layer (running read-out reaches 50% of its critical-layer separation) + attn-write peak."""
    sep = [resid["negative"][L] - resid["positive"][L] for L in range(n_layers)]
    ref = sep[crit] if crit < n_layers else sep[-1]
    onset = crit
    if ref != 0:
        for L in range(n_layers):
            if np.sign(sep[L]) == np.sign(ref) and abs(sep[L]) >= 0.5 * abs(ref):
                onset = L
                break
    asep = [abs(attn["negative"][L] - attn["positive"][L]) for L in range(n_layers)]
    peak = int(np.argmax(asep))
    # half of the write mass is below which layer? (cumulative |neg-pos| attn writes)
    tot = sum(asep) or 1.0
    cum, half = 0.0, n_layers
    for L in range(n_layers):
        cum += asep[L]
        if cum >= 0.5 * tot:
            half = L
            break
    summary = (f"LOCALIZATION: running read-out (neg−pos) reaches 50% of its L{crit} separation "
               f"({ref:+.3f}) by layer {onset}; the attention that WRITES the differentiating signal "
               f"peaks at layer {peak} (half the cumulative write mass below layer {half}). "
               f"=> the context effect enters the read-out in the {'early' if onset < n_layers//3 else 'mid' if onset < 2*n_layers//3 else 'late'}"
               f" band, not at the L{crit} read-out. NEXT: activation-patch image- vs question-token "
               f"residual groups around layer {min(onset, peak)} to name the carrier.")
    return {"resid_onset_layer": int(onset), "attn_write_peak_layer": peak,
            "attn_write_half_below_layer": int(half), "critical_layer_separation": float(ref),
            "summary": summary}


def _plot(resid, attn, crit, loc, n_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(n_layers))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for c, style in zip(CONDS, ("k-", "r-", "g-")):
        axes[0].plot(xs, resid[c], style, label=c, lw=1.8)
    axes[0].axvline(crit, color="gray", ls=":", lw=1, label=f"L{crit} read-out")
    axes[0].axvline(loc["resid_onset_layer"], color="orange", ls="--", lw=1,
                    label=f"onset L{loc['resid_onset_layer']}")
    axes[0].set_title("running read-out (resid_post · w)"); axes[0].set_xlabel("layer")
    axes[0].set_ylabel("pleasantness projection"); axes[0].legend(fontsize=7)
    axes[1].plot(xs, [attn["negative"][L] - attn["positive"][L] for L in xs], "b-", lw=1.8)
    axes[1].axvline(loc["attn_write_peak_layer"], color="purple", ls="--", lw=1,
                    label=f"write peak L{loc['attn_write_peak_layer']}")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_title("per-layer write (attn_out · w): neg − pos"); axes[1].set_xlabel("layer")
    axes[1].legend(fontsize=7)
    fig.suptitle("Stage F layerwise localization (positive images): where context enters the read-out")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_f_layerwise.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — layer-resolved localization")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="positive-group image count (default 60)")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit)


if __name__ == "__main__":
    main()
