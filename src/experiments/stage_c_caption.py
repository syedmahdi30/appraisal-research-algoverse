"""Stage C caption baseline — shared appraisal geometry vs verbalization (experiment-1.md).

The read-out transfer (stage_c_transfer.py) shows the frozen text pleasantness direction
tracks EMOTIC valence under image conditioning (rho~0.48). That alone cannot say WHY:
  (A) shared geometry — the model represents pleasantness the same way for text and images; or
  (B) verbalization — the model internally describes the image, and the pleasantness direction
      just reads that implicit caption.

This baseline separates them. For each image we GENERATE a NEUTRAL caption ("Describe this
image in one sentence." — no emotion words invited), then run the caption through the TEXT
pipeline and apply the SAME frozen probe, correlating with EMOTIC valence (rho_cap):
  - rho_cap >= rho_img  -> the appraisal signal is fully in a neutral verbal description
                           => transfer is VERBALIZATION-MEDIATED (the mundane explanation).
  - rho_cap << rho_img  -> image activations carry appraisal info the caption does not
                           => SHARED GEOMETRY (the non-verbal claim).

Preconditions: results/stage_a/probes.npz and results/stage_c/metrics.json (for rho_img).
Generation is autoregressive and slow — run `--preview 5` first to eyeball the captions.
Run on the A100 with HF_TOKEN set and EMOTIC downloaded. Never re-fit probes (data-rules.md).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import keep_language_taps
from ..bridge.multimodal import TEXT_EMOTION_PROMPT, build_image_inputs
from ..data.emotic import load_split as load_emotic_split
from ..paths import FIGURES_DIR, STAGE_A_DIR, STAGE_C_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import load_config, load_probes, run_stamp, save_json
from .stage_c_transfer import _corr

# Neutral, non-emotional caption prompt — the fair test is whether a plain description still
# carries the appraisal signal. Emotion words are deliberately NOT invited (no leakage).
CAPTION_PROMPT = (
    "<start_of_turn>user\n<start_of_image>"
    "Describe this image in one sentence.<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def generate_caption(bridge, image, max_new_tokens, dev) -> str:
    """Greedy-decode a neutral caption for one image via the underlying HF model.

    Uses bridge.original_model.generate (the raw Gemma3 HF path — more robust than the TL
    generate wrapper). pixel_values is consumed on the first decode step (vision runs once).
    """
    inputs = build_image_inputs(bridge, image, prompt=CAPTION_PROMPT)
    kw = {k: v.to(dev) for k, v in inputs.items() if torch.is_tensor(v)}
    prompt_len = kw["input_ids"].shape[-1]
    with torch.no_grad():
        gen = bridge.original_model.generate(**kw, max_new_tokens=max_new_tokens, do_sample=False)
    new = gen[0, prompt_len:]
    text = bridge.tokenizer.decode(new, skip_special_tokens=True).strip()
    # Gemma prefixes a meta line ("Here's a one-sentence description...:\n\n<caption>").
    # Drop it so the TEXT pipeline sees a natural description, not the boilerplate.
    if "\n\n" in text:
        text = text.split("\n\n", 1)[1].strip()
    return text


def text_readout(bridge, caption, layer, tap) -> np.ndarray:
    """Last-token activation at the frozen probe site for a caption run through the TEXT prompt."""
    keep = keep_language_taps((tap,))
    name = f"blocks.{layer}.{tap}"
    ids = bridge.to_tokens(TEXT_EMOTION_PROMPT.format(text=caption))
    with torch.no_grad():
        _, cache = bridge.run_with_cache(ids, names_filter=keep)
    return cache[name][0, ids.shape[-1] - 1].float().cpu().numpy()


def run(config_path: str, preview: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()

    probes = load_probes(STAGE_A_DIR / "probes.npz")
    stage_a = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    layer = int(cfg.get("critical_layer", stage_a.get("critical_layer", 18)))
    tap = cfg.get("tap", "hook_attn_out")
    seed = int(cfg.get("seed", 0))
    n_images = cfg.get("n_images")
    appraisals = [a for a in cfg.get("appraisals", ["pleasantness", "unpleasantness"]) if a in probes.names]
    max_new = int(cfg.get("caption_max_new_tokens", 40))

    df = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    if n_images and n_images < len(df):
        df = df.sample(n=int(n_images), random_state=seed).reset_index(drop=True)
    if preview:
        df = df.head(preview)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    dev = next(bridge.parameters()).device

    captions, acts, valid = [], [], []
    for i, path in enumerate(tqdm(df["image_path"].tolist(), desc="caption+readout")):
        try:
            cap = generate_caption(bridge, Image.open(path).convert("RGB"), max_new, dev)
            acts.append(text_readout(bridge, cap, layer, tap))
            captions.append(cap)
            valid.append(True)
            if i < 5:
                tqdm.write(f"  [caption {i}] {cap}")
        except (FileNotFoundError, OSError):
            valid.append(False)

    if preview:
        print("\nPreview only — check the captions above look like plain, non-emotional "
              "descriptions. Re-run WITHOUT --preview for the full baseline.")
        return {"preview": True, "captions": captions}

    valid = np.array(valid, dtype=bool)
    X_cap = np.stack(acts)
    dfv = df.loc[valid].reset_index(drop=True)
    valence = dfv["valence"].to_numpy(dtype=np.float64) if "valence" in dfv.columns else np.full(len(dfv), np.nan)

    img_metrics = load_config(STAGE_C_DIR / "metrics.json") if (STAGE_C_DIR / "metrics.json").exists() else {}
    metrics = {
        "run": run_stamp(), "layer": layer, "tap": tap, "seed": seed,
        "n_captioned": len(dfv), "n_skipped_unreadable": int((~valid).sum()),
        "caption_prompt": CAPTION_PROMPT.strip(), "max_new_tokens": max_new,
        "sample_captions": captions[:10],
        "caption_readout": {}, "compare_to_image": {},
    }
    for a in appraisals:
        coef, inter = probes.coef[probes.index(a)], probes.intercept[probes.index(a)]
        cap_corr = _corr(predict(X_cap, coef, inter), valence)
        metrics["caption_readout"][a] = cap_corr
        img_sp = img_metrics.get("image_readout", {}).get(a, {}).get("vs_valence", {}).get("spearman")
        if img_sp is not None and cap_corr["spearman"] is not None:
            metrics["compare_to_image"][a] = {
                "caption_spearman": cap_corr["spearman"], "image_spearman": img_sp,
                "caption_minus_image": cap_corr["spearman"] - img_sp,
            }

    metrics["verdict"] = _caption_verdict(metrics)
    save_json(metrics, STAGE_C_DIR / "caption_metrics.json")
    _plot(metrics)
    return metrics


def _caption_verdict(metrics) -> str:
    """Provisional mechanism verdict (human confirms). Compares |rho_cap| vs |rho_img| for
    pleasantness — same magnitude means a neutral caption already carries the signal."""
    c = metrics["compare_to_image"].get("pleasantness")
    if not c:
        return "inconclusive (no image-side comparison — run stage_c_transfer first)"
    cap, img = abs(c["caption_spearman"]), abs(c["image_spearman"])
    if img < 0.05:
        return "inconclusive (image read-out itself is ~0)"
    if cap >= img - 0.05:
        return (f"verbalization-mediated: caption read-out matches/exceeds image "
                f"(|cap|={cap:.2f} >= |img|={img:.2f}) — a neutral description already carries the "
                f"appraisal, so direct transfer is NOT evidence of non-verbal shared geometry")
    if cap <= 0.5 * img:
        return (f"supports shared geometry: caption read-out is much weaker "
                f"(|cap|={cap:.2f} vs |img|={img:.2f}) — image activations carry appraisal info a "
                f"neutral caption does not")
    return (f"partial: caption explains some but not all (|cap|={cap:.2f} vs |img|={img:.2f}) — "
            f"the image path adds appraisal signal beyond a neutral caption")


def _plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp = metrics["compare_to_image"]
    if not comp:
        return
    appraisals = list(comp)
    x = np.arange(len(appraisals))
    fig, ax = plt.subplots(figsize=(1.6 * len(appraisals) + 3, 4.2))
    ax.bar(x - 0.2, [abs(comp[a]["image_spearman"]) for a in appraisals], 0.4, label="image read-out")
    ax.bar(x + 0.2, [abs(comp[a]["caption_spearman"]) for a in appraisals], 0.4, label="caption read-out")
    ax.set_xticks(x)
    ax.set_xticklabels(appraisals)
    ax.set_ylabel("|Spearman| vs EMOTIC valence")
    ax.set_title(f"Stage C caption baseline (n={metrics['n_captioned']})\n"
                 "image >> caption -> shared geometry; image ~ caption -> verbalization")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_c_caption_baseline.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C caption baseline — verbalization vs shared geometry")
    ap.add_argument("--config", default="config/stage_c.yaml")
    ap.add_argument("--preview", type=int, default=None,
                    help="caption only N images and print them (sanity check before the full run)")
    args = ap.parse_args()
    m = run(args.config, preview=args.preview)
    if m.get("preview"):
        return
    print(f"\nStage C caption baseline — L{m['layer']} {m['tap']}  "
          f"({m['n_captioned']} captioned, {m['n_skipped_unreadable']} skipped)\n")
    for a, c in m["compare_to_image"].items():
        print(f"  {a:16s} caption rho={c['caption_spearman']:+.3f}  vs  image rho={c['image_spearman']:+.3f}  "
              f"(caption−image={c['caption_minus_image']:+.3f})")
    print(f"\n  VERDICT: {m['verdict']}")
    print(f"  metrics -> {STAGE_C_DIR/'caption_metrics.json'}   "
          f"figure -> {FIGURES_DIR/'stage_c_caption_baseline.png'}")


if __name__ == "__main__":
    main()
