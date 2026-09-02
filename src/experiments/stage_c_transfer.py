"""Stage C — cross-modal appraisal read-out with FROZEN text probes (docs/experiment-1.md).

Question: does the text-trained appraisal direction survive when the SAME model looks at an
image? We apply the frozen Stage A probes (layer 18, hook_attn_out) to image-conditioned
last-token activations and ask whether the read-out tracks EMOTIC ground truth.

Why NOT raw r2: the probes predict crowd-enVENT pleasantness on a 1-5 scale; EMOTIC's only
appraisal-like ground truth is continuous valence on 1-10. A frozen 1-5 probe scored by r2
against a 1-10 target looks broken from the scale offset alone, even with perfect rank
agreement. So the transfer metrics here are SCALE-INVARIANT:
  - PRIMARY:   Spearman/Pearson correlation of the pleasantness read-out vs EMOTIC valence.
  - SECONDARY: polarity AUC — does the read-out rank positive-emotion images (shared-7 = joy)
               above negative ones (anger/disgust/fear/sadness)? (categorical, scale-free)
  - CONTROL:   norm-matched random directions — must sit at ~0 correlation.
  - TRANSFER GAP: the same correlation computed on TEXT test activations (probe vs the 1-5
               pleasantness rating) minus the image-side correlation. Apples-to-apples.

EMOTIC has NO ground truth for the other four appraisals (suddenness, predictability,
responsibility), so they are not scored here — they can only be validated via steering
(Stage D). Unpleasantness is scored and is expected to ANTI-correlate with valence.

Never re-fit probes on image data (data-rules.md). Run on the A100 with HF_TOKEN set and
EMOTIC downloaded/converted. See docs/colab.md.
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
from ..data.crowd_envent import load_split as load_text_split
from ..data.emotic import load_split as load_emotic_split
from ..paths import FIGURES_DIR, STAGE_A_DIR, STAGE_C_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import load_config, load_probes, run_stamp, save_json
from .shared.reporting import (
    correlation as _corr,
    polarity_auc as _auc,
    polarity_vector as _polarity,
    random_direction_controls as _random_controls,
    shared_emotic_label as _shared_label,
    transfer_verdict as _verdict,
)


# --------------------------------------------------------------------------- activations
def image_activations(bridge, image_paths, layer, tap):
    """Return (X [m, d], valid_mask [n]) last-token activations under image conditioning.

    Unreadable/missing images are skipped (mask False) rather than killing a long run;
    the count is surfaced in the metrics so it stays honest.
    """
    keep = keep_language_taps((tap,))
    name = f"blocks.{layer}.{tap}"
    rows, valid = [], []
    for path in tqdm(image_paths, desc="image acts"):
        try:
            inputs = build_image_inputs(bridge, Image.open(path).convert("RGB"))
            with torch.no_grad():  # vision tower does 4096-patch eager attn; grad graph OOMs
                _, cache = bridge.run_with_cache(
                    inputs["input_ids"], pixel_values=inputs["pixel_values"], names_filter=keep,
                )
            last = inputs["input_ids"].shape[-1] - 1
            rows.append(cache[name][0, last].float().cpu().numpy())
            valid.append(True)
        except (FileNotFoundError, OSError):
            valid.append(False)
    X = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
    return X, np.array(valid, dtype=bool)


def text_activations(bridge, texts, layer, tap):
    """Return [n, d] last-token text activations at the same probe site (transfer-gap ref)."""
    keep = keep_language_taps((tap,))
    name = f"blocks.{layer}.{tap}"
    rows = []
    for text in tqdm(texts, desc="text acts"):
        ids = bridge.to_tokens(TEXT_EMOTION_PROMPT.format(text=text))
        with torch.no_grad():
            _, cache = bridge.run_with_cache(ids, names_filter=keep)
        last = ids.shape[-1] - 1
        rows.append(cache[name][0, last].float().cpu().numpy())
    return np.stack(rows)


# --------------------------------------------------------------------------- run
def run(config_path: str, n_images_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()

    probes_path = STAGE_A_DIR / "probes.npz"
    if not probes_path.exists():
        raise FileNotFoundError(
            f"{probes_path} missing — Stage A must pass and save frozen probes before Stage C."
        )
    probes = load_probes(probes_path)

    stage_a = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    layer = int(cfg.get("critical_layer", stage_a.get("critical_layer", 18)))
    tap = cfg.get("tap", "hook_attn_out")
    seed = int(cfg.get("seed", 0))
    # override: 0 / negative -> full split (no subset); positive -> that many; None -> config.
    n_images = n_images_override if n_images_override is not None else cfg.get("n_images")
    if n_images is not None and n_images <= 0:
        n_images = None
    appraisals = [a for a in cfg.get("appraisals", ["pleasantness", "unpleasantness"]) if a in probes.names]
    positive = cfg.get("positive_labels", ["joy"])
    negative = cfg.get("negative_labels", ["anger", "disgust", "fear", "sadness"])
    n_random = int(cfg.get("n_random", 100))

    # --- EMOTIC test subset (deterministic) -------------------------------------------
    df = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    if n_images and n_images < len(df):
        df = df.sample(n=int(n_images), random_state=seed).reset_index(drop=True)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))

    X_img, valid = image_activations(bridge, df["image_path"].tolist(), layer, tap)
    n_skipped = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)

    valence = df["valence"].to_numpy(dtype=np.float64) if "valence" in df.columns else np.full(len(df), np.nan)
    shared = [_shared_label(c) for c in df["categories"]]
    polarity = _polarity(shared, positive, negative)
    n_single = int(sum(s is not None for s in shared))

    metrics = {
        "run": run_stamp(), "layer": layer, "tap": tap, "seed": seed,
        "n_images_requested": int(n_images) if n_images else len(df) + n_skipped,
        "n_images_scored": len(df), "n_skipped_unreadable": n_skipped,
        "n_single_label": n_single, "n_dropped_multilabel": len(df) - n_single,
        "polarity_groups": {"positive": positive, "negative": negative},
        "caveats": [
            "EMOTIC has no appraisal ground truth; pleasantness is anchored to continuous "
            "valence (1-10) and to shared-7 emotion polarity — both are proxies.",
            "EMOTIC-26 -> shared-7 mapping is lossy; single-label filtering drops multilabel rows.",
            "Metrics are correlation/AUC (scale-invariant): raw r2 is NOT comparable across the "
            "1-5 (probe) vs 1-10 (valence) scales.",
        ],
        "image_readout": {}, "random_control": {}, "text_reference": {}, "transfer": {},
    }

    # One random-direction null vs valence (same target for every appraisal). The probe's
    # empirical p = fraction of random directions whose |spearman| matches or beats it.
    ctrl = _random_controls(X_img, valence, n_random, seed)
    ctrl_abs = np.asarray(ctrl.pop("_abs"), dtype=float)
    metrics["random_control"] = ctrl

    for a in appraisals:
        coef, inter = probes.coef[probes.index(a)], probes.intercept[probes.index(a)]
        pred = predict(X_img, coef, inter)
        vc = _corr(pred, valence)
        pval = None
        if vc["spearman"] is not None and ctrl_abs.size:
            pval = float((1 + int(np.sum(ctrl_abs >= abs(vc["spearman"])))) / (ctrl_abs.size + 1))
        metrics["image_readout"][a] = {
            "vs_valence": vc,
            "polarity_auc": _auc(pred, polarity),
            "vs_control_p": pval,
        }

    # --- transfer gap: same correlation on TEXT test activations at the same site -------
    if cfg.get("text_reference", True):
        n_text = int(cfg.get("n_text", 1000))
        tdf = load_text_split("test", seed=seed)
        if n_text < len(tdf):
            tdf = tdf.sample(n=n_text, random_state=seed).reset_index(drop=True)
        X_txt = text_activations(bridge, tdf["text"].tolist(), layer, tap)
        for a in appraisals:
            if a not in tdf.columns:
                continue
            coef, inter = probes.coef[probes.index(a)], probes.intercept[probes.index(a)]
            txt_corr = _corr(predict(X_txt, coef, inter), tdf[a].to_numpy(dtype=np.float64))
            metrics["text_reference"][a] = txt_corr
            img_sp = metrics["image_readout"][a]["vs_valence"]["spearman"]
            # Sign-safe retention: the text probe reads its OWN 1-5 rating (unpleasantness
            # correlates + there), the image probe reads valence (unpleasantness correlates
            # - there). Subtracting the two mixes opposite-signed targets, so compare
            # MAGNITUDES: retention = |image effect| / |text effect|.
            if txt_corr["spearman"] and img_sp is not None:
                metrics["transfer"][a] = {
                    "text_abs_spearman": abs(txt_corr["spearman"]),
                    "image_abs_spearman": abs(img_sp),
                    "retention": abs(img_sp) / abs(txt_corr["spearman"]),
                }

    metrics["verdict"] = _verdict(metrics, appraisals)
    save_json(metrics, STAGE_C_DIR / "metrics.json")
    _plot(metrics, X_img, probes, appraisals, valence)
    return metrics


def _plot(metrics, X_img, probes, appraisals, valence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [a for a in appraisals if a in probes.names]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.2), squeeze=False)
    for ax, a in zip(axes[0], cols):
        pred = predict(X_img, probes.coef[probes.index(a)], probes.intercept[probes.index(a)])
        m = np.isfinite(valence)
        ax.scatter(pred[m], valence[m], s=8, alpha=0.35)
        c = metrics["image_readout"][a]["vs_valence"]
        sp = c["spearman"]
        ax.set_xlabel(f"frozen {a} read-out (text-trained)")
        ax.set_ylabel("EMOTIC valence (1–10)")
        ax.set_title(f"{a}\nSpearman={sp:+.3f} (n={c['n']})" if sp is not None else a)
    fig.suptitle(f"Stage C read-out transfer — L{metrics['layer']} {metrics['tap']} "
                 f"(EMOTIC test, n={metrics['n_images_scored']})")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_c_readout.png", dpi=130)
    fig.savefig(FIGURES_DIR / "stage_c_readout.pdf")  # vector twin for the paper
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C — cross-modal frozen-probe read-out")
    ap.add_argument("--config", default="config/stage_c.yaml")
    ap.add_argument("--n-images", type=int, default=None,
                    help="override config n_images; 0 (or --full) = whole test split")
    ap.add_argument("--full", action="store_true", help="score the entire test split (~7,280)")
    args = ap.parse_args()
    m = run(args.config, n_images_override=0 if args.full else args.n_images)
    print(f"\nStage C read-out — L{m['layer']} {m['tap']}  "
          f"(EMOTIC test: {m['n_images_scored']} scored, {m['n_skipped_unreadable']} skipped)\n")
    c = m["random_control"]
    print(f"  random-direction null (n={c.get('n_valid')}): "
          f"mean|rho|={_fmt(c.get('mean'))} p95={_fmt(c.get('p95'))} max={_fmt(c.get('max'))}\n")
    for a, r in m["image_readout"].items():
        v, auc = r["vs_valence"], r["polarity_auc"]
        ret = m["transfer"].get(a, {}).get("retention")
        print(f"  {a:16s} valence: spearman={_fmt(v['spearman'])} pearson={_fmt(v['pearson'])} "
              f"(n={v['n']})  |  polarity AUC={_fmt(auc['auc'])} (n={auc['n']})  |  "
              f"p_vs_ctrl={_fmt(r.get('vs_control_p'))}  |  retention={_fmt(ret)}")
    print(f"\n  VERDICT: {m['verdict']}")
    print(f"  metrics -> {STAGE_C_DIR/'metrics.json'}   figure -> {FIGURES_DIR/'stage_c_readout.png'}")


def _fmt(x):
    return f"{x:+.3f}" if isinstance(x, (int, float)) else "  n/a"


if __name__ == "__main__":
    main()
