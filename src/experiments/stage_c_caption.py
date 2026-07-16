"""Stage C caption baseline + mechanism test — verbalization vs shared geometry (experiment-1.md).

The read-out transfer (stage_c_transfer.py) shows the frozen text pleasantness direction
tracks EMOTIC valence under image conditioning (rho~0.48). This script asks WHY, by
generating a NEUTRAL caption per image ("Describe this image in one sentence." — no emotion
words invited), and comparing two read-outs of the SAME frozen probe on the SAME images:

  - IMAGE read-out:   probe applied to image-conditioned activations (the direct path).
  - CAPTION read-out: probe applied to the caption run through the TEXT pipeline (verbal path).

Aggregate: |rho_cap| ~ |rho_img| => verbalization-mediated; |rho_cap| << |rho_img| => the
image path carries more. Mechanism (the rigorous version): the SEMIPARTIAL correlation — the
UNIQUE contribution of the image read-out to valence AFTER controlling for the caption
read-out. If it is ~0, transfer is fully verbalization-mediated; if it survives, the image
path adds signal beyond a neutral caption (but note: a lossy caption confounds this residual,
so a surviving semipartial is an UPPER BOUND on any non-verbal contribution, not proof of
shared geometry — analysis-rules).

Both read-outs are computed in ONE pass per image and persisted (caption_readout.parquet) so
richer-caption / Stage D re-analyses never re-run the expensive generation step.

Preconditions: results/stage_a/probes.npz. Generation is autoregressive and slow — run
`--preview 5` first. Run on the A100 with HF_TOKEN set and EMOTIC downloaded. Never re-fit.
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


def image_readout(bridge, image, layer, tap) -> np.ndarray:
    """Last-token activation at the frozen probe site under IMAGE conditioning (direct path)."""
    keep = keep_language_taps((tap,))
    name = f"blocks.{layer}.{tap}"
    inputs = build_image_inputs(bridge, image)  # default IMAGE_EMOTION_PROMPT
    with torch.no_grad():
        _, cache = bridge.run_with_cache(
            inputs["input_ids"], pixel_values=inputs["pixel_values"], names_filter=keep,
        )
    return cache[name][0, inputs["input_ids"].shape[-1] - 1].float().cpu().numpy()


def _semipartial(valence, cap_pred, img_pred) -> dict:
    """Unique (semipartial) contribution of the IMAGE read-out to valence beyond the CAPTION.

    Rank-based. part_r = (r_iv - r_cv*r_ic) / sqrt(1 - r_ic^2); the p-value is the t-test on
    adding the image rank to an OLS of valence rank on caption rank (same hypothesis). r_iv,
    r_cv, r_ic are Spearman correlations image/caption/inter-readout.
    """
    from scipy.stats import rankdata, spearmanr
    from scipy.stats import t as tdist

    v0, c0, i0 = (np.asarray(x, dtype=float) for x in (valence, cap_pred, img_pred))
    m = np.isfinite(v0) & np.isfinite(c0) & np.isfinite(i0)
    n = int(m.sum())
    if n < 10:
        return {"n": n, "part_r_image_unique": None, "p": None,
                "r_iv": None, "r_cv": None, "r_ic": None}
    r_iv = float(spearmanr(i0[m], v0[m])[0])
    r_cv = float(spearmanr(c0[m], v0[m])[0])
    r_ic = float(spearmanr(i0[m], c0[m])[0])
    part_r = (r_iv - r_cv * r_ic) / np.sqrt(max(1 - r_ic**2, 1e-9))

    def z(x):
        r = rankdata(x)
        return (r - r.mean()) / r.std()

    X = np.column_stack([np.ones(n), z(c0[m]), z(i0[m])])
    beta, *_ = np.linalg.lstsq(X, z(v0[m]), rcond=None)
    resid = z(v0[m]) - X @ beta
    dof = n - 3
    sigma2 = float(resid @ resid) / dof
    se_i = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[2, 2]))
    t_i = beta[2] / se_i if se_i > 0 else 0.0
    p = float(2 * tdist.sf(abs(t_i), dof))
    return {"n": n, "part_r_image_unique": float(part_r), "p": p,
            "r_iv": r_iv, "r_cv": r_cv, "r_ic": r_ic}


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
    max_new = int(cfg.get("caption_max_new_tokens", 64))

    df = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    if n_images and n_images < len(df):
        df = df.sample(n=int(n_images), random_state=seed).reset_index(drop=True)
    if preview:
        df = df.head(preview)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    dev = next(bridge.parameters()).device

    captions, cap_acts, img_acts, valid = [], [], [], []
    for i, path in enumerate(tqdm(df["image_path"].tolist(), desc="caption+readout")):
        try:
            image = Image.open(path).convert("RGB")
            cap = generate_caption(bridge, image, max_new, dev)
            cap_acts.append(text_readout(bridge, cap, layer, tap))
            img_acts.append(image_readout(bridge, image, layer, tap))
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
    X_cap, X_img = np.stack(cap_acts), np.stack(img_acts)
    dfv = df.loc[valid].reset_index(drop=True)
    valence = dfv["valence"].to_numpy(dtype=np.float64) if "valence" in dfv.columns else np.full(len(dfv), np.nan)

    metrics = {
        "run": run_stamp(), "layer": layer, "tap": tap, "seed": seed,
        "n_captioned": len(dfv), "n_skipped_unreadable": int((~valid).sum()),
        "caption_prompt": CAPTION_PROMPT.strip(), "max_new_tokens": max_new,
        "sample_captions": captions[:10], "readout": {},
    }
    persist = {"image_path": dfv["image_path"].to_numpy(), "caption": captions, "valence": valence}
    for a in appraisals:
        coef, inter = probes.coef[probes.index(a)], probes.intercept[probes.index(a)]
        cap_pred, img_pred = predict(X_cap, coef, inter), predict(X_img, coef, inter)
        persist[f"pred_caption_{a}"] = cap_pred
        persist[f"pred_image_{a}"] = img_pred
        metrics["readout"][a] = {
            "image": _corr(img_pred, valence),
            "caption": _corr(cap_pred, valence),
            "semipartial": _semipartial(valence, cap_pred, img_pred),
        }

    # Persist per-image read-outs + captions so mechanism re-analyses skip re-generation.
    import pandas as pd
    pd.DataFrame(persist).to_parquet(STAGE_C_DIR / "caption_readout.parquet")

    metrics["verdict"] = _mechanism_verdict(metrics["readout"].get("pleasantness"))
    save_json(metrics, STAGE_C_DIR / "caption_metrics.json")
    _plot(metrics)
    return metrics


def _mechanism_verdict(pleasantness) -> str:
    """Provisional mechanism verdict (human confirms), keyed on pleasantness."""
    if not pleasantness:
        return "inconclusive (pleasantness not scored)"
    sp = pleasantness["semipartial"]
    pr, p, r_iv, r_cv = sp["part_r_image_unique"], sp["p"], sp["r_iv"], sp["r_cv"]
    if pr is None or not r_iv:
        return "inconclusive (insufficient data for semipartial)"
    frac = abs(r_cv) / abs(r_iv)
    base = (f"caption reproduces ~{frac * 100:.0f}% of the image read-out "
            f"(r_cap={r_cv:+.2f} vs r_img={r_iv:+.2f}); ")
    if p is not None and p < 0.05 and abs(pr) >= 0.10:
        return (base + f"image retains a SIGNIFICANT unique contribution beyond a neutral "
                f"caption (semipartial r={pr:+.2f}, p={p:.3f}) — but caption lossiness confounds "
                f"this residual, so it is an upper bound, NOT proof of non-verbal shared geometry")
    return (base + f"NO significant unique image contribution beyond the caption "
            f"(semipartial r={pr:+.2f}, p={p:.3f}) — transfer is fully verbalization-mediated")


def _plot(metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = list(metrics["readout"])
    x = np.arange(len(ap))
    fig, ax = plt.subplots(figsize=(1.7 * len(ap) + 3, 4.3))
    ax.bar(x - 0.2, [abs(metrics["readout"][a]["image"]["spearman"]) for a in ap], 0.4, label="image read-out")
    ax.bar(x + 0.2, [abs(metrics["readout"][a]["caption"]["spearman"]) for a in ap], 0.4, label="neutral-caption read-out")
    for j, a in enumerate(ap):
        pr = metrics["readout"][a]["semipartial"]["part_r_image_unique"]
        if pr is not None:
            ax.annotate(f"unique={pr:+.2f}", (j, 0.02), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(ap)
    ax.set_ylabel("|Spearman| vs EMOTIC valence")
    ax.set_title(f"Stage C caption baseline + semipartial (n={metrics['n_captioned']})\n"
                 "image ~ caption -> verbalization; 'unique' = image contribution beyond caption")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_c_caption_baseline.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C caption baseline + semipartial mechanism test")
    ap.add_argument("--config", default="config/stage_c.yaml")
    ap.add_argument("--preview", type=int, default=None,
                    help="caption only N images and print them (sanity check before the full run)")
    args = ap.parse_args()
    m = run(args.config, preview=args.preview)
    if m.get("preview"):
        return
    print(f"\nStage C caption baseline — L{m['layer']} {m['tap']}  "
          f"({m['n_captioned']} captioned, {m['n_skipped_unreadable']} skipped)\n")
    for a, r in m["readout"].items():
        sp = r["semipartial"]
        print(f"  {a:16s} image rho={r['image']['spearman']:+.3f}  caption rho={r['caption']['spearman']:+.3f}  "
              f"|  image-caption r={sp['r_ic']:+.3f}  |  unique(image) r={sp['part_r_image_unique']:+.3f} "
              f"(p={sp['p']:.3f})")
    print(f"\n  VERDICT: {m['verdict']}")
    print(f"  metrics -> {STAGE_C_DIR/'caption_metrics.json'}   "
          f"data -> {STAGE_C_DIR/'caption_readout.parquet'}   "
          f"figure -> {FIGURES_DIR/'stage_c_caption_baseline.png'}")


if __name__ == "__main__":
    main()
