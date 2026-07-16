"""Stage D — cross-modal steering (the causal capstone). See docs/experiment-1.md.

Stage C showed the frozen text appraisal direction READS valence from image activations
beyond captions — but that is correlational. Stage D asks the causal question: does the
TEXT-derived appraisal direction, injected UNDER IMAGE INPUT, *shift* the model's emotion
output the way appraisal theory predicts? A causal effect cannot be explained away by "the
model is just captioning" — it is the strongest evidence for a shared appraisal handle.

Recipe (reuses the Stage A v2 finding: the read-out direction does NOT steer; the
difference-of-means direction does):
  1. Build TEXT-derived Δμ_a = mean(act | rating high) − mean(act | rating low) at the
     Stage A critical layer's residual stream (resid_post L18), from crowd-enVENT train.
     These are frozen, text-learned directions — the cross-modal transfer is text→image.
  2. For each EMOTIC test image: measure the closed-vocab emotion VALENCE score
     (P[positive] − P[negative]) with no steering, then with β·Δμ_a added at resid_post L18,
     last token, UNDER IMAGE CONDITIONING (pixel_values). Δ = steered − base.
  3. Compare against a norm-matched RANDOM direction (null) and, for specificity, a
     non-valence appraisal (suddenness) that should NOT move valence.

Theory-predicted signs: +pleasantness → valence up; +unpleasantness → valence down;
random/suddenness ≈ flat. No autoregressive generation — one forward per (image, dir, β),
under torch.no_grad() (the SigLIP tower's 4096-patch attention OOMs a 40 GB A100 otherwise).

Run on the A100 with HF_TOKEN set and EMOTIC downloaded. `--limit` shrinks the image sweep.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import make_steer_hook, resid_post_name
from ..bridge.multimodal import build_image_inputs
from ..data.crowd_envent import load_split as load_text_split, sample_tak_subset
from ..data.emotic import load_split as load_emotic_split
from ..paths import FIGURES_DIR, STAGE_A_DIR, STAGE_D_DIR, ensure_dirs
from .common import load_config, run_stamp, save_json
from .stage_a_steering import emotion_token_ids, valence_score
from .stage_a_steering_v2 import diff_of_means, extract_resid


def _image_valence(bridge, ids, pixel_values, hooks, tok_ids) -> float:
    """Emotion valence score at the last token under image conditioning (+ optional steer)."""
    with torch.no_grad():
        logits = bridge.run_with_hooks(ids, pixel_values=pixel_values, fwd_hooks=hooks)
    return valence_score(logits[0, -1], tok_ids)


def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    stage_a = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    layers = list(cfg.get("steering_layers", [int(stage_a.get("critical_layer", 18))]))
    betas = list(cfg.get("betas", [-3, -2, -1, 1, 2, 3]))
    n_dir = int(cfg.get("n_dir", 1200))
    n_images = limit_override or int(cfg.get("n_images", 150))
    appraisals = list(cfg.get("appraisals", ["pleasantness", "unpleasantness", "suddenness"]))
    seed = int(cfg.get("seed", 0))

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    dev = next(bridge.parameters()).device

    # --- TEXT-derived difference-of-means directions at resid_post (frozen, text-learned) ---
    dtr = sample_tak_subset(load_text_split("train", seed=seed), seed=seed).head(n_dir)
    tr_acts = extract_resid(bridge, dtr["text"].tolist(), layers, "build-dirs")
    dmu = {a: {} for a in appraisals}
    for a in appraisals:
        for l in layers:
            v = diff_of_means(tr_acts[l], dtr[a].to_numpy())
            if v is None:
                raise ValueError(f"not enough high/low examples for {a} — widen n_dir")
            dmu[a][l] = v
    # norm-matched random control: per layer, scaled to the appraisals' mean ‖Δμ‖
    rng = np.random.default_rng(seed)
    rand = {}
    for l in layers:
        ref = float(np.mean([np.linalg.norm(dmu[a][l]) for a in appraisals]))
        r = rng.standard_normal(tr_acts[l].shape[1]).astype(np.float32)
        rand[l] = r / np.linalg.norm(r) * ref
    directions = {**{a: dmu[a] for a in appraisals}, "_random": rand}
    scaled = {d: {l: torch.tensor(v, dtype=torch.float32, device=dev) for l, v in per.items()}
              for d, per in directions.items()}

    # --- steer image-conditioned forwards, measure valence shift ---
    idf = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    idf = idf.sample(n=min(n_images, len(idf)), random_state=seed).reset_index(drop=True)

    deltas = {d: {b: [] for b in betas} for d in directions}
    base_vals, n_ok, n_skip = [], 0, 0
    for path in tqdm(idf["image_path"].tolist(), desc="steering images"):
        try:
            inputs = build_image_inputs(bridge, Image.open(path).convert("RGB"))
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        ids, pv = inputs["input_ids"], inputs["pixel_values"]
        base = _image_valence(bridge, ids, pv, [], tok_ids)
        base_vals.append(base)
        for d, per in scaled.items():
            for b in betas:
                hooks = [(resid_post_name(l), make_steer_hook(per[l], b)) for l in layers]
                deltas[d][b].append(_image_valence(bridge, ids, pv, hooks, tok_ids) - base)
        n_ok += 1

    mean_delta = {d: {b: float(np.mean(v)) for b, v in bs.items()} for d, bs in deltas.items()}
    slope = {d: float(np.polyfit(betas, [bs[b] for b in betas], 1)[0]) for d, bs in mean_delta.items()}
    dmu_norm = {a: {l: float(np.linalg.norm(dmu[a][l])) for l in layers} for a in appraisals}
    metrics = {
        "run": run_stamp(), "method": "cross_modal_diff_of_means_steering",
        "layers": layers, "betas": betas, "n_dir": len(dtr), "n_images": n_ok,
        "n_skipped": n_skip, "base_valence_mean": float(np.mean(base_vals)) if base_vals else None,
        "dmu_norm": dmu_norm, "mean_delta_valence": mean_delta, "slope_vs_beta": slope,
        "verdict": _verdict(slope, appraisals),
    }
    save_json(metrics, STAGE_D_DIR / "steering_metrics.json")
    _plot(mean_delta, betas, layers)
    return metrics


def _verdict(slope, appraisals) -> str:
    """Provisional causal verdict (human confirms): theory-predicted signs + beats the null."""
    r = abs(slope.get("_random", 0.0))
    pl, un = slope.get("pleasantness"), slope.get("unpleasantness")
    if pl is None:
        return "inconclusive (pleasantness not steered)"
    thr = max(3 * r, 0.005)

    def beats(s):
        return s is not None and abs(s) > thr

    sud = slope.get("suddenness")
    spec = "" if sud is None else (f" specificity ok (suddenness {sud:+.3f} ~ flat)" if abs(sud) <= thr
                                   else f" CAUTION: suddenness also moves ({sud:+.3f})")
    if pl > 0 and beats(pl) and (un is None or (un < 0 and beats(un))):
        return (f"supports cross-modal CAUSAL transfer: text-derived +pleasantness raises image "
                f"valence (slope {pl:+.3f}), +unpleasantness lowers it ({un:+.3f}), both beat the "
                f"random null ({slope.get('_random', 0):+.3f}).{spec}")
    if beats(pl) or beats(un):
        return (f"partial: some direction beats the null (pleasantness {pl:+.3f}, unpleasantness "
                f"{un}) but signs/magnitudes are mixed — try a multi-layer band or larger β.{spec}")
    return (f"fails to support causal transfer: appraisal directions ~ random null "
            f"(pleasantness {pl:+.3f}, random {slope.get('_random', 0):+.3f}) — try a layer band / larger β")


def _plot(mean_delta, betas, layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for d, bs in mean_delta.items():
        xs = sorted(bs)
        style = "k--" if d == "_random" else "-o"
        ax.plot(xs, [bs[b] for b in xs], style, ms=4, label=d.replace("_random", "random (control)"))
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("β  (multiples of the low→high Δμ shift)")
    ax.set_ylabel("Δ valence  (P[pos] − P[neg])")
    ax.set_title(f"Stage D cross-modal steering under image input (layers {layers})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_d_steering.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage D — cross-modal steering under image input")
    ap.add_argument("--config", default="config/stage_d.yaml")
    ap.add_argument("--limit", type=int, default=None, help="use only N images (dry run)")
    args = ap.parse_args()
    m = run(args.config, limit_override=args.limit)
    print(f"\nStage D steering — layers {m['layers']}, {m['n_dir']} dir examples, "
          f"{m['n_images']} images ({m['n_skipped']} skipped), base valence "
          f"{m['base_valence_mean']:+.3f}.")
    print("β = multiples of the low→high Δμ shift (text-derived, injected under image input).\n")
    print(f"{'direction':22s} " + "  ".join(f"β={b:+d}" for b in m["betas"]) + f"  {'slope':>8s}")
    for d, bs in m["mean_delta_valence"].items():
        name = d.replace("_random", "random (control)")
        print(f"{name:22s} " + "  ".join(f"{bs[b]:+.3f}" for b in m["betas"])
              + f"  {m['slope_vs_beta'][d]:+8.3f}")
    print(f"\n  VERDICT: {m['verdict']}")
    print(f"  figure -> {FIGURES_DIR/'stage_d_steering.png'}")


if __name__ == "__main__":
    main()
