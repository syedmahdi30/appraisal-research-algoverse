"""Stage A steering v2 — contrastive (difference-of-means) directions at natural scale.

v1 (probe unique-effect vectors scaled as a fraction of the residual norm) never beat a
norm-matched random control. Two known reasons, two fixes:
  1. DIRECTION: probe/ridge weights are optimized to READ a feature, not MOVE it. v2 uses
     Δμ_a = mean(act | rating high) − mean(act | rating low) — the appraisal's actual
     footprint in activation space (ActAdd / CAA style), which steers better.
  2. SCALE: steer by β·Δμ_a in NATURAL units (β=1 == one full low→high appraisal shift),
     so there is no arbitrary magnitude to tune.
  Optional MULTI-LAYER injection (a band of layers) so the nudge isn't washed out by one
  RMSNorm. Control = random direction matched to ‖Δμ_a‖ per layer. Readout = emotion valence.

Run on the A100 with HF_TOKEN set. `--limit` shrinks both direction-building and the sweep.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import make_steer_hook, resid_post_name
from ..bridge.multimodal import TEXT_EMOTION_PROMPT
from ..data.crowd_envent import load_split, sample_tak_subset
from ..paths import FIGURES_DIR, STAGE_A_DIR, ensure_dirs
from .common import load_config, run_stamp, save_json
from .stage_a_steering import emotion_token_ids, valence_score


def extract_resid(bridge, texts, layers, desc):
    """Return {layer: [n, d]} last-token residual-stream activations at each layer."""
    names = {resid_post_name(l) for l in layers}
    store = {l: [] for l in layers}
    for text in tqdm(texts, desc=desc):
        ids = bridge.to_tokens(TEXT_EMOTION_PROMPT.format(text=text))
        _, cache = bridge.run_with_cache(ids, names_filter=lambda n: n in names)
        last = ids.shape[-1] - 1
        for l in layers:
            store[l].append(cache[resid_post_name(l)][0, last].float().cpu().numpy())
    return {l: np.stack(v) for l, v in store.items()}


def diff_of_means(acts, ratings, hi=4, lo=2):
    """Δμ = mean(act | rating>=hi) − mean(act | rating<=lo). Natural-scale steering vector."""
    r = np.asarray(ratings)
    if (r >= hi).sum() < 5 or (r <= lo).sum() < 5:
        return None
    return acts[r >= hi].mean(0) - acts[r <= lo].mean(0)


def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layers = list(cfg.get("steering_v2_layers", [18]))
    betas = list(cfg.get("steering_v2_betas", [-3, -2, -1, 1, 2, 3]))
    n_dir = limit_override or int(cfg.get("steering_v2_n_dir", 1200))
    n_prompts = (limit_override // 2 or 1) if limit_override else int(cfg.get("steering_v2_n_prompts", 120))
    appraisals = cfg.get("steering_appraisals", ["pleasantness", "unpleasantness"])
    seed = int(cfg.get("seed", 0))

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    dev = next(bridge.parameters()).device

    # --- build directions from train activations ---
    dtr = sample_tak_subset(load_split("train", seed=seed), seed=seed).head(n_dir)
    tr_acts = extract_resid(bridge, dtr["text"].tolist(), layers, "build-dirs")
    dmu = {a: {} for a in appraisals}
    for a in appraisals:
        for l in layers:
            v = diff_of_means(tr_acts[l], dtr[a].to_numpy())
            if v is None:
                raise ValueError(f"not enough high/low examples for {a} — widen n_dir")
            dmu[a][l] = v
    # norm-matched random control: random dir per layer, scaled to the appraisals' mean ‖Δμ‖
    rng = np.random.default_rng(seed)
    rand = {}
    for l in layers:
        ref = np.mean([np.linalg.norm(dmu[a][l]) for a in appraisals])
        r = rng.standard_normal(tr_acts[l].shape[1]).astype(np.float32)
        rand[l] = (r / np.linalg.norm(r) * ref)
    directions = {**{a: dmu[a] for a in appraisals}, "_random": rand}

    def tens(vec):
        return torch.tensor(vec, dtype=torch.float32, device=dev)

    scaled = {d: {l: tens(v) for l, v in per.items()} for d, per in directions.items()}

    # --- steer test prompts, measure valence shift ---
    prompts = load_split("test", seed=seed)["text"].head(n_prompts).tolist()
    token_ids = [bridge.to_tokens(TEXT_EMOTION_PROMPT.format(text=t)) for t in prompts]
    base_vals = [valence_score(bridge.run_with_hooks(ids, fwd_hooks=[])[0, -1], tok_ids) for ids in token_ids]

    deltas = {d: {b: [] for b in betas} for d in directions}
    for i, ids in enumerate(tqdm(token_ids, desc="steering v2")):
        for d, per in scaled.items():
            for b in betas:
                hooks = [(resid_post_name(l), make_steer_hook(per[l], b)) for l in layers]
                logits = bridge.run_with_hooks(ids, fwd_hooks=hooks)
                deltas[d][b].append(valence_score(logits[0, -1], tok_ids) - base_vals[i])

    mean_delta = {d: {b: float(np.mean(v)) for b, v in bs.items()} for d, bs in deltas.items()}
    slope = {d: float(np.polyfit(betas, [bs[b] for b in betas], 1)[0]) for d, bs in mean_delta.items()}
    dmu_norm = {a: {l: float(np.linalg.norm(dmu[a][l])) for l in layers} for a in appraisals}
    metrics = {"run": run_stamp(), "method": "diff_of_means_natural_scale", "layers": layers,
               "betas": betas, "n_dir": len(dtr), "n_prompts": len(prompts),
               "dmu_norm": dmu_norm, "mean_delta_valence": mean_delta, "slope_vs_beta": slope}

    save_json(metrics, STAGE_A_DIR / "steering_v2_metrics.json")
    _plot(mean_delta, betas, layers)
    return metrics


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
    ax.set_title(f"Stage A steering v2 (diff-of-means, layers {layers})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_a_steering_v2.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A steering v2 — diff-of-means directions")
    ap.add_argument("--config", default="config/stage_a.yaml")
    ap.add_argument("--limit", type=int, default=None, help="shrink direction-build + sweep (dry run)")
    args = ap.parse_args()
    m = run(args.config, limit_override=args.limit)
    print(f"\nSteering v2 done (layers {m['layers']}, {m['n_dir']} dir examples, {m['n_prompts']} prompts).")
    print("β = multiples of the low→high Δμ shift (natural units).\n")
    print(f"{'direction':22s} " + "  ".join(f"β={b:+d}" for b in m["betas"]) + f"  {'slope':>8s}")
    for d, bs in m["mean_delta_valence"].items():
        name = d.replace("_random", "random (control)")
        print(f"{name:22s} " + "  ".join(f"{bs[b]:+.3f}" for b in m["betas"])
              + f"  {m['slope_vs_beta'][d]:+8.3f}")
    print(f"\nfigure -> {FIGURES_DIR/'stage_a_steering_v2.png'}")


if __name__ == "__main__":
    main()
