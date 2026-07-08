"""Stage A — steering test (the causal half of the go/no-go gate).

Loads the frozen unique-effect vectors (results/stage_a/probes.npz) and, at the critical
layer's residual stream, adds beta * z_a_unit at the last token. We measure the shift in a
closed-vocab emotion VALENCE score (P[positive emotions] - P[negative emotions]) vs beta,
and compare each appraisal direction to a norm-matched RANDOM direction (the control).

Theory-predicted signs: +pleasantness -> valence up; +unpleasantness -> valence down.
Pleasantness/unpleasantness are the cleanest test (strongest probes, unambiguous valence);
the other appraisals are reported but interpreted cautiously.

CAVEAT: z_a_unit is unit-norm; whether beta in {1,2,4} is large enough depends on the
residual-stream scale, which we print. If effects are flat, raise `steering_betas`.

Run on the A100 with HF_TOKEN set. Use `--limit` for a quick pipeline dry run.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import make_steer_hook, resid_post_name
from ..bridge.multimodal import TEXT_EMOTION_PROMPT
from ..data import EMOTION_LABELS
from ..data.crowd_envent import load_split
from ..paths import FIGURES_DIR, STAGE_A_DIR, ensure_dirs
from .common import load_config, load_probes, run_stamp, save_json

POSITIVE = ("joy", "pride", "relief", "trust")
NEGATIVE = ("anger", "boredom", "disgust", "fear", "guilt", "sadness", "shame")
# surprise / neutral are excluded from the valence score (ambiguous / neutral)


def emotion_token_ids(bridge) -> dict[str, int]:
    """First-subtoken id per emotion label (all verified single-token on Gemma)."""
    return {w: bridge.tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in EMOTION_LABELS}


def valence_score(logits_last, tok_ids) -> float:
    """Closed-vocab valence = P(positive labels) - P(negative labels) at the last token."""
    idx = torch.tensor([tok_ids[w] for w in EMOTION_LABELS], device=logits_last.device)
    probs = torch.softmax(logits_last[idx].float(), dim=-1)
    p = {w: probs[i].item() for i, w in enumerate(EMOTION_LABELS)}
    return sum(p[w] for w in POSITIVE) - sum(p[w] for w in NEGATIVE)


def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    betas = list(cfg.get("steering_betas", [-4, -2, -1, 1, 2, 4]))
    n_prompts = limit_override or int(cfg.get("steering_n_prompts", 60))
    device = cfg.get("device", "cuda")

    probes = load_probes(STAGE_A_DIR / "probes.npz")
    meta = load_config(STAGE_A_DIR / "metrics.json")
    crit = int(meta.get("critical_layer", 18))
    appraisals = cfg.get("steering_appraisals", probes.names)

    df = load_split("test")
    prompts = df["text"].head(n_prompts).tolist()

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=device)
    tok_ids = emotion_token_ids(bridge)
    hook_name = resid_post_name(crit)

    # norm-matched random control (unit norm, like the z_a_unit vectors)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    rand = rng.standard_normal(probes.z_unit.shape[1]).astype(np.float32)
    rand /= np.linalg.norm(rand)

    def z_tensor(vec):
        return torch.tensor(vec, dtype=torch.float32, device=next(bridge.parameters()).device)

    directions = {a: z_tensor(probes.steering_vector(a)) for a in appraisals}
    directions["_random"] = z_tensor(rand)

    # {direction: {beta: [per-prompt delta valence]}}
    deltas = {d: {b: [] for b in betas} for d in directions}
    resid_norms = []

    for text in tqdm(prompts, desc="steering"):
        input_ids = bridge.to_tokens(TEXT_EMOTION_PROMPT.format(text=text))
        base_logits = bridge.run_with_hooks(input_ids, fwd_hooks=[])
        base_val = valence_score(base_logits[0, -1], tok_ids)
        for d, z in directions.items():
            for b in betas:
                hook = make_steer_hook(z, b)
                logits = bridge.run_with_hooks(input_ids, fwd_hooks=[(hook_name, hook)])
                deltas[d][b].append(valence_score(logits[0, -1], tok_ids) - base_val)

    mean_delta = {d: {b: float(np.mean(v)) for b, v in bs.items()} for d, bs in deltas.items()}
    metrics = {"run": run_stamp(), "critical_layer": crit, "betas": betas,
               "n_prompts": len(prompts), "mean_delta_valence": mean_delta,
               "valence_groups": {"positive": POSITIVE, "negative": NEGATIVE}}

    save_json(metrics, STAGE_A_DIR / "steering_metrics.json")
    _plot(mean_delta, betas)
    return metrics


def _plot(mean_delta, betas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for d, bs in mean_delta.items():
        xs = sorted(bs)
        style = "k--" if d == "_random" else "-o"
        ax.plot(xs, [bs[b] for b in xs], style, ms=4, label=d.replace("_random", "random (control)"))
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("steering strength β"); ax.set_ylabel("Δ valence  (P[pos] − P[neg])")
    ax.set_title("Stage A steering: emotion-valence shift vs appraisal direction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_a_steering.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A — causal steering test")
    ap.add_argument("--config", default="config/stage_a.yaml")
    ap.add_argument("--limit", type=int, default=None, help="use only N test prompts (dry run)")
    args = ap.parse_args()
    m = run(args.config, limit_override=args.limit)
    print(f"\nSteering done (crit layer {m['critical_layer']}, {m['n_prompts']} prompts).")
    print(f"{'direction':22s} " + "  ".join(f"β={b:+d}" for b in m["betas"]))
    for d, bs in m["mean_delta_valence"].items():
        name = d.replace("_random", "random (control)")
        print(f"{name:22s} " + "  ".join(f"{bs[b]:+.3f}" for b in m["betas"]))
    print(f"\nfigure -> {FIGURES_DIR/'stage_a_steering.png'}")


if __name__ == "__main__":
    main()
