"""Stage D on raw HuggingFace — cross-modal causal steering without the bridge.

WHY. The bridge corrupts Gemma's OUTPUT while leaving internal representations sound: `resid_post`
cosine vs raw HF stays >= 0.978 through L32 (probe site 0.980, which is why Stage C reproduces at
rho +0.510 vs the published +0.507), yet the final logits differ by 6.15 nats and flip the argmax,
and no-context image separation collapses to AUC 0.788 vs 0.982. Stage D injects at resid_post L18
(a site we now know is faithful) but SCORES BEHAVIOUR (P(pos) - P(neg) over the 13 emotion labels),
so its published slope +0.329 is read off the corrupted end of the model and has to be re-measured.

This is the causal capstone: Stage C shows the text-derived direction READS valence from images;
Stage D asks whether injecting it CHANGES what the model says. No captioning account explains a
causal effect, so the paper's shared-channel claim rests on this number.

SITE VERIFICATION, same discipline as the Stage C port. The directions are difference-of-means
vectors built from TEXT activations at resid_post L18 — and the text path is exact across stacks, so
a correct `resid_post` site must reproduce the published vector norms almost exactly:

    pleasantness 352.819   unpleasantness 358.377   suddenness 194.937

`--verify-dirs` builds the directions and checks them against those references before any image
sweep. A large mismatch means the hooked module is not the residual stream the probe/steering site
was defined on, and steering results would be meaningless — the Stage C port caught exactly this
class of error (its obvious `self_attn` default scored r^2 -6.26 where the real site scored +0.634).

    python -m src.experiments.stage_d_steering_hf --verify-dirs     # do this first
    python -m src.experiments.stage_d_steering_hf                   # full sweep
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.multimodal import IMAGE_EMOTION_PROMPT, TEXT_EMOTION_PROMPT
from ..data.crowd_envent import load_split as load_text_split, sample_tak_subset
from ..data.emotic import load_split as load_emotic_split
from ..paths import STAGE_A_DIR, STAGE_D_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .shared.hf_runtime import find_language_layers, load_gemma_hf
from .shared.readouts import closed_vocab_valence, first_content_token_ids
from .stage_a_steering_v2 import diff_of_means

# Published Δμ norms at resid_post L18 (results/stage_d/steering_metrics.json). Built from TEXT
# activations, and the text path is exact across stacks, so a correct site reproduces these.
REFERENCE_DMU_NORM = {"pleasantness": 352.8188, "unpleasantness": 358.3771, "suddenness": 194.9367}


# --------------------------------------------------------------------------- resid_post plumbing
@contextmanager
def resid_capture(model, layer: int):
    """Capture the LAST-token output of decoder `layer` — the raw-HF `hook_resid_post` equivalent."""
    layers = find_language_layers(model, verbose=False)
    store: list = [None]

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        store[0] = t[0, -1].detach().float().cpu().numpy()

    h = layers[layer].register_forward_hook(hook)
    try:
        yield store
    finally:
        h.remove()


@contextmanager
def steer(model, layer: int, z: torch.Tensor, beta: float):
    """Add `beta * z` to the residual stream at decoder `layer`, LAST position only.

    Mirrors `bridge.hooks.make_steer_hook`: pure, single-position, dtype-cast to the activation so
    bf16 stays bf16. The layer's output is cloned rather than modified in place, and the tuple shape
    is preserved for transformers versions whose decoder layers return `(hidden, ...)`.
    """
    layers = find_language_layers(model, verbose=False)

    def hook(_m, _i, out):
        tup = isinstance(out, tuple)
        h = (out[0] if tup else out).clone()
        h[:, -1, :] = h[:, -1, :] + beta * z.to(h.dtype)
        return (h,) + tuple(out[1:]) if tup else h

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def text_resid(model, processor, texts, layer: int, desc: str) -> np.ndarray:
    rows = []
    with resid_capture(model, layer) as store:
        for text in tqdm(texts, desc=desc):
            ids = processor.tokenizer(TEXT_EMOTION_PROMPT.format(text=text),
                                      return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                model(input_ids=ids)
            rows.append(store[0])
    return np.stack(rows)


def image_valence(model, enc, tok_ids) -> float:
    with torch.no_grad():
        out = model(**enc)
    return closed_vocab_valence(out.logits[0, -1].float(), tok_ids)


# --------------------------------------------------------------------------- directions
def build_directions(model, processor, layer: int, n_dir: int, seed: int, appraisals):
    """Text-derived Δμ per appraisal at resid_post `layer`, plus a norm-matched random control."""
    dtr = sample_tak_subset(load_text_split("train", seed=seed), seed=seed).head(n_dir)
    acts = text_resid(model, processor, dtr["text"].tolist(), layer, "build-dirs")
    dmu, norms = {}, {}
    for a in appraisals:
        v = diff_of_means(acts, dtr[a].to_numpy())
        if v is None:
            raise ValueError(f"not enough high/low examples for {a} — widen --n-dir")
        dmu[a] = v
        norms[a] = float(np.linalg.norm(v))
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(acts.shape[1]).astype(np.float32)
    dmu["_random"] = r / np.linalg.norm(r) * float(np.mean(list(norms.values())))
    norms["_random"] = float(np.linalg.norm(dmu["_random"]))
    return dmu, norms, len(dtr)


def check_norms(norms: dict) -> dict:
    """Compare Δμ norms against the published references — the site-verification gate."""
    print(f"\n  {'appraisal':16s} {'raw HF':>10s} {'published':>10s} {'rel err':>9s}")
    worst, rows = 0.0, {}
    for a, ref in REFERENCE_DMU_NORM.items():
        if a not in norms:
            continue
        got = norms[a]
        rel = abs(got - ref) / ref
        worst = max(worst, rel)
        rows[a] = {"raw_hf": got, "published": ref, "rel_err": rel}
        print(f"  {a:16s} {got:>10.3f} {ref:>10.3f} {rel:>8.2%}")
    ok = worst < 0.05
    print(f"\n  VERDICT: {'MATCH — resid_post site is correct (worst rel err %.2f%%)' % (worst*100) if ok else 'MISMATCH — worst rel err %.2f%%; the hooked module is probably not the residual stream the directions were defined on. Do NOT trust a steering sweep from this site.' % (worst*100)}")
    return {"ok": bool(ok), "worst_rel_err": float(worst), "per_appraisal": rows}


# --------------------------------------------------------------------------- run
def run(config_path: str, model_name: str, limit_override: int | None = None,
        verify_only: bool = False, force: bool = False) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    stage_a = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    layer = int(cfg.get("steering_layers", [int(stage_a.get("critical_layer", 18))])[0])
    betas = list(cfg.get("betas", [-3, -2, -1, 1, 2, 3]))
    n_dir = int(cfg.get("n_dir", 1200))
    n_images = limit_override or int(cfg.get("n_images", 150))
    appraisals = list(cfg.get("appraisals", ["pleasantness", "unpleasantness", "suddenness"]))
    seed = int(cfg.get("seed", 0))

    model, processor = load_gemma_hf(model_name)
    tok_ids = first_content_token_ids(processor)
    dmu, norms, n_used = build_directions(model, processor, layer, n_dir, seed, appraisals)
    check = check_norms(norms)
    if verify_only:
        out = {"run": run_stamp(), "git": git_hash(), "stack": "raw_huggingface", "layer": layer,
               "n_dir": n_used, "dmu_norm": norms, "site_check": check}
        save_json(out, STAGE_D_DIR / "dir_verification_hf.json")
        print(f"  data -> {STAGE_D_DIR/'dir_verification_hf.json'}")
        return out
    if not check["ok"] and not force:
        raise RuntimeError(
            f"Δμ norms do not match the published references (worst rel err {check['worst_rel_err']:.2%}). "
            f"The hooked module is probably not the residual stream the directions were defined on; a "
            f"steering sweep from the wrong site produces meaningless slopes. Pass --force to override.")

    dev = next(model.parameters()).device
    z = {d: torch.tensor(v, dtype=torch.float32, device=dev) for d, v in dmu.items()}

    idf = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    idf = idf.sample(n=min(n_images, len(idf)), random_state=seed).reset_index(drop=True)

    deltas = {d: {b: [] for b in betas} for d in z}
    base_vals, n_ok, n_skip = [], 0, 0
    for path in tqdm(idf["image_path"].tolist(), desc="steering images"):
        try:
            img = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        enc = processor(text=IMAGE_EMOTION_PROMPT, images=[img], return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}   # built once, reused for every arm
        base = image_valence(model, enc, tok_ids)
        base_vals.append(base)
        for d, vec in z.items():
            for b in betas:
                with steer(model, layer, vec, b):
                    deltas[d][b].append(image_valence(model, enc, tok_ids) - base)
        n_ok += 1

    mean_delta = {d: {b: float(np.mean(v)) for b, v in bs.items()} for d, bs in deltas.items()}
    slope = {d: float(np.polyfit(betas, [bs[b] for b in betas], 1)[0]) for d, bs in mean_delta.items()}
    metrics = {
        "run": run_stamp(), "git": git_hash(), "stack": "raw_huggingface", "model": model_name,
        "method": "cross_modal_diff_of_means_steering", "layer": layer, "betas": betas,
        "n_dir": n_used, "n_images": n_ok, "n_skipped": n_skip,
        "base_valence_mean": float(np.mean(base_vals)) if base_vals else None,
        "dmu_norm": norms, "site_check": check,
        "mean_delta_valence": mean_delta, "slope_vs_beta": slope,
        "supersedes": "results/stage_d/steering_metrics.json (bridge; behaviour read-out corrupted)",
        "published_slopes": {"pleasantness": 0.3293, "unpleasantness": -0.3087,
                             "suddenness": -0.0726, "_random": -0.0270},
    }
    save_json(metrics, STAGE_D_DIR / "steering_metrics_hf.json")
    _print(metrics)
    return metrics


def _print(m: dict) -> None:
    print(f"\nStage D [raw HF] — resid_post L{m['layer']}, {m['n_images']} images, "
          f"{m['n_dir']} direction examples")
    print(f"  base valence mean: {m['base_valence_mean']:+.4f}")
    pub = m["published_slopes"]
    print(f"\n  {'direction':16s} {'slope (raw HF)':>15s} {'published':>11s}")
    for d, s in m["slope_vs_beta"].items():
        print(f"  {d:16s} {s:>+15.4f} {pub.get(d, float('nan')):>+11.4f}")
    pl, rd = m["slope_vs_beta"].get("pleasantness"), m["slope_vs_beta"].get("_random")
    su = m["slope_vs_beta"].get("suddenness")
    if pl is not None and rd:
        print(f"\n  pleasantness vs random null: {abs(pl/rd):.1f}x   (published ~12x)")
    if pl is not None and su:
        print(f"  pleasantness vs suddenness control: {abs(pl/su):.1f}x   (published ~4.5x)")
    print("\n  READING: a slope near the published +0.33 means the causal claim survives the bridge")
    print("  bug; a slope collapsing toward the random null means it was an artefact of the")
    print("  corrupted behavioural read-out.")
    print(f"  metrics -> {STAGE_D_DIR/'steering_metrics_hf.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage D cross-modal steering on raw HuggingFace")
    ap.add_argument("--config", default="config/stage_d.yaml")
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--limit", type=int, default=None, help="number of EMOTIC images")
    ap.add_argument("--verify-dirs", action="store_true",
                    help="build directions and check Δμ norms against the published site (run first)")
    ap.add_argument("--force", action="store_true", help="sweep even if the site check fails")
    a = ap.parse_args()
    run(a.config, a.model, limit_override=a.limit, verify_only=a.verify_dirs, force=a.force)


if __name__ == "__main__":
    main()
