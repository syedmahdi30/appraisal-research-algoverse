"""Stage C on raw HuggingFace — re-run of the cross-modal frozen-probe read-out without the bridge.

WHY. `diagnose_image_pathway` showed TransformerBridge and raw HF compute DIFFERENT forwards for
Gemma-3 on byte-identical inputs: the text path agrees (behavioural valence |Δ| = 0.0011) but the
image path does not (0/5 argmax agreement, valence |Δ| up to 0.80), and the bridge's no-context image
separation is degraded (AUC 0.788 vs 0.982 on raw HF). Every image-conditioned bridge result is
therefore untrustworthy, including the published Stage C transfer (rho = +0.507). This module
re-measures it on the reference implementation.

This is the GATING experiment for the paper: if the text-trained probe no longer reads valence from
image activations, the shared-channel foundation that Stages D/E/F build on is gone.

THE TAP PROBLEM, and how this module avoids guessing. The probe was fit on activations the bridge
called `blocks.18.hook_attn_out`. Raw HF has no such name, and several modules are plausible
candidates (`self_attn` output, its `o_proj` output, the `post_attention_layernorm` output — Gemma
applies a post-attention norm, so these genuinely differ). Reading the probe off the WRONG module
would produce a near-zero correlation and look exactly like "transfer failed" — a catastrophic false
negative on the one experiment that decides the paper's fate.

So `--verify-tap` identifies the right module empirically and WITHOUT the bridge: the frozen probe is
known to reach validation r^2 = 0.641 on crowd-enVENT TEXT at this site (Stage A, and the text path
is exact across stacks). Extract text activations from each candidate module, score the probe, and
the candidate that reproduces ~0.64 is the correct tap. A candidate scoring ~0 is the wrong module.
Run that first; the main pass refuses to run on an unverified tap unless forced.

    python -m src.experiments.stage_c_transfer_hf --verify-tap      # do this first
    python -m src.experiments.stage_c_transfer_hf --full             # verified tap is the default

VERIFIED RESULT (Gemma-3-4B, 2026-08-22). The correct module is `post_attention_layernorm`, NOT the
plausible-looking `self_attn`: probe r^2 = +0.634 (Stage A reference 0.641) vs **-6.26** for
`self_attn`/`o_proj`, whose read-out correlates with the target at rho = +0.04. Gemma applies the
post-attention norm before the residual add and TransformerLens folds it into the attn-block output.
Running the images through `self_attn` would have returned near-zero correlation and read as
"cross-modal transfer failed" — the false negative this verification exists to prevent.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.multimodal import IMAGE_EMOTION_PROMPT, TEXT_EMOTION_PROMPT
from ..data.crowd_envent import load_split as load_text_split
from ..data.emotic import load_split as load_emotic_split
from ..paths import FIGURES_DIR, STAGE_A_DIR, STAGE_C_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .shared.hf_runtime import find_language_layers, last_token_tap, load_gemma_hf
from .shared.reporting import (
    correlation,
    polarity_auc,
    polarity_vector,
    random_direction_controls,
    shared_emotic_label,
    transfer_verdict,
)

# Candidate raw-HF modules for the bridge's `hook_attn_out`, relative to a decoder layer.
# Gemma applies `post_attention_layernorm` to the attention output BEFORE the residual add, so these
# are genuinely different tensors and only one matches what the probe was fit on.
# Verified on Gemma-3-4B: `post_attention_layernorm` is the match (r^2 +0.634 vs Stage A 0.641);
# `self_attn` and its `o_proj` return the SAME pre-norm tensor and score r^2 -6.26.
CANDIDATE_TAPS = ("self_attn", "self_attn.o_proj", "post_attention_layernorm")
DEFAULT_MODEL = "google/gemma-3-4b-it"


# --------------------------------------------------------------------------- raw-HF compatibility façades
find_lm_layers = find_language_layers


def load_hf(model_name: str = DEFAULT_MODEL):
    return load_gemma_hf(model_name)


# --------------------------------------------------------------------------- activation extraction
def text_activations(model, processor, texts, layer: int, tap: str) -> np.ndarray:
    rows = []
    with last_token_tap(model, layer, tap) as store:
        for text in tqdm(texts, desc=f"text acts [{tap}]"):
            ids = processor.tokenizer(TEXT_EMOTION_PROMPT.format(text=text),
                                      return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                model(input_ids=ids)
            rows.append(store[0])
    return np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)


def image_activations(model, processor, image_paths, layer: int, tap: str):
    """(X [m, d], valid_mask [n]) last-token activations under image conditioning.

    Unreadable images are skipped rather than killing a long run; the count is surfaced in metrics.
    """
    rows, valid = [], []
    with last_token_tap(model, layer, tap) as store:
        for path in tqdm(image_paths, desc=f"image acts [{tap}]"):
            try:
                img = Image.open(path).convert("RGB")
            except (FileNotFoundError, OSError):
                valid.append(False)
                continue
            enc = processor(text=IMAGE_EMOTION_PROMPT, images=[img], return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                model(**enc)
            rows.append(store[0])
            valid.append(True)
    X = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
    return X, np.array(valid, dtype=bool)


# --------------------------------------------------------------------------- tap verification
def _r2(pred, y) -> float:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    ss_res = float(((y[m] - pred[m]) ** 2).sum())
    ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def verify_tap(model_name: str, layer: int, n_text: int, seed: int,
               appraisal: str = "pleasantness") -> dict:
    """Find which raw-HF module the frozen probe was fit on, using the probe itself as the oracle.

    No bridge required: Stage A reports validation r^2 = 0.641 for pleasantness at this site, and the
    text forward is exact across stacks, so the correct module is the one that reproduces it. A
    candidate scoring near 0 (or negative) is simply the wrong tensor.
    """
    probes = load_probes(STAGE_A_DIR / "probes.npz")
    idx = probes.index(appraisal)
    coef, inter = probes.coef[idx], probes.intercept[idx]

    tdf = load_text_split("val", seed=seed)
    if n_text < len(tdf):
        tdf = tdf.sample(n=n_text, random_state=seed).reset_index(drop=True)
    y = tdf[appraisal].to_numpy(dtype=float)

    model, processor = load_gemma_hf(model_name)
    stage_a = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    target_r2 = None
    for row in (load_config(STAGE_A_DIR / "summary.json").get("rows", [])
                if (STAGE_A_DIR / "summary.json").exists() else []):
        if row.get("appraisal") == appraisal:
            target_r2 = row.get("val_r2")

    print(f"\n=== tap verification (probe={appraisal}, layer={layer}, n={len(tdf)}) ===")
    if target_r2:
        print(f"  Stage A reference validation r^2 = {target_r2:.3f} — the correct tap reproduces this")

    results = {}
    for tap in CANDIDATE_TAPS:
        try:
            X = text_activations(model, processor, tdf["text"].tolist(), layer, tap)
            r2 = _r2(predict(X, coef, inter), y)
            c = correlation(predict(X, coef, inter), y)
            results[tap] = {"r2": r2, "spearman": c["spearman"], "d_model": int(X.shape[1])}
            print(f"  {tap:28s} r^2 = {r2:+.4f}   spearman = {c['spearman']:+.4f}   d={X.shape[1]}")
        except Exception as e:
            results[tap] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {tap:28s} FAILED: {type(e).__name__}: {e}")

    ok = {k: v for k, v in results.items() if isinstance(v.get("r2"), float) and np.isfinite(v["r2"])}
    best = max(ok, key=lambda k: ok[k]["r2"]) if ok else None
    verdict = "no candidate produced a finite r^2"
    if best is not None:
        br2 = ok[best]["r2"]
        if target_r2 and br2 >= 0.8 * target_r2:
            verdict = (f"MATCH — '{best}' reproduces the Stage A probe (r^2 {br2:.3f} vs reference "
                       f"{target_r2:.3f}); use --tap {best}")
        elif br2 > 0.2:
            verdict = (f"PARTIAL — best is '{best}' at r^2 {br2:.3f}, below the Stage A reference "
                       f"{target_r2 if target_r2 else float('nan'):.3f}. The probe may need refitting "
                       f"on raw-HF text activations before Stage C is interpretable.")
        else:
            verdict = (f"NO MATCH — best candidate '{best}' scores r^2 {br2:.3f}. None of these "
                       f"modules is the site the probe was fit on; do not run Stage C until resolved.")
    print(f"\n  VERDICT: {verdict}")
    out = {"run": run_stamp(), "git": git_hash(), "model": model_name, "layer": layer,
           "appraisal": appraisal, "n_text": int(len(tdf)), "stage_a_reference_r2": target_r2,
           "candidates": results, "best_tap": best, "verdict": verdict}
    save_json(out, STAGE_C_DIR / "tap_verification_hf.json")
    print(f"  data -> {STAGE_C_DIR/'tap_verification_hf.json'}")
    return out


# --------------------------------------------------------------------------- main run
def run(config_path: str, tap: str, model_name: str, n_images_override: int | None = None,
        force: bool = False) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()

    ver_path = STAGE_C_DIR / "tap_verification_hf.json"
    if not force:
        if not ver_path.exists():
            raise RuntimeError(
                "no tap verification found — run `--verify-tap` first so the probe is read off the "
                "module it was fit on. Reading the wrong module yields a near-zero correlation that "
                "is indistinguishable from a genuine transfer failure. Use --force to override.")
        v = load_config(ver_path)
        if v.get("best_tap") != tap or not str(v.get("verdict", "")).startswith("MATCH"):
            raise RuntimeError(
                f"tap verification does not endorse --tap {tap} (best={v.get('best_tap')}, "
                f"verdict={v.get('verdict')}). Re-verify or pass --force.")

    probes = load_probes(STAGE_A_DIR / "probes.npz")
    stage_a = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    layer = int(cfg.get("critical_layer", stage_a.get("critical_layer", 18)))
    seed = int(cfg.get("seed", 0))
    n_images = n_images_override if n_images_override is not None else cfg.get("n_images")
    if n_images is not None and n_images <= 0:
        n_images = None
    appraisals = [a for a in cfg.get("appraisals", ["pleasantness", "unpleasantness"])
                  if a in probes.names]
    positive = cfg.get("positive_labels", ["joy"])
    negative = cfg.get("negative_labels", ["anger", "disgust", "fear", "sadness"])
    n_random = int(cfg.get("n_random", 100))

    df = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    if n_images and n_images < len(df):
        df = df.sample(n=int(n_images), random_state=seed).reset_index(drop=True)

    model, processor = load_gemma_hf(model_name)
    X_img, valid = image_activations(model, processor, df["image_path"].tolist(), layer, tap)
    n_skipped = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)

    valence = (df["valence"].to_numpy(dtype=np.float64) if "valence" in df.columns
               else np.full(len(df), np.nan))
    shared = [shared_emotic_label(c) for c in df["categories"]]
    polarity = polarity_vector(shared, positive, negative)
    n_single = int(sum(s is not None for s in shared))

    metrics = {
        "run": run_stamp(), "git": git_hash(), "stack": "raw_huggingface",
        "model": model_name, "layer": layer, "tap": tap, "seed": seed,
        "n_images_scored": len(df), "n_skipped_unreadable": n_skipped,
        "n_single_label": n_single, "n_dropped_multilabel": len(df) - n_single,
        "polarity_groups": {"positive": positive, "negative": negative},
        "supersedes": "results/stage_c/metrics.json (TransformerBridge; image path shown unreliable)",
        "image_readout": {}, "random_control": {}, "text_reference": {}, "transfer": {},
    }

    ctrl = random_direction_controls(X_img, valence, n_random, seed)
    ctrl_abs = np.asarray(ctrl.pop("_abs"), dtype=float)
    metrics["random_control"] = ctrl

    for a in appraisals:
        coef, inter = probes.coef[probes.index(a)], probes.intercept[probes.index(a)]
        pred = predict(X_img, coef, inter)
        vc = correlation(pred, valence)
        pval = None
        if vc["spearman"] is not None and ctrl_abs.size:
            pval = float((1 + int(np.sum(ctrl_abs >= abs(vc["spearman"])))) / (ctrl_abs.size + 1))
        metrics["image_readout"][a] = {"vs_valence": vc, "polarity_auc": polarity_auc(pred, polarity),
                                       "vs_control_p": pval}

    if cfg.get("text_reference", True):
        n_text = int(cfg.get("n_text", 1000))
        tdf = load_text_split("test", seed=seed)
        if n_text < len(tdf):
            tdf = tdf.sample(n=n_text, random_state=seed).reset_index(drop=True)
        X_txt = text_activations(model, processor, tdf["text"].tolist(), layer, tap)
        for a in appraisals:
            if a not in tdf.columns:
                continue
            coef, inter = probes.coef[probes.index(a)], probes.intercept[probes.index(a)]
            tc = correlation(predict(X_txt, coef, inter), tdf[a].to_numpy(dtype=np.float64))
            metrics["text_reference"][a] = tc
            img_sp = metrics["image_readout"][a]["vs_valence"]["spearman"]
            if tc["spearman"] and img_sp is not None:
                metrics["transfer"][a] = {"text_abs_spearman": abs(tc["spearman"]),
                                          "image_abs_spearman": abs(img_sp),
                                          "retention": abs(img_sp) / abs(tc["spearman"])}

    metrics["verdict"] = transfer_verdict(metrics, appraisals)
    save_json(metrics, STAGE_C_DIR / "metrics_hf.json")
    _print(metrics)
    return metrics


def _fmt(x):
    return f"{x:+.3f}" if isinstance(x, (int, float)) else "  n/a"


def _print(m: dict) -> None:
    print(f"\nStage C [raw HF] — L{m['layer']} {m['tap']}  "
          f"({m['n_images_scored']} scored, {m['n_skipped_unreadable']} skipped)")
    c = m["random_control"]
    print(f"  random-direction null (n={c.get('n_valid')}): mean|rho|={_fmt(c.get('mean'))} "
          f"p95={_fmt(c.get('p95'))} max={_fmt(c.get('max'))}")
    for a, r in m["image_readout"].items():
        v, auc = r["vs_valence"], r["polarity_auc"]
        ret = m["transfer"].get(a, {}).get("retention")
        print(f"  {a:16s} valence spearman={_fmt(v['spearman'])} (n={v['n']})  |  "
              f"AUC={_fmt(auc['auc'])}  |  p_vs_ctrl={_fmt(r.get('vs_control_p'))}  |  "
              f"retention={_fmt(ret)}")
    print(f"\n  VERDICT: {m['verdict']}")
    print("  COMPARE to the published bridge run (rho=+0.507, AUC 0.898): a materially lower value "
          "here means the published Stage C was an artefact of the bridge's image path.")
    print(f"  metrics -> {STAGE_C_DIR/'metrics_hf.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C cross-modal read-out on raw HuggingFace")
    ap.add_argument("--config", default="config/stage_c.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tap", default="post_attention_layernorm", choices=CANDIDATE_TAPS)
    ap.add_argument("--verify-tap", action="store_true",
                    help="identify the correct module by reproducing the Stage A probe r^2 (run first)")
    ap.add_argument("--n-text-verify", type=int, default=300)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--full", action="store_true", help="score the entire test split (~7,280)")
    ap.add_argument("--force", action="store_true", help="run without an endorsing tap verification")
    a = ap.parse_args()
    if a.verify_tap:
        verify_tap(a.model, a.layer, a.n_text_verify, a.seed)
    else:
        run(a.config, a.tap, a.model,
            n_images_override=0 if a.full else a.n_images, force=a.force)


if __name__ == "__main__":
    main()
