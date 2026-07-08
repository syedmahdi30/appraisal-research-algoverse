"""Stage A — text-only appraisal replication gate (docs/experiment-1.md).

This run does the PROBE-LOCALIZATION half of the gate:
  1. boot Gemma, verify label tokenization (closed-vocab sanity)
  2. cache last-token LM activations for all 3 taps in ONE forward pass per example,
     for the train and val splits
  3. per tap/appraisal, fit ridge on TRAIN, score r2 on VAL, sweep layers
     (expect the Tak-style mid-layer, MHSA-dominant peak); report a shuffled-label baseline
  4. pick the critical layer on val (MHSA tap), fit frozen probes there on train
  5. write results/stage_a/{metrics.json, probes.npz}

NOT YET in this run (next iteration, tracked in docs/experiment-1.md): closed-vocab
correctness filtering, and the steering test at beta in {+-1,+-2,+-4}. The GO/NO-GO gate
needs both halves — this run establishes localization; steering follows.

Run on the A100 with HF_TOKEN set. Use `limit` in the config for a fast pipeline dry run.
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.hooks import TAP_SUFFIXES, keep_language_taps
from ..bridge.multimodal import TEXT_EMOTION_PROMPT
from ..data import APPRAISAL_TARGETS, EMOTION_LABELS, verify_label_tokenization
from ..data.crowd_envent import load_split, sample_tak_subset
from ..paths import STAGE_A_DIR, ensure_dirs
from ..probes.evaluate import best_layer, probe_r2
from ..probes.train import fit_appraisal_probes, fit_ridge
from .common import load_config, run_stamp, save_json, save_probes


def extract_all_taps(bridge, texts, n_layers, desc="extract"):
    """One forward pass per text; return {tap: {layer: [n, d]}} of last-token activations."""
    keep = keep_language_taps()  # all three LM taps
    store = {tap: {i: [] for i in range(n_layers)} for tap in TAP_SUFFIXES}
    for text in tqdm(texts, desc=desc):
        input_ids = bridge.to_tokens(TEXT_EMOTION_PROMPT.format(text=text))
        _, cache = bridge.run_with_cache(input_ids, names_filter=keep)
        last = input_ids.shape[-1] - 1
        for tap in TAP_SUFFIXES:
            for i in range(n_layers):
                store[tap][i].append(cache[f"blocks.{i}.{tap}"][0, last].float().cpu().numpy())
    return {tap: {i: np.stack(v) for i, v in layers.items()} for tap, layers in store.items()}


def _load_targets(split, tak_subset, seed, limit):
    df = load_split(split, seed=seed)
    if split == "train" and tak_subset:
        df = sample_tak_subset(df, seed=seed)
    if limit:
        df = df.head(limit)
    targets = [a for a in APPRAISAL_TARGETS if a in df.columns]
    Y = {a: df[a].to_numpy(dtype=np.float32) for a in targets}
    return df["text"].tolist(), Y


def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    seed = int(cfg.get("seed", 0))
    alpha = float(cfg.get("ridge_alpha", 1.0))
    limit = limit_override if limit_override is not None else cfg.get("limit")

    tr_texts, Ytr = _load_targets("train", cfg.get("tak_subset", True), seed, limit)
    va_texts, Yva = _load_targets("val", False, seed, (limit // 4 or 1) if limit else None)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    n_layers = int(bridge.cfg.n_layers)
    tok_report = verify_label_tokenization(bridge.tokenizer, EMOTION_LABELS)

    Xtr = extract_all_taps(bridge, tr_texts, n_layers, desc="train acts")
    Xva = extract_all_taps(bridge, va_texts, n_layers, desc="val acts")

    rng = np.random.default_rng(seed)
    metrics = {"run": run_stamp(), "seed": seed, "alpha": alpha, "limit": limit,
               "n_train": len(tr_texts), "n_val": len(va_texts),
               "tokenization": tok_report, "layerwise_val_r2": {},
               "shuffled_baseline_val_r2": {}, "best_layer": {}}

    for tap in TAP_SUFFIXES:
        metrics["layerwise_val_r2"][tap] = {}
        metrics["shuffled_baseline_val_r2"][tap] = {}
        for appraisal in Ytr:
            ytr, yva = Ytr[appraisal], Yva[appraisal]
            lr2, base = {}, {}
            for layer in range(n_layers):
                coef, b = fit_ridge(Xtr[tap][layer], ytr, alpha=alpha)
                lr2[layer] = probe_r2(Xva[tap][layer], yva, coef, b)
                # shuffled-label baseline: same fit on permuted y, scored on val
                sc, sb = fit_ridge(Xtr[tap][layer], rng.permutation(ytr), alpha=alpha)
                base[layer] = probe_r2(Xva[tap][layer], yva, sc, sb)
            metrics["layerwise_val_r2"][tap][appraisal] = lr2
            metrics["shuffled_baseline_val_r2"][tap][appraisal] = base
            layer, score = best_layer(lr2)
            metrics["best_layer"].setdefault(appraisal, {})[tap] = {"layer": layer, "val_r2": score}

    # Critical layer: median over appraisals of the best MHSA-tap layer (selected on val).
    crit_layer = int(np.median([v["hook_attn_out"]["layer"] for v in metrics["best_layer"].values()]))
    metrics["critical_layer"] = crit_layer

    # Frozen probes for Stage C: fit at the critical MHSA layer on TRAIN activations.
    probes = fit_appraisal_probes(Xtr["hook_attn_out"][crit_layer], Ytr, alpha=alpha)
    save_probes(probes, STAGE_A_DIR / "probes.npz")
    save_json(metrics, STAGE_A_DIR / "metrics.json")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A — text appraisal replication gate")
    ap.add_argument("--config", default="config/stage_a.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only N train examples (fast dry run); overrides the config")
    args = ap.parse_args()
    m = run(args.config, limit_override=args.limit)
    print(f"\nStage A done. critical_layer={m['critical_layer']} "
          f"n_train={m['n_train']} n_val={m['n_val']}. See {STAGE_A_DIR/'metrics.json'}")
    # quick console peek: best val r2 per appraisal at the MHSA tap
    for appraisal, taps in m["best_layer"].items():
        bl = taps["hook_attn_out"]
        print(f"  {appraisal:18s} MHSA best layer {bl['layer']:2d}  val_r2={bl['val_r2']:.3f}")


if __name__ == "__main__":
    main()
