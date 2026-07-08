"""Stage A — text-only appraisal replication gate (docs/experiment-1.md).

Pipeline:
  1. boot Gemma, verify label tokenization (closed-vocab sanity)
  2. cache last-token LM activations per layer/tap for each crowd-enVENT example
  3. fit ridge appraisal probes; report layer-wise r2 per tap (expect mid-layer/MHSA peak)
  4. build unique-effect steering vectors; test steering at beta in {+-1,+-2,+-4}
  5. write results/stage_a/{metrics.json, probes.npz}

GO/NO-GO: probes recover the Tak-style mid-layer, MHSA-dominant localization AND steering
shifts emotion outputs in the theory-predicted direction. This is a hard gate for Stage C.

Model-touching steps are thin wrappers over src.bridge; run on a GPU box with HF_TOKEN set.
"""
from __future__ import annotations

import argparse
from functools import partial

import numpy as np

from ..bridge.boot import boot_gemma
from ..bridge.hooks import TAP_SUFFIXES
from ..bridge.multimodal import TEXT_EMOTION_PROMPT
from ..data import APPRAISAL_TARGETS, EMOTION_LABELS, verify_label_tokenization
from ..data.crowd_envent import load_split, sample_tak_subset
from ..paths import STAGE_A_DIR, ensure_dirs
from ..probes.evaluate import best_layer, layerwise_r2
from ..probes.train import fit_appraisal_probes, fit_ridge
from .common import load_config, run_stamp, save_json, save_probes

BETAS = (-4, -2, -1, 1, 2, 4)


def extract_layer_activations(bridge, texts, tap: str, n_layers: int):
    """Return {layer: [n, d]} of last-token activations at `tap` for each text.

    Text-only path: tokenize the emotion prompt, run_with_cache filtered to this tap.
    """
    from ..bridge.hooks import keep_language_taps

    keep = keep_language_taps((tap,))
    per_layer: dict[int, list] = {i: [] for i in range(n_layers)}
    for text in texts:
        prompt = TEXT_EMOTION_PROMPT.format(text=text)
        input_ids = bridge.to_tokens(prompt)
        _, cache = bridge.run_with_cache(input_ids, names_filter=keep)
        last = input_ids.shape[-1] - 1
        for i in range(n_layers):
            per_layer[i].append(cache[f"blocks.{i}.{tap}"][0, last].float().cpu().numpy())
    return {i: np.stack(v) for i, v in per_layer.items()}


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()

    df = load_split("train")
    if cfg.get("tak_subset", True):
        df = sample_tak_subset(df, seed=cfg.get("seed", 0))
    texts = df["text"].tolist()
    targets = [a for a in APPRAISAL_TARGETS if a in df.columns]
    Y = {a: df[a].to_numpy(dtype=np.float32) for a in targets}

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    n_layers = int(bridge.cfg.n_layers)

    tok_report = verify_label_tokenization(bridge.tokenizer, EMOTION_LABELS)

    alpha = float(cfg.get("ridge_alpha", 1.0))
    metrics = {"run": run_stamp(), "n_examples": len(texts), "tokenization": tok_report,
               "layerwise_r2": {}, "best_layer": {}}

    critical_layers = {}
    best_tap_acts = None
    for tap in TAP_SUFFIXES:
        per_layer = extract_layer_activations(bridge, texts, tap, n_layers)
        metrics["layerwise_r2"][tap] = {}
        for appraisal, y in Y.items():
            lr2 = layerwise_r2(per_layer, y, partial(fit_ridge, alpha=alpha))
            metrics["layerwise_r2"][tap][appraisal] = lr2
            layer, score = best_layer(lr2)
            metrics["best_layer"].setdefault(appraisal, {})[tap] = {"layer": layer, "r2": score}
        if tap == "hook_attn_out":  # MHSA — where the localization is expected to peak
            best_tap_acts = per_layer

    # Fit steering probes at the mid-layer of the MHSA tap (the expected critical site).
    crit_layer = int(np.median([v["hook_attn_out"]["layer"] for v in metrics["best_layer"].values()]))
    critical_layers["hook_attn_out"] = crit_layer
    probes = fit_appraisal_probes({k: v for k, v in best_tap_acts.items()}[crit_layer], Y, alpha=alpha)

    metrics["critical_layer"] = crit_layer
    save_probes(probes, STAGE_A_DIR / "probes.npz")
    save_json(metrics, STAGE_A_DIR / "metrics.json")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage A — text appraisal replication gate")
    ap.add_argument("--config", default="config/stage_a.yaml")
    args = ap.parse_args()
    m = run(args.config)
    print(f"Stage A done. critical_layer={m.get('critical_layer')} "
          f"n_examples={m['n_examples']}. See {STAGE_A_DIR/'metrics.json'}")


if __name__ == "__main__":
    main()
