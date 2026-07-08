"""Stage C — cross-modal read-out with FROZEN text probes (docs/experiment-1.md).

Preconditions: results/stage_a/probes.npz must exist (the frozen text-trained probes).
Never re-fit probes on image data here (data-rules.md).

Pipeline:
  1. load frozen Stage A probes
  2. for each EMOTIC image: build multimodal inputs, cache last-token activation at the
     Stage A critical layer/tap
  3. score each frozen appraisal probe on image activations -> transfer r2
  4. validate the pleasantness probe against EMOTIC continuous valence (1-10)
  5. report the transfer gap = text-probe r2 (text) - text-probe r2 (image)
  6. write results/stage_c/metrics.json

Run on a GPU box with HF_TOKEN set and EMOTIC downloaded.
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.emotic import load_split, to_shared_single_label
from ..paths import STAGE_A_DIR, STAGE_C_DIR, ensure_dirs
from ..probes.evaluate import probe_r2
from .common import load_config, load_probes, run_stamp, save_json


def image_activations(bridge, image_paths, layer: int, tap: str):
    """Return [n, d] frozen-layer last-token activations under image conditioning."""
    from ..bridge.hooks import keep_language_taps

    keep = keep_language_taps((tap,))
    name = f"blocks.{layer}.{tap}"
    rows = []
    for path in image_paths:
        inputs = build_image_inputs(bridge, Image.open(path).convert("RGB"))
        _, cache = bridge.run_with_cache(
            inputs["input_ids"], pixel_values=inputs["pixel_values"], names_filter=keep,
        )
        last = inputs["input_ids"].shape[-1] - 1
        rows.append(cache[name][0, last].float().cpu().numpy())
    return np.stack(rows)


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()

    probes_path = STAGE_A_DIR / "probes.npz"
    if not probes_path.exists():
        raise FileNotFoundError(
            f"{probes_path} missing — Stage A must pass and save frozen probes before Stage C."
        )
    probes = load_probes(probes_path)

    stage_a_metrics = load_config(STAGE_A_DIR / "metrics.json") if (STAGE_A_DIR / "metrics.json").exists() else {}
    layer = int(cfg.get("critical_layer", stage_a_metrics.get("critical_layer", 20)))
    tap = cfg.get("tap", "hook_attn_out")

    df = to_shared_single_label(load_split(cfg.get("split", "test")))
    n_dropped = df.attrs.get("n_dropped_multilabel", None)
    image_paths = df["image_path"].tolist()

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    X_img = image_activations(bridge, image_paths, layer, tap)

    metrics = {"run": run_stamp(), "layer": layer, "tap": tap,
               "n_images": len(image_paths), "n_dropped_multilabel": n_dropped,
               "transfer_r2": {}, "valence_check": None}

    # Frozen-probe read-out r2 for any appraisal with an image-side target column.
    for i, appraisal in enumerate(probes.names):
        if appraisal in df.columns:
            y = df[appraisal].to_numpy(dtype=np.float32)
            metrics["transfer_r2"][appraisal] = probe_r2(X_img, y, probes.coef[i], probes.intercept[i])

    # Pleasantness probe vs EMOTIC continuous valence (1-10) as ground-truth proxy.
    if "pleasantness" in probes.names and "valence" in df.columns:
        i = probes.index("pleasantness")
        y_val = df["valence"].to_numpy(dtype=np.float32)
        metrics["valence_check"] = probe_r2(X_img, y_val, probes.coef[i], probes.intercept[i])

    save_json(metrics, STAGE_C_DIR / "metrics.json")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage C — cross-modal frozen-probe read-out")
    ap.add_argument("--config", default="config/stage_c.yaml")
    args = ap.parse_args()
    m = run(args.config)
    print(f"Stage C done. layer={m['layer']} tap={m['tap']} n_images={m['n_images']}. "
          f"See {STAGE_C_DIR/'metrics.json'}")


if __name__ == "__main__":
    main()
