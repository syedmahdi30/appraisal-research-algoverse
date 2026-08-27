"""Stage F — LLaVA-1.5 same-image activation-patching pilot.

This is the behavioral analogue of Gemma's carrier experiment. A positive-context run donates
decoder residual states to a negative-context run of the same positive-valence image. Recovery is
measured with complete teacher-forced emotion-label sequence scoring; the Gemma probe is not reused
because it lives in a different activation space.

The default layer band is deliberately broad because LLaVA has no layerwise localization result yet.
Run a small A100 smoke pass first, then the 60-image pilot::

    python -m src.experiments.stage_f_llava_patching --limit 2
    python -m src.experiments.stage_f_llava_patching
    python -m src.experiments.stage_f_llava_patching --reanalyze

Use the separate ``requirements-qwen.txt`` environment. No paper artifact is modified by this runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import NEGATIVE_CONTEXTS, POSITIVE_CONTEXTS
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import PROCESSED_DIR, STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .multitoken_scoring import score_label_sequences
from .shared.artifacts import model_key
from .shared.hf_runtime import capture_residuals, find_language_layers, patch_residuals
from .shared.patching import (
    SAME_IMAGE_GROUPS,
    aligned_patch_groups,
    behavioral_same_image_recovery,
    segment_prompt_positions,
)
from .shared.readouts import QUESTION, processor_pad_token_id
from .shared.sampling import select_extreme_rows
from .stage_f_llava import DEFAULT_MODEL, build_inputs, load_llava

GROUPS = SAME_IMAGE_GROUPS
SCORE_RULE = "sum of teacher-forced conditional token log probabilities"


def _artifact_paths(model_name: str = DEFAULT_MODEL) -> tuple[Path, Path]:
    """Return collision-safe outputs for one LLaVA-family checkpoint."""
    key = "llava" if model_name == DEFAULT_MODEL else model_key(model_name)
    stem = f"patching_{key}_sequence"
    return STAGE_F_DIR / f"{stem}.parquet", STAGE_F_DIR / f"{stem}_metrics.json"


def _parse_layer_band(spec: str | None, n_layers: int) -> list[int]:
    """Parse an inclusive CLI band or choose the Qwen-matched proportional default."""
    if n_layers < 3:
        raise ValueError(f"decoder depth must be at least 3, got {n_layers}")
    if spec is None:
        band = list(range(round(0.35 * n_layers), n_layers - 2))
    else:
        try:
            start_text, end_text = spec.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("layer band must use inclusive START-END syntax") from exc
        if start < 0 or end < start or end >= n_layers:
            raise ValueError(
                f"layer band {start}-{end} is outside decoder depth 0-{n_layers - 1}"
            )
        band = list(range(start, end + 1))
    if not band:
        raise ValueError(f"layer band is empty for decoder depth {n_layers}")
    return band


def _validated_groups(donor_segment: dict, recipient_segment: dict) -> dict:
    """Return aligned patch groups or fail before an uninterpretable GPU run continues."""
    if min(donor_segment["img_len"], recipient_segment["img_len"]) <= 8:
        raise RuntimeError(
            "image placeholder was not expanded into a patch-token block (<=8 repeated tokens)"
        )
    if not donor_segment["image_ok"] or not recipient_segment["image_ok"]:
        raise RuntimeError("image token segmentation failed")
    if not donor_segment["question_ok"] or not recipient_segment["question_ok"]:
        raise RuntimeError("question token segmentation failed")
    groups, ok = aligned_patch_groups(donor_segment, recipient_segment)
    missing = [group for group in GROUPS if not ok.get(group)]
    if missing:
        raise RuntimeError(f"unaligned LLaVA patch groups: {', '.join(missing)}")
    return groups


def _sequence_valence(model, inputs, label_ids, pad_token_id: int,
                      label_batch_size: int) -> float:
    scored = score_label_sequences(
        model,
        inputs,
        label_ids,
        pad_token_id=pad_token_id,
        label_batch_size=label_batch_size,
    )
    return float(scored["sequence_sum"]["valence"])


def _verdict(recovery: dict) -> str:
    if "image" not in recovery:
        return "no images analysed"
    values = {group: recovery[group]["val"] for group in GROUPS}
    if max(abs(value) for value in values.values()) < 0.05:
        return "NO RECOVERY: validate decoder hooks and the selected layer band before interpretation."
    dominant = max(("bos", "prefix_delim", "suffix_delim"), key=values.get)
    image_read = (
        "image tokens are causally inert for the context delta"
        if abs(values["image"]) < 0.08
        else f"image tokens recover {values['image']:.0%}"
    )
    return (
        f"{image_read}; question {values['question']:.0%}, structure "
        f"{values['structure']:.0%}, all-text {values['text_all']:.0%}; dominant structural "
        f"group {dominant} ({values[dominant]:.0%}). Position-group recoveries are not assumed "
        "additive."
    )


def _analyze(
    df: pd.DataFrame,
    *,
    model_name: str,
    patch_layers: list[int],
    n_layers: int,
    donor_context: str,
    recipient_context: str,
    n_skipped: int,
    data_path: Path,
    metrics_path: Path,
    n_boot: int = 2000,
) -> dict:
    recovery = behavioral_same_image_recovery(df, GROUPS, n_boot=n_boot)
    metrics = {
        "run": run_stamp(),
        "git": git_hash(),
        "model": model_name,
        "read_out": "behavioral_valence",
        "score_mode": "sequence",
        "score_rule": SCORE_RULE,
        "n_layers": n_layers,
        "patch_layers": patch_layers,
        "n_images": int(len(df)),
        "n_skipped": n_skipped,
        "donor_positive_context": donor_context,
        "recipient_negative_context": recipient_context,
        "recovery": recovery,
        "verdict": _verdict(recovery),
    }
    save_json(metrics, metrics_path)

    print(
        f"\nStage F [LLaVA: {model_name}] sequence patching — {len(df)} positive images "
        f"({n_skipped} skipped); decoder layers {patch_layers[0]}-{patch_layers[-1]} of "
        f"{n_layers}."
    )
    if "image" in recovery:
        print(f"  baselines: positive {recovery['pos_val']:+.3f}, negative {recovery['neg_val']:+.3f}")
        print(f"  {'group':13s} {'recovery':>9s}   95% CI")
        for group in GROUPS:
            item = recovery[group]
            print(
                f"  {group:13s} {item['val'] * 100:>8.0f}%   "
                f"[{item['ci95'][0] * 100:+.0f}%, {item['ci95'][1] * 100:+.0f}%]"
            )
    print(f"  VERDICT: {metrics['verdict']}")
    print(f"  data -> {data_path}   metrics -> {metrics_path}")
    return metrics


def run(
    config_path: str,
    model_name: str,
    *,
    limit_override: int | None = None,
    layers_override: str | None = None,
    pos_idx: int | None = None,
    neg_idx: int | None = None,
    label_batch_size: int = 4,
) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    n_images = limit_override or int(cfg.get("patch_n_images", 60))
    positive_index = pos_idx if pos_idx is not None else int(cfg.get("patch_pos_idx", 0))
    negative_index = neg_idx if neg_idx is not None else int(cfg.get("patch_neg_idx", 2))
    positive_context = POSITIVE_CONTEXTS[positive_index]
    negative_context = NEGATIVE_CONTEXTS[negative_index]

    model, processor = load_llava(model_name)
    language_layers = find_language_layers(model)
    patch_layers = _parse_layer_band(layers_override, len(language_layers))
    token_report = verify_label_tokenization(processor.tokenizer)
    label_ids = {label: token_report[label]["ids"] for label in EMOTION_LABELS}
    pad_token_id = processor_pad_token_id(processor)

    frame = pd.read_parquet(PROCESSED_DIR / "emotic_test.parquet").reset_index(drop=True)
    selected = select_extreme_rows(frame, n_images * 2)
    selected = selected[selected["image_group"] == "positive"].head(n_images)

    rows = []
    n_skipped = 0
    for _, image_row in tqdm(list(selected.iterrows()), desc="stage-f llava patching"):
        try:
            image = Image.open(image_row["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skipped += 1
            continue

        donor_inputs = build_inputs(processor, image, positive_context)
        recipient_inputs = build_inputs(processor, image, negative_context)
        donor_segment = segment_prompt_positions(
            processor.tokenizer, donor_inputs["input_ids"], QUESTION, expected_image_tokens=None
        )
        recipient_segment = segment_prompt_positions(
            processor.tokenizer, recipient_inputs["input_ids"], QUESTION,
            expected_image_tokens=None,
        )
        groups = _validated_groups(donor_segment, recipient_segment)

        with capture_residuals(model, patch_layers) as donor_residuals:
            positive_valence = _sequence_valence(
                model, donor_inputs, label_ids, pad_token_id, label_batch_size
            )
        if set(donor_residuals) != set(patch_layers):
            missing = sorted(set(patch_layers) - set(donor_residuals))
            raise RuntimeError(f"decoder hooks did not capture layers: {missing}")
        negative_valence = _sequence_valence(
            model, recipient_inputs, label_ids, pad_token_id, label_batch_size
        )

        row = {
            "image_path": image_row["image_path"],
            "image_valence": float(image_row["valence"]),
            "pos_val": positive_valence,
            "neg_val": negative_valence,
        }
        for group in GROUPS:
            donor_indices, recipient_indices = groups[group]
            with patch_residuals(
                model,
                donor_residuals,
                donor_indices,
                recipient_indices,
                patch_all_batch_rows=True,
            ):
                row[f"patch_{group}_val"] = _sequence_valence(
                    model, recipient_inputs, label_ids, pad_token_id, label_batch_size
                )
        rows.append(row)

    data_path, metrics_path = _artifact_paths(model_name)
    df = pd.DataFrame(rows)
    df.to_parquet(data_path)
    return _analyze(
        df,
        model_name=model_name,
        patch_layers=patch_layers,
        n_layers=len(language_layers),
        donor_context=positive_context,
        recipient_context=negative_context,
        n_skipped=n_skipped,
        data_path=data_path,
        metrics_path=metrics_path,
    )


def reanalyze(model_name: str = DEFAULT_MODEL, n_boot: int = 2000) -> dict:
    """Recompute recovery and confidence intervals from a completed parquet on CPU."""
    ensure_dirs()
    data_path, metrics_path = _artifact_paths(model_name)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} missing — run the LLaVA patching pass first.")
    previous = load_config(metrics_path) if metrics_path.exists() else {}
    patch_layers = previous.get("patch_layers", [])
    if not patch_layers:
        raise ValueError(f"{metrics_path} does not record a patch layer band")
    return _analyze(
        pd.read_parquet(data_path),
        model_name=previous.get("model", model_name),
        patch_layers=patch_layers,
        n_layers=int(previous.get("n_layers", max(patch_layers) + 1)),
        donor_context=previous.get("donor_positive_context", ""),
        recipient_context=previous.get("recipient_negative_context", ""),
        n_skipped=int(previous.get("n_skipped", 0)),
        data_path=data_path,
        metrics_path=metrics_path,
        n_boot=n_boot,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage F LLaVA — same-image activation patching with sequence scoring"
    )
    parser.add_argument("--config", default="config/stage_f.yaml")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layers", default=None, help="inclusive decoder band START-END")
    parser.add_argument("--pos-idx", type=int, default=None)
    parser.add_argument("--neg-idx", type=int, default=None)
    parser.add_argument("--label-batch-size", type=int, default=4)
    parser.add_argument("--reanalyze", action="store_true")
    args = parser.parse_args()
    if args.reanalyze:
        reanalyze(args.model)
    else:
        run(
            args.config,
            args.model,
            limit_override=args.limit,
            layers_override=args.layers,
            pos_idx=args.pos_idx,
            neg_idx=args.neg_idx,
            label_batch_size=args.label_batch_size,
        )


if __name__ == "__main__":
    main()
