"""Shared activation-patching math and TransformerBridge hook primitives."""
from __future__ import annotations

import numpy as np
import torch

from ...data.conflict_contexts import NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS

SAME_IMAGE_GROUPS = (
    "image", "question", "bos", "prefix_delim", "suffix_delim", "structure", "text_all"
)
CROSS_IMAGE_CONTEXT_BANKS = {
    "neutral": NEUTRAL_CONTEXTS,
    "positive": POSITIVE_CONTEXTS,
    "negative": NEGATIVE_CONTEXTS,
}


def find_subsequence(haystack, needle) -> int | None:
    """Return the first exact subsequence start, or ``None`` for no match."""
    if not needle:
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return None


def segment_prompt_positions(tokenizer, input_ids, question: str,
                             expected_image_tokens: int | None) -> dict:
    """Partition one multimodal prompt without assuming a backend-specific tokenizer wrapper."""
    token_values = input_ids[0].tolist()
    n_tokens = len(token_values)
    best_start, best_length = 0, 0
    start = 0
    while start < n_tokens:
        end = start
        while end + 1 < n_tokens and token_values[end + 1] == token_values[start]:
            end += 1
        length = end - start + 1
        if length > best_length:
            best_start, best_length = start, length
        start = end + 1

    image_end = best_start + best_length
    image = np.arange(best_start, image_end)
    question_start, question_length = None, 0
    for anchor in (" " + question, question):
        encoded = tokenizer.encode(anchor, add_special_tokens=False)
        question_start = find_subsequence(token_values, encoded)
        if question_start is not None:
            question_length = len(encoded)
            break
    if question_start is None or question_start <= image_end:
        question_start, question_length = max(image_end, n_tokens - 12), 0

    context = np.arange(image_end, question_start)
    question_positions = (
        np.arange(question_start, min(question_start + question_length, n_tokens))
        if question_length else np.array([], dtype=int)
    )
    excluded = set(image.tolist()) | set(context.tolist())
    template = np.array([index for index in range(n_tokens) if index not in excluded])
    image_ok = (
        best_length == expected_image_tokens
        if expected_image_tokens is not None else bool(best_length)
    )
    return {
        "image": image,
        "context": context,
        "question": question_positions,
        "template": template,
        "n": n_tokens,
        "img_len": int(best_length),
        "question_ok": bool(question_length),
        "image_ok": image_ok,
    }


def aligned_patch_groups(donor_segment, recipient_segment) -> tuple[dict, dict]:
    """Map donor positions to aligned recipient positions for same-image patching."""
    out, ok = {}, {}
    donor_image, recipient_image = donor_segment["image"], recipient_segment["image"]
    ok["image"] = bool(
        len(donor_image)
        and len(donor_image) == len(recipient_image)
        and int(donor_image[0]) == int(recipient_image[0])
    )
    if ok["image"]:
        out["image"] = (donor_image, recipient_image)

    donor_question, recipient_question = donor_segment["question"], recipient_segment["question"]
    ok["question"] = bool(
        len(donor_question) and len(donor_question) == len(recipient_question)
    )
    if ok["question"]:
        out["question"] = (donor_question, recipient_question)

    ok["structure"] = ok["bos"] = ok["prefix_delim"] = ok["suffix_delim"] = False
    if ok["image"] and ok["question"]:
        prefix = np.arange(0, int(donor_segment["image"][0]))
        donor_question_end = int(donor_question[-1]) + 1
        recipient_question_end = int(recipient_question[-1]) + 1
        donor_suffix = np.arange(donor_question_end, donor_segment["n"] - 1)
        recipient_suffix = np.arange(recipient_question_end, recipient_segment["n"] - 1)
        if len(donor_suffix) == len(recipient_suffix) and len(prefix) >= 2 and len(donor_suffix) >= 1:
            out["structure"] = (
                np.concatenate([prefix, donor_suffix]),
                np.concatenate([prefix, recipient_suffix]),
            )
            out["bos"] = (prefix[:1], prefix[:1])
            out["prefix_delim"] = (prefix[1:], prefix[1:])
            out["suffix_delim"] = (donor_suffix, recipient_suffix)
            ok["structure"] = ok["bos"] = ok["prefix_delim"] = ok["suffix_delim"] = True
    if ok["question"] and ok["structure"]:
        out["text_all"] = (
            np.concatenate([out["question"][0], out["structure"][0]]),
            np.concatenate([out["question"][1], out["structure"][1]]),
        )
        ok["text_all"] = True
    return out, ok


def cross_image_groups(segment, expected_image_tokens: int | None = 256) -> tuple[dict, dict]:
    """Build identical-sequence patch groups, excluding the final read-out query token."""
    n_tokens = int(segment["n"])
    image = np.asarray(segment["image"], dtype=int)
    context = np.asarray(segment["context"], dtype=int)
    question = np.asarray(segment["question"], dtype=int)
    ok = {}
    ok["image"] = (
        bool(len(image) == expected_image_tokens)
        if expected_image_tokens is not None else bool(len(image))
    )
    ok["question"] = bool(len(question))
    ok["context"] = bool(len(context))
    image_start = int(image[0]) if len(image) else 0
    question_end = (
        int(question[-1]) + 1
        if len(question) else (int(image[-1]) + 1 if len(image) else 0)
    )
    prefix = np.arange(0, image_start)
    suffix = np.arange(question_end, n_tokens - 1)
    structure = (
        np.concatenate([prefix, suffix])
        if (len(prefix) or len(suffix)) else np.array([], dtype=int)
    )
    ok["structure"] = bool(len(structure))
    text_all = (
        np.unique(np.concatenate([array for array in (context, question, structure) if len(array)]))
        if (len(context) or len(question) or len(structure)) else np.array([], dtype=int)
    )
    all_positions = (
        np.unique(np.concatenate([array for array in (image, text_all) if len(array)]))
        if (len(image) or len(text_all)) else np.array([], dtype=int)
    )
    ok["text_all"] = bool(len(text_all))
    ok["all"] = bool(len(all_positions))
    out = {
        "image": image,
        "context": context,
        "question": question,
        "structure": structure,
        "text_all": text_all,
        "all": all_positions,
    }
    return out, ok


def stash_activation(store):
    """Return a TransformerBridge hook that records its activation without changing it."""
    def hook(activation, hook):  # noqa: ARG001 - TransformerBridge hook contract
        store["act"] = activation.detach()
        return activation
    return hook


def bridge_patch_hook(recipient_indices, donor_values):
    """Overwrite batch-zero recipient positions with donor residual values."""
    indices = torch.as_tensor(recipient_indices, dtype=torch.long)

    def hook(activation, hook):  # noqa: ARG001 - TransformerBridge hook contract
        activation[0, indices, :] = donor_values.to(activation.dtype)
        return activation
    return hook


def collapse_duplicate_image_rows(df, image_column: str = "image_path"):
    """Return one deterministic row per image, validating repeated measurements first.

    Stage F selection can contain multiple annotations that resolve to the same underlying image.
    Those rows are not independent experimental units. Repeated model read-outs for an identical
    image must also be identical; a mismatch is raised instead of being silently averaged.

    Frames without ``image_column`` retain their historical one-row-per-unit behavior, which keeps
    the helper usable for small synthetic tests and legacy artifacts without path provenance.
    """
    frame = df.reset_index(drop=True).copy()
    if frame.empty or image_column not in frame:
        return frame
    if frame[image_column].isna().any():
        raise ValueError(f"{image_column} contains missing values; bootstrap units are ambiguous")

    duplicated = frame[frame.duplicated(image_column, keep=False)]
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column.startswith(("pos_", "neg_", "patch_"))
    ]
    for image_path, repeated in duplicated.groupby(image_column, sort=False):
        if not numeric_columns:
            continue
        values = repeated[numeric_columns].to_numpy(dtype=float)
        if not np.allclose(values, values[:1], rtol=1e-7, atol=1e-9, equal_nan=True):
            raise ValueError(
                f"repeated image {image_path!r} has inconsistent model read-outs"
            )
    return frame.drop_duplicates(image_column, keep="first").reset_index(drop=True)


def same_image_resampling_metadata(df, image_column: str = "image_path") -> dict:
    """Describe the independent units used by same-image estimates and confidence intervals."""
    unique = collapse_duplicate_image_rows(df, image_column=image_column)
    uses_image_paths = image_column in df
    return {
        "n_rows": int(len(df)),
        "n_images": int(len(unique)),
        "n_unique_images": int(len(unique)),
        "resampling_unit": f"unique {image_column}" if uses_image_paths else "row",
    }


def same_image_recovery(df, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Aggregate probe and behavioral recovery over unique images with paired CIs."""
    frame = collapse_duplicate_image_rows(df)
    out = {
        key: float(frame[key].mean())
        for key in ("pos_probe", "neg_probe", "pos_val", "neg_val")
    }
    if frame.empty:
        return out
    rng = np.random.default_rng(seed)
    bootstrap_indices = [rng.integers(0, len(frame), len(frame)) for _ in range(n_boot)]
    for group in groups:
        probe, probe_ci = _bootstrap_recovery(
            frame[f"patch_{group}_probe"].to_numpy(dtype=float),
            frame["pos_probe"].to_numpy(dtype=float),
            frame["neg_probe"].to_numpy(dtype=float),
            bootstrap_indices,
        )
        valence, valence_ci = _bootstrap_recovery(
            frame[f"patch_{group}_val"].to_numpy(dtype=float),
            frame["pos_val"].to_numpy(dtype=float),
            frame["neg_val"].to_numpy(dtype=float),
            bootstrap_indices,
        )
        out[group] = {
            "probe": probe,
            "probe_ci95": probe_ci,
            "val": valence,
            "val_ci95": valence_ci,
        }
    return out


def behavioral_same_image_recovery(df, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Aggregate behavioral recovery with paired bootstrap resampling over unique images."""
    frame = collapse_duplicate_image_rows(df)
    if frame.empty:
        return {"pos_val": float("nan"), "neg_val": float("nan")}
    positive = frame["pos_val"].to_numpy(dtype=float)
    negative = frame["neg_val"].to_numpy(dtype=float)
    out = {"pos_val": float(positive.mean()), "neg_val": float(negative.mean())}
    rng = np.random.default_rng(seed)
    bootstrap_indices = [rng.integers(0, len(frame), len(frame)) for _ in range(n_boot)]
    for group in groups:
        estimate, ci = _bootstrap_recovery(
            frame[f"patch_{group}_val"].to_numpy(dtype=float),
            positive,
            negative,
            bootstrap_indices,
        )
        out[group] = {"val": estimate, "ci95": ci}
    return out


def _bootstrap_recovery(patch, positive, negative, bootstrap_indices) -> tuple[float, list]:
    numerator, denominator = patch - negative, positive - negative
    mean_denominator = denominator.mean()
    estimate = float(numerator.mean() / mean_denominator) if mean_denominator else float("nan")
    values = []
    for indices in bootstrap_indices:
        sampled_denominator = denominator[indices].mean()
        if sampled_denominator:
            values.append(numerator[indices].mean() / sampled_denominator)
    ci = (
        [float(value) for value in np.percentile(values, [2.5, 97.5])]
        if values else [float("nan"), float("nan")]
    )
    return estimate, ci


def cross_image_recovery(df, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Return cross-image recovery with shared clustered-bootstrap resamples across groups."""
    out = {key: float(df[key].mean()) for key in ("pos_probe", "neg_probe", "pos_val", "neg_val")}
    if df.empty:
        return out
    n_pairs = len(df)
    rng = np.random.default_rng(seed)
    bootstrap_indices = [rng.integers(0, n_pairs, n_pairs) for _ in range(n_boot)]
    for group in groups:
        probe, probe_ci = _bootstrap_recovery(
            df[f"patch_{group}_probe"].to_numpy(),
            df["pos_probe"].to_numpy(),
            df["neg_probe"].to_numpy(),
            bootstrap_indices,
        )
        valence, valence_ci = _bootstrap_recovery(
            df[f"patch_{group}_val"].to_numpy(),
            df["pos_val"].to_numpy(),
            df["neg_val"].to_numpy(),
            bootstrap_indices,
        )
        out[group] = {
            "probe": probe,
            "probe_ci95": probe_ci,
            "val": valence,
            "val_ci95": valence_ci,
        }
    out["n_pairs"] = int(n_pairs)
    return out


def probe_recovery_valid(patch_layers, critical_layer) -> bool:
    """Whether every residual patch is upstream of the frozen probe tap."""
    return bool(patch_layers) and max(patch_layers) < critical_layer
