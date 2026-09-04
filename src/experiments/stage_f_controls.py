"""Stage F — the three controls an external reviewer asked for, on the raw-HuggingFace Qwen path.

All three answer objections the paper currently concedes rather than tests. They run through
`stage_f_qwen`'s reference implementation, NOT TransformerBridge: the withdrawn prompt sweep
(`stage_f_prompts.py`) booted via `boot_gemma`, whose wrapper altered the multimodal forward pass,
which is why its results were withdrawn. Nothing here may reintroduce that path.

  --person {crop,box}   Target-person grounding. EMOTIC annotates one person per row but the
                        published run shows the whole photograph with no pointer, and 65% of our
                        images carry more than one annotated person (81% of the positive group), so
                        "this person" is not referentially grounded. `box` draws an outline around
                        the annotated person and keeps the scene, isolating grounding from content;
                        `crop` removes the scene entirely, which tests person-level vs scene-level
                        reading but also deletes information, so `box` is the cleaner control and
                        `crop` the stronger one. Run both: agreement is the informative case.

  --axis {frame,question}
                        Robustness of the matched-pair asymmetry to wording. `frame` re-renders the
                        six valence swaps in four context frames (the original, plus the caption,
                        report and user-message carriers section 5 names); `question` varies the
                        question and holds the frame fixed, closing the withdrawn sweep. Frame 0 and
                        question 0 reproduce the published stimuli exactly, so each sweep contains
                        its own replication of the published number as a built-in check.

  --generate            Free-form generation instead of forced-choice scoring. The paper scores
                        closed-vocabulary log-probabilities at one position; a deployed system emits
                        text. This decodes greedily and maps the answer onto the 13 labels, so the
                        forced-choice effect can be checked against what the model would actually say.

Every mode writes the same column schema as `conflict_qwen.parquet` plus a `variant` column, so
`shared.reporting.minimal_pair_asymmetry` and `flip_override` consume the output unchanged and each
variant is scored exactly as the published run was. `--reanalyze` recomputes from the saved parquet
on CPU with no model load, mirroring `stage_f_qwen`.

GPU pass runs on Colab in the `requirements-qwen.txt` env, with EMOTIC staged at the parquet's paths.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from ..data.conflict_contexts import (CONTEXT_FRAMES, QUESTION_VARIANTS, TEXT_CODE,
                                      build_conditions,
                                      build_frame_conditions)
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .shared.readouts import (QUESTION, closed_vocab_logprobs, first_content_token_ids,
                              model_readout)
from .analyze_stage_f_unbounded import add_readouts, mirror_contrast, override_gap
from .shared.reporting import minimal_pair_asymmetry
from .stage_f_qwen import DEFAULT_MODEL, load_qwen, select_extreme_images

CROP_MARGIN = 0.10   # fraction of box size added on each side, so the crop is not flush to the person


def resolve_image_paths(df, images_root: str | None):
    """Rebuild `image_path` under `images_root` from the parquet's own `folder` and `filename`.

    The processed parquet stores absolute Colab paths
    (`/content/.../data/raw/emotic/emotic/<folder>/<filename>`), so a session that mounts EMOTIC
    anywhere else resolves nothing. Every row also carries `folder` and `filename`, and every
    `image_path` equals `<root>/<folder>/<filename>`, so the join is exact rather than prefix surgery.
    """
    if not images_root:
        return df
    root = str(images_root).rstrip("/")
    out = df.copy()
    out["image_path"] = root + "/" + out["folder"].astype(str) + "/" + out["filename"].astype(str)
    return out


# --------------------------------------------------------------------------- prompt construction
def build_inputs(processor, image, context_sentence, question: str = QUESTION):
    """Qwen chat inputs for one image, one context, one question.

    `stage_f_qwen.build_inputs` hardcodes the published question through `shared.readouts.user_text`.
    This takes the question explicitly so the question axis can move, and reproduces that function
    byte-for-byte when `question` is left at its default.
    """
    context = "" if not context_sentence else f"Context: {context_sentence} "
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": f"{context}{question}"})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image] if image is not None else None,
                     padding=True, return_tensors="pt")


# --------------------------------------------------------------------------- person grounding
def _bbox(row) -> tuple[int, int, int, int] | None:
    raw = row.get("bbox")
    if raw is None:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in list(raw))
    except (TypeError, ValueError):
        return None
    return int(x1), int(y1), int(x2), int(y2)


def ground_person(image: Image.Image, row, mode: str) -> Image.Image:
    """Return the image with the annotated person cropped to, outlined, or left alone.

    Returns the image unchanged when the row carries no usable box, and the caller records that in
    `n_ungrounded` — a silently-skipped row would make the control look cleaner than it is.
    """
    if mode == "none":
        return image
    box = _bbox(row)
    if box is None:
        return image
    x1, y1, x2, y2 = box
    width, height = image.size
    if mode == "crop":
        margin_x = int(CROP_MARGIN * max(1, x2 - x1))
        margin_y = int(CROP_MARGIN * max(1, y2 - y1))
        left, top = max(0, x1 - margin_x), max(0, y1 - margin_y)
        right, bottom = min(width, x2 + margin_x), min(height, y2 + margin_y)
        if right - left < 8 or bottom - top < 8:   # degenerate box: keep the whole frame
            return image
        return image.crop((left, top, right, bottom))
    if mode == "box":
        marked = image.copy()
        outline_width = max(3, int(0.006 * max(width, height)))
        ImageDraw.Draw(marked).rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=outline_width)
        return marked
    raise ValueError(f"unknown grounding mode {mode!r}")


# --------------------------------------------------------------------------- variant banks
def _variants(axis: str) -> list[tuple[str, list, str]]:
    """`(variant_name, conditions, question)` per variant for the requested axis."""
    if axis == "frame":
        return [(name, build_frame_conditions(i), QUESTION)
                for i, (name, _template) in enumerate(CONTEXT_FRAMES)]
    if axis == "question":
        return [(name, build_conditions("minimal"), text) for name, text in QUESTION_VARIANTS]
    raise ValueError(f"unknown axis {axis!r} — expected 'frame' or 'question'")


# --------------------------------------------------------------------------- scoring pass
def _score_rows(model, processor, tok_ids, sel, variants, grounding: str, desc: str):
    """Forward-score every (image, variant, condition) cell. Shared by the sweep and person modes."""
    rows, n_missing, n_ungrounded = [], 0, 0
    for _, row in tqdm(list(sel.iterrows()), desc=desc):
        try:
            base_image = Image.open(row["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_missing += 1
            continue
        if grounding != "none" and _bbox(row) is None:
            n_ungrounded += 1
        image = ground_person(base_image, row, grounding)
        for variant, conditions, question in variants:
            for condition, context_id, sentence in conditions:
                valence, logprobs = model_readout(
                    model, build_inputs(processor, image, sentence, question), tok_ids)
                rows.append({
                    "image_path": row["image_path"], "image_valence": float(row["valence"]),
                    "image_group": row["image_group"], "condition": condition,
                    "context_id": context_id, "context": sentence or "",
                    "text_code": TEXT_CODE[condition], "probe_readout": float("nan"),
                    "valence": valence, "variant": variant, "grounding": grounding,
                    **{f"lp_{label}": logprobs[label] for label in EMOTION_LABELS}})
    return pd.DataFrame(rows), n_missing, n_ungrounded


MAX_MISSING_FRACTION = 0.05


def check_coverage(n_scored: int, n_selected: int, n_missing: int, allow_missing: bool) -> None:
    """Refuse to report a control computed on a fraction of the intended images.

    A run whose images are not mounted still produces a plausible-looking variant table: the
    per-variant statistics are computed on whatever loaded, the interval widens, and nothing in the
    output says why. That happened on the first frame sweep -- 4 of 150 images loaded, 240 rows
    instead of 9,000, and the printed table looked ordinary. Coverage is therefore a hard gate rather
    than a logged number, because the failure is silent by construction.
    """
    if n_selected and n_missing > MAX_MISSING_FRACTION * n_selected:
        message = (
            f"ABORT: {n_missing} of {n_selected} selected images could not be opened "
            f"({n_missing / n_selected:.0%}); only {n_scored} were scored. The EMOTIC images are "
            f"probably not mounted at the paths in data/processed/emotic_test.parquet. Verify with\n"
            f"    import pandas as pd, os\n"
            f"    d = pd.read_parquet('data/processed/emotic_test.parquet')\n"
            f"    print(d.image_path.map(os.path.exists).mean())   # want 1.0\n"
            f"Pass --allow-missing to score the partial set anyway (NOT comparable to the "
            f"published numbers)."
        )
        if not allow_missing:
            raise RuntimeError(message)
        print(f"\n!! {message}\n")


def paired_variant_delta(df: pd.DataFrame, base: str, other: str,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap of the change in mirror contrast between two variants.

    Two variants scored on the same photographs are NOT independent, so comparing their separate
    confidence intervals answers the wrong question: overlapping intervals do not mean the variants
    agree, and disjoint ones would overstate the difference. The control asks whether grounding
    CHANGED the asymmetry, which is a paired quantity.

    Each bootstrap replicate draws one set of photograph and sentence indices and applies it to both
    variants, so the shared stimulus variance cancels the way it does in the underlying design. The
    returned interval is on `other - base`; an interval covering zero means the control reproduced.
    """
    from .analyze_stage_f_unbounded import _effect_matrix

    scored = add_readouts(df)
    grids = {}
    for name in (base, other):
        sub = scored[scored["variant"] == name]
        drop = _effect_matrix(sub, "positive", "negative", "valence")
        rise = _effect_matrix(sub, "negative", "positive", "valence")
        if drop is None or rise is None:
            return {}
        grids[name] = (drop, rise)

    # Pair on the photographs and sentences the two variants share, so one index set drives both.
    import numpy as np
    shared = {}
    for axis, position in (("drop", 0), ("rise", 1)):
        a, b = grids[base][position], grids[other][position]
        rows = a.index.intersection(b.index)
        cols = a.columns.intersection(b.columns)
        shared[axis] = (a.loc[rows, cols].to_numpy(float), b.loc[rows, cols].to_numpy(float))

    def mirror(drop_cells, rise_cells):
        return abs(np.nanmean(drop_cells)) - abs(np.nanmean(rise_cells))

    base_value = mirror(shared["drop"][0], shared["rise"][0])
    other_value = mirror(shared["drop"][1], shared["rise"][1])

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        di = rng.integers(0, shared["drop"][0].shape[0], shared["drop"][0].shape[0])
        dj = rng.integers(0, shared["drop"][0].shape[1], shared["drop"][0].shape[1])
        ri = rng.integers(0, shared["rise"][0].shape[0], shared["rise"][0].shape[0])
        rj = rng.integers(0, shared["rise"][0].shape[1], shared["rise"][0].shape[1])
        dsel, rsel = np.ix_(di, dj), np.ix_(ri, rj)
        deltas[k] = (mirror(shared["drop"][1][dsel], shared["rise"][1][rsel])
                     - mirror(shared["drop"][0][dsel], shared["rise"][0][rsel]))

    low, high = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    return {
        "base": base, "other": other,
        "base_mirror": float(base_value), "other_mirror": float(other_value),
        "delta": float(other_value - base_value),
        "delta_ci95_paired": [low, high],
        "reproduces": bool(low <= 0.0 <= high),
        "n_photographs": [int(shared["drop"][0].shape[0]), int(shared["rise"][0].shape[0])],
    }


def _versions() -> dict:
    """Record the library versions a control ran under.

    The published Qwen runs did not record them, so a control that disagrees cannot be separated from
    a version difference by inspection. The paired/`original` baselines are the real defence; this
    makes the question answerable rather than guessed.
    """
    out = {}
    for name in ("torch", "transformers", "numpy", "pandas", "PIL"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:
            out[name] = "absent"
    return out


def _analyze(df: pd.DataFrame, model_name: str, stem: str, extra: dict) -> dict:
    """Per-variant matched-pair contrast and override gap, printed side by side."""
    per_variant = {}
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        # Use the estimators the paper's numbers come from, verified against the published matched
        # run: `mirror_contrast` on the bounded readout reproduces +0.496 with crossed [+0.11,+0.83]
        # and image-only [+0.31,+0.67], and `minimal_pair_asymmetry` reproduces +1.148
        # [+0.943,+1.344]. `asymmetry_vs_floor` is NOT used: it differences cell means (+0.478 on the
        # same data) where the paper averages per photograph, so it would not be comparable.
        scored = add_readouts(sub)
        per_variant[str(variant)] = {
            "n_rows": int(len(sub)),
            "n_images": int(sub["image_path"].nunique()),
            "minimal_pair_asymmetry": minimal_pair_asymmetry(sub),
            "mirror_contrast": mirror_contrast(scored, "valence"),
            "override_gap": override_gap(scored),
        }
    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name,
               "read_out": "behavioral_valence", "n_rows": int(len(df)),
               "per_variant": per_variant, "versions": _versions(), **extra}
    save_json(metrics, STAGE_F_DIR / f"{stem}_metrics.json")

    n_img = int(df["image_path"].nunique()) if len(df) else 0
    skipped = extra.get("n_skipped") or 0
    skip_note = f", {skipped} SKIPPED" if skipped else ""
    print(f"\nStage F controls [{stem}] — {model_name}, "
          f"{len(df)} rows over {n_img} images{skip_note}")
    print(f"  {'variant':10s} {'within-item':>11s} {'95% CI':>18s} "
          f"{'mirror':>8s} {'crossed CI':>18s} {'clears 0':>9s}")
    for variant, m in per_variant.items():
        paired, mir = m["minimal_pair_asymmetry"], m["mirror_contrast"]
        within, wci = paired.get("paired_asymmetry"), paired.get("ci95") or [None, None]
        mirror, cci = mir.get("asymmetry_index"), mir.get("ci95_crossed") or [None, None]
        within_text = _fmt(within, "+.3f")
        mirror_text = _fmt(mirror, "+.3f")
        wci_text = _ci(wci)
        cci_text = _ci(cci)
        clears = str(mir.get("crossed_clears_zero"))
        print(f"  {variant:10s} {within_text:>11s} {wci_text:>18s} "
              f"{mirror_text:>8s} {cci_text:>18s} {clears:>9s}")
    # A paired run carries its own baseline, so report the change rather than two intervals the
    # reader would otherwise eyeball for overlap.
    variants = list(per_variant)
    if len(variants) == 2 and "none" in variants:
        other = next(v for v in variants if v != "none")
        paired = paired_variant_delta(df, "none", other)
        if paired:
            metrics["paired_delta"] = paired
            save_json(metrics, STAGE_F_DIR / f"{stem}_metrics.json")
            low, high = paired["delta_ci95_paired"]
            verdict = ("reproduces: the interval covers zero" if paired["reproduces"]
                       else "CHANGED: the interval excludes zero")
            print(f"\n  paired change in mirror contrast, {other} vs none: "
                  f"{paired['delta']:+.3f} [{low:+.3f}, {high:+.3f}]  -> {verdict}")
            print("  (paired on the same photographs; comparing the two rows' separate intervals "
                  "for overlap would answer a different question)")

    print(f"  data -> {STAGE_F_DIR / (stem + '.parquet')}")
    print("  Published matched reference: within-item +1.148 [+0.943,+1.344]; "
          "mirror +0.496, crossed [+0.11,+0.83].")
    return metrics


def _ci(pair) -> str:
    """Render a two-element interval, tolerating None/NaN from an empty variant."""
    return f"[{_fmt(pair[0], '+.3f')},{_fmt(pair[1], '+.3f')}]"


def _fmt(value, spec: str) -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


# --------------------------------------------------------------------------- modes
def run_sweep(config_path: str, model_name: str, axis: str, limit: int | None,
              allow_missing: bool = False, images_root: str | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    sel = resolve_image_paths(select_extreme_images(limit or int(cfg.get("n_images", 150))),
                              images_root)
    model, processor = load_qwen(model_name)
    tok_ids = first_content_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}
    variants = _variants(axis)
    df, n_missing, _ = _score_rows(model, processor, tok_ids, sel, variants, "none",
                                   f"stage-f controls {axis}")
    check_coverage(int(df["image_path"].nunique()) if len(df) else 0, len(sel), n_missing,
                   allow_missing)
    stem = f"controls_{axis}_qwen"
    df.to_parquet(STAGE_F_DIR / f"{stem}.parquet")
    return _analyze(df, model_name, stem,
                    {"axis": axis, "n_skipped": n_missing, "tokenization_multi_token": multi})


def run_person(config_path: str, model_name: str, mode: str, limit: int | None,
               allow_missing: bool = False, images_root: str | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    sel = resolve_image_paths(select_extreme_images(limit or int(cfg.get("n_images", 150))),
                              images_root)
    model, processor = load_qwen(model_name)
    tok_ids = first_content_token_ids(processor)
    # Score an ungrounded arm on the SAME images in the same invocation. Comparing a grounded run
    # against the published 121-image numbers would confound grounding with image set, which matters
    # because mscoco/ade20k images ship separately from EMOTIC and a session may hold only a subset.
    # With both arms on one image set the contrast is internally valid whatever subset is mounted.
    minimal = build_conditions("minimal")
    df_base, n_missing, _ = _score_rows(model, processor, tok_ids, sel, [("none", minimal, QUESTION)],
                                        "none", "stage-f controls person=none (paired baseline)")
    df_grounded, _, n_ungrounded = _score_rows(model, processor, tok_ids, sel,
                                               [(mode, minimal, QUESTION)], mode,
                                               f"stage-f controls person={mode}")
    df = pd.concat([df_base, df_grounded], ignore_index=True)
    check_coverage(int(df["image_path"].nunique()) if len(df) else 0, len(sel), n_missing,
                   allow_missing)
    stem = f"controls_person_{mode}_qwen"
    df.to_parquet(STAGE_F_DIR / f"{stem}.parquet")
    return _analyze(df, model_name, stem,
                    {"grounding": mode, "crop_margin": CROP_MARGIN, "n_skipped": n_missing,
                     "n_ungrounded": n_ungrounded})


def run_generate(config_path: str, model_name: str, limit: int | None,
                 max_new_tokens: int = 8, allow_missing: bool = False,
                 images_root: str | None = None) -> dict:
    """Greedy free-form generation on the matched bank, mapped onto the 13 labels.

    Deterministic (`do_sample=False`) so the comparison to the forced-choice readout is not a sampling
    artifact. An answer counts as a label if any label appears as a word in the decoded text; answers
    matching none are recorded as `other` rather than dropped, because a model that stops naming
    emotions under negative context is itself a result.
    """
    cfg = load_config(config_path)
    ensure_dirs()
    sel = resolve_image_paths(select_extreme_images(limit or int(cfg.get("n_images", 150))),
                              images_root)
    model, processor = load_qwen(model_name)
    conditions = build_conditions("minimal")

    positive = {"joy", "pride", "relief", "trust"}
    rows, n_missing = [], 0
    for _, row in tqdm(list(sel.iterrows()), desc="stage-f controls generate"):
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_missing += 1
            continue
        for condition, context_id, sentence in conditions:
            inputs = build_inputs(processor, image, sentence)
            moved = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
            text = processor.tokenizer.decode(
                out[0][moved["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            words = {w.strip(".,!?;:'\"").lower() for w in text.split()}
            hit = next((label for label in EMOTION_LABELS if label in words), "other")
            rows.append({"image_path": row["image_path"], "image_group": row["image_group"],
                         "condition": condition, "context_id": context_id,
                         "context": sentence or "", "generated": text, "generated_label": hit,
                         "generated_polarity": ("positive" if hit in positive
                                                else "other" if hit == "other" else "negative")})

    df = pd.DataFrame(rows)
    check_coverage(int(df["image_path"].nunique()) if len(df) else 0, len(sel), n_missing,
                   allow_missing)
    stem = "controls_generate_qwen"
    df.to_parquet(STAGE_F_DIR / f"{stem}.parquet")
    return _analyze_generate(df, model_name, stem)


def _analyze_generate(df: pd.DataFrame, model_name: str, stem: str) -> dict:
    """Override rate computed on generated answers, per image group."""
    summary = {}
    for group in ("positive", "negative"):
        sub = df[df["image_group"] == group]
        if sub.empty:
            continue
        opposite = "negative" if group == "positive" else "positive"
        conflicting = sub[sub["condition"] == opposite]
        neutral = sub[sub["condition"] == "neutral"]
        summary[group] = {
            "n": int(len(conflicting)),
            "conflicting_context_flips": float(
                (conflicting["generated_polarity"] == opposite).mean()) if len(conflicting) else None,
            "neutral_context_flips": float(
                (neutral["generated_polarity"] == opposite).mean()) if len(neutral) else None,
            "unparsed_rate": float((sub["generated_label"] == "other").mean()),
        }
    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name,
               "n_rows": int(len(df)), "per_group": summary, "versions": _versions()}
    save_json(metrics, STAGE_F_DIR / f"{stem}_metrics.json")
    print(f"\nStage F controls [generation] — {model_name}, {len(df)} generations")
    for group, m in summary.items():
        print(f"  {group:8s} images: conflicting context flips the generated answer "
              f"{_fmt(m['conflicting_context_flips'], '.1%')} vs neutral "
              f"{_fmt(m['neutral_context_flips'], '.1%')} "
              f"(unparsed {_fmt(m['unparsed_rate'], '.1%')}, n={m['n']})")
    print(f"  data -> {STAGE_F_DIR / (stem + '.parquet')}")
    return metrics


def reanalyze(stem: str) -> dict:
    """Recompute a saved control run's metrics on CPU, no model load."""
    ensure_dirs()
    path = STAGE_F_DIR / f"{stem}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the GPU pass first.")
    df = pd.read_parquet(path)
    mpath = STAGE_F_DIR / f"{stem}_metrics.json"
    model_name = load_config(mpath).get("model", "unknown") if mpath.exists() else "unknown"
    if "generated_label" in df.columns:
        return _analyze_generate(df, model_name, stem)
    return _analyze(df, model_name, stem, {})


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F reviewer controls (raw-HF Qwen)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--person", choices=["crop", "box"], help="target-person grounding control")
    ap.add_argument("--axis", choices=["frame", "question"], help="wording robustness sweep")
    ap.add_argument("--generate", action="store_true", help="free-form generation control")
    ap.add_argument("--limit", type=int, default=None, help="EMOTIC image count")
    ap.add_argument("--images-root", default=None,
                    help="rebuild image paths under this root from the parquet's folder/filename "
                         "(e.g. /content/emotic) when the baked-in absolute paths do not resolve")
    ap.add_argument("--allow-missing", action="store_true",
                    help="score a partial image set anyway (NOT comparable to published numbers)")
    ap.add_argument("--reanalyze", metavar="STEM",
                    help="recompute from results/stage_f/<STEM>.parquet on CPU")
    args = ap.parse_args()

    if args.reanalyze:
        reanalyze(args.reanalyze)
    elif args.person:
        run_person(args.config, args.model, args.person, args.limit, args.allow_missing,
                   args.images_root)
    elif args.axis:
        run_sweep(args.config, args.model, args.axis, args.limit, args.allow_missing,
                  args.images_root)
    elif args.generate:
        run_generate(args.config, args.model, args.limit, allow_missing=args.allow_missing,
                     images_root=args.images_root)
    else:
        ap.error("pick one of --person, --axis, --generate, --reanalyze")


if __name__ == "__main__":
    main()
