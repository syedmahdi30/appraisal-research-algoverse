"""Stage F — does the VISUAL TOKEN BUDGET set a model's resistance to textual override?

The paper's boundary claim is currently categorical and rests on one null: Gemma-3 and Qwen3-VL show
the negativity asymmetry, LLaVA-1.5 does not, and the difference is attributed to "image-anchoring".
But LLaVA-1.5 differs from the other two on many axes at once (pooled vs unpooled patches, frozen vs
trained tower, projector design, language backbone, model age, instruction-tuning quality), so the
attribution is not identified. This module tests a SPECIFIC, MEASURABLE variable:

    how many token positions does the image occupy, relative to the text?

Hypothesis: the more of the context the image holds, the harder it is for a text sentence to override
it. Two experiments, both driven from here:

  (A) CROSS-MODEL — run the same conflict battery on several designs and record, per run, the actual
      image-token count alongside the override gap. Turns "these two yes, that one no" into a trend
      over a measured architectural quantity.

  (B) WITHIN-MODEL (the causal test) — on a native dynamic-resolution model (Qwen3-VL, LLaVA-NeXT),
      the image-token count is an INPUT knob: feed the same photo at several resolutions and the token
      budget changes while the weights, the prompt, the context bank and the scoring stay identical.
      If the override gap moves with the budget under fixed weights, the cross-model trend is causal
      rather than a correlation across confounded checkpoints.

  CONFOUND, and the control for it. Lowering resolution also removes visual INFORMATION, so a moving
  override gap could just mean "blurrier image, less confident model". Every run therefore also
  reports `image_discriminability`: how well the model separates the positive- from the negative-
  valence photos with NO context at all (mean gap + AUC over the no-context rows). If discriminability
  is flat across resolutions while the override gap moves, the token budget is doing the work; if they
  degrade together, the two are confounded and the run must be reported as such.

PROVENANCE. Every run writes to a path keyed by (model slug, token-budget tag) — never a fixed
filename. An earlier fixed-path runner was invoked repeatedly with different flags and each run
overwrote its predecessor, permanently losing three published numbers; `--force` is required to
overwrite an existing parquet.

  python -m src.experiments.stage_f_token_budget --model Qwen/Qwen3-VL-8B-Instruct --max-side 896
  python -m src.experiments.stage_f_token_budget --model llava-hf/llava-v1.6-mistral-7b-hf
  python -m src.experiments.stage_f_token_budget --model <id> --text-only   # stimulus control
  python -m src.experiments.stage_f_token_budget --model <id> --text-only --bank minimal
  python -m src.experiments.stage_f_token_budget --aggregate      # CPU: build the trend table
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      build_conditions,
                                      TEXT_CODE)
from ..data.labels import EMOTION_LABELS, verify_label_tokenization
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, load_config, run_stamp, save_json
from .stage_f_qwen import (_user_text, emotion_token_ids, readout, select_extreme_images)


def _conditions(bank: str = "full"):
    """Context bank for the base pass, delegated to the shared builder.

    Previously this duplicated the full bank inline, which meant the raw-HF runner could not reproduce
    the minimal-pair control at all --- the only runner that could was the bridge-based
    `stage_f_conflict`, so the minimal-pair numbers had no raw-HF counterpart. Delegating to
    `build_conditions` keeps the ids identical across stacks (`p{i}`/`n{i}`/`z{i}` for the full bank,
    a SHARED `mp{i}` on both members of each minimal pair), which is what lets
    `analyze_stage_f._minimal_pair_asymmetry` recover the within-item flip by grouping on
    `context_id`.
    """
    return build_conditions(bank)


def _key_suffix(style: str = "chat", bank: str = "full") -> str:
    """The style/bank tail of a run key, without the model or budget parts.

    Split out of `slug` so that code needing to *glob* across token budgets can rebuild the tail
    exactly. The budget tag sits before this tail, so `f"{slug(...)}_px*"` looks for a filename that
    can never exist; `_base_runs_for` needs the pieces, not the finished key.
    """
    s = ""
    if style != "chat":
        s += f"_{style}"
    if bank != "full":
        s += f"_{bank}"
    return s


def slug(model_name: str, max_side: int | None, style: str = "chat", bank: str = "full") -> str:
    """Filesystem-safe run key: model + token-budget tag + prompt style + bank, so runs never collide.

    `bank` is part of the key for the same reason the others are: a minimal-pair run and a full-bank
    run of the same model are different experiments, and a shared path would silently overwrite one
    with the other --- the failure mode that destroyed three published numbers before per-run paths
    were introduced.
    """
    s = re.sub(r"[^a-z0-9]+", "-", model_name.lower().split("/")[-1]).strip("-")
    if max_side:
        s = f"{s}_px{max_side}"
    return s + _key_suffix(style, bank)


# --------------------------------------------------------------------------- model dispatch
def load_any(model_name: str, max_side: int | None = None):
    """Load a VLM + processor by family. Returns (model, processor, family).

    `max_side` is threaded into the Qwen processor as a pixel budget where the API supports it; for
    every family the image is ALSO resized before processing (see `_prep_image`), which is the
    family-agnostic way to move the token budget on a dynamic-resolution model.
    """
    from transformers import AutoProcessor
    lname = model_name.lower()
    if "qwen" in lname:
        if "qwen3" in lname:
            from transformers import Qwen3VLForConditionalGeneration as Cls
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        family = "qwen"
        proc_kwargs = {"max_pixels": max_side * max_side} if max_side else {}
    else:
        try:
            from transformers import AutoModelForImageTextToText as Cls  # llava 1.5/NeXT, paligemma, idefics
        except ImportError:
            from transformers import LlavaForConditionalGeneration as Cls
        family = "llava"
        proc_kwargs = {}
    model = Cls.from_pretrained(model_name, torch_dtype="auto", device_map="auto").eval()
    try:
        processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)
    except (TypeError, ValueError):
        # Not every processor version accepts a pixel budget; `_prep_image` resizes anyway, so the
        # token budget still moves. Fail soft rather than lose a GPU session to a kwarg.
        print(f"  [warn] processor rejected {list(proc_kwargs)}; relying on image resize alone")
        processor = AutoProcessor.from_pretrained(model_name)
    return model, processor, family


def _prep_image(img: Image.Image, max_side: int | None) -> Image.Image:
    """Downscale so the long side is <= max_side, preserving aspect ratio (no-op if already smaller).

    On a native dynamic-resolution model this is what changes the image-token count; on a fixed-grid
    model the count is unchanged and only the visual detail drops — which is exactly why
    `image_discriminability` is reported next to every run.
    """
    if not max_side or max(img.size) <= max_side:
        return img
    w, h = img.size
    scale = max_side / float(max(w, h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


def build_inputs(processor, image, context_sentence, family: str, style: str = "chat"):
    """Inputs for one image + context; image=None gives the text-only (image-ablated) form.

    `style` selects the prompt construction, which is NOT a cosmetic choice:
      * "chat"   — `processor.apply_chat_template(...)`, the portable path used for every raw-HF
                   model (Qwen, LLaVA 1.5 / NeXT). LLaVA-1.5 reproduces its published numbers exactly
                   through it.
      * "legacy" — the hand-written Gemma scaffold in `conflict_contexts.context_prompt`, which is
                   what the PUBLISHED Gemma run fed to TransformerBridge. Gemma is the one model whose
                   published numbers came from the bridge path rather than this one, and it does not
                   reproduce under "chat" (+12% vs the published +65%), so this flag exists to isolate
                   the template from the model stack. Gemma-specific: the scaffold hardcodes
                   `<start_of_turn>` / `<start_of_image>`.
    """
    if style == "legacy":
        from ..data.conflict_contexts import context_prompt
        text = context_prompt(context_sentence)
        imgs = [image] if image is not None else None
        return processor(text=text, images=imgs, return_tensors="pt")
    if family == "qwen":
        content = ([{"type": "image", "image": image}] if image is not None else [])
        content.append({"type": "text", "text": _user_text(context_sentence)})
    else:
        content = ([{"type": "image"}] if image is not None else [])
        content.append({"type": "text", "text": _user_text(context_sentence)})
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                         tokenize=False, add_generation_prompt=True)
    imgs = [image] if image is not None else None
    kw = {"text": [text], "images": imgs, "padding": True} if family == "qwen" else \
         {"text": text, "images": imgs}
    return processor(return_tensors="pt", **kw)


def count_image_tokens(processor, image, family: str, style: str = "chat") -> dict:
    """Measure how many sequence positions the image actually occupies, empirically.

    Length of the prompt WITH the image minus the same prompt WITHOUT it. This is processor-level and
    family-agnostic, so it works wherever the placeholder is expanded at tokenization time. Some older
    processors leave a single unexpanded placeholder and expand inside the model instead; that shows up
    as a delta of ~1 and is flagged rather than silently reported as a one-token image.
    """
    with_img = build_inputs(processor, image, None, family, style)["input_ids"].shape[-1]
    without = build_inputs(processor, None, None, family, style)["input_ids"].shape[-1]
    delta = int(with_img - without)
    return {"image_tokens": delta, "prompt_tokens_with_image": int(with_img),
            "prompt_tokens_text_only": int(without),
            "image_token_fraction": delta / with_img if with_img else float("nan"),
            "expansion_ok": bool(delta > 8),
            "note": ("" if delta > 8 else
                     "placeholder appears UNEXPANDED at tokenization (delta<=8); the true image-token "
                     "count is applied inside the model — read it from the model's vision config "
                     "instead of this field.")}


# --------------------------------------------------------------------------- controls & metrics
def image_discriminability(df: pd.DataFrame) -> dict:
    """Can the model still tell the two image groups apart with NO context? The resolution control.

    Uses only the `none` rows, so it is a pure vision read: if this collapses at low resolution the
    image simply became unreadable, and any change in the override gap is confounded with visual
    quality rather than attributable to the token budget.
    """
    d = df[df["condition"] == "none"]
    pos = d[d["image_group"] == "positive"]["valence"].to_numpy(dtype=float)
    neg = d[d["image_group"] == "negative"]["valence"].to_numpy(dtype=float)
    if not len(pos) or not len(neg):
        return {}
    # AUC via the Mann-Whitney U identity (no sklearn dependency in the raw-HF envs).
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return {"mean_valence_positive_images": float(pos.mean()),
            "mean_valence_negative_images": float(neg.mean()),
            "discriminability_gap": float(pos.mean() - neg.mean()),
            "auc": float(auc), "n_pos": int(len(pos)), "n_neg": int(len(neg))}


def _analyze(df, model_name, tokens: dict, max_side, multi=None, n_skipped=0) -> dict:
    from .analyze_stage_f import _asymmetry_vs_floor, _flip_override
    return {"run": run_stamp(), "git": git_hash(), "model": model_name, "max_side": max_side,
            "read_out": "behavioral_valence", "n_images": int(df["image_path"].nunique()) if len(df) else 0,
            "n_rows": int(len(df)), "n_skipped": n_skipped,
            "image_tokens": tokens,
            "image_discriminability": image_discriminability(df) if len(df) else {},
            "asymmetry_vs_floor": _asymmetry_vs_floor(df) if len(df) else {},
            "flip_override": _flip_override(df) if len(df) else {},
            "tokenization_multi_token": multi or {}}


def _print(m: dict) -> None:
    t, d, f = m["image_tokens"], m["image_discriminability"], m["flip_override"]
    print(f"\nStage F token-budget [{m['model']}]  max_side={m['max_side']}  "
          f"{m['n_images']} images, {m['n_rows']} rows")
    print(f"  image tokens: {t['image_tokens']} of {t['prompt_tokens_with_image']} prompt positions "
          f"({t['image_token_fraction']:.1%})" + ("" if t["expansion_ok"] else "  [!] " + t["note"]))
    if d:
        print(f"  image discriminability (no context): gap {d['discriminability_gap']:+.3f}  "
              f"AUC {d['auc']:.3f}   <- must stay flat across resolutions for the budget reading")
    if f:
        print(f"  override: neg-ctx over positive image {f['neg_ctx_overrides_pos_img']:.0%}  vs  "
              f"pos-ctx over negative image {f['pos_ctx_overrides_neg_img']:.0%}  "
              f"(gap {f['dominance_gap']:+.0%}, CI [{f['dominance_gap_ci95'][0]:+.0%},"
              f"{f['dominance_gap_ci95'][1]:+.0%}])")


# --------------------------------------------------------------------------- base pass
def run_base(config_path: str, model_name: str, max_side: int | None,
             limit_override: int | None = None, force: bool = False, style: str = "chat",
             bank: str = "full") -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    key = slug(model_name, max_side, style, bank)
    out_pq = STAGE_F_DIR / f"conflict_{key}.parquet"
    if out_pq.exists() and not force:
        raise FileExistsError(
            f"{out_pq} already exists — refusing to overwrite a completed run. Pass --force to "
            f"replace it, or change --max-side / --model so the run gets its own key.")

    n_images = limit_override or int(cfg.get("n_images", 150))
    sel = select_extreme_images(n_images)
    model, processor, family = load_any(model_name, max_side)
    tok_ids = emotion_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    tokens, rows, n_skip = None, [], 0
    for _, r in tqdm(list(sel.iterrows()), desc=f"token-budget {key}"):
        try:
            img = _prep_image(Image.open(r["image_path"]).convert("RGB"), max_side)
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        if tokens is None:            # measured on the first real image, at the resolution actually used
            tokens = count_image_tokens(processor, img, family, style)
        for cond, cid, sentence in _conditions(bank):
            val, lp = readout(model, build_inputs(processor, img, sentence, family, style), tok_ids)
            rows.append({"image_path": r["image_path"], "image_valence": float(r["valence"]),
                         "image_group": r["image_group"], "condition": cond, "context_id": cid,
                         "context": sentence or "", "text_code": TEXT_CODE[cond],
                         "probe_readout": float("nan"),   # no probe off-Gemma; column kept for schema
                         "valence": val, **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})

    df = pd.DataFrame(rows)
    df.to_parquet(out_pq)
    metrics = _analyze(df, model_name, tokens or {}, max_side, multi=multi, n_skipped=n_skip)
    metrics["prompt_style"] = style
    metrics["bank"] = bank
    save_json(metrics, STAGE_F_DIR / f"conflict_{key}_metrics.json")
    _print(metrics)
    print(f"  data -> {out_pq}")
    return metrics


# --------------------------------------------------------------------------- text-only control
def _base_runs_for(model_name: str, style: str = "chat", bank: str = "full") -> list[Path]:
    """Every base-run metrics file for this model AND bank, across token budgets.

    Must match on bank: a minimal-pair text-only control compared against a full-bank base run would
    be comparing different sentences, which is precisely the confound the control exists to test.
    The glob is assembled from key parts because the budget tag precedes the style/bank tail.
    """
    core = re.sub(r"[^a-z0-9]+", "-", model_name.lower().split("/")[-1]).strip("-")
    tail = _key_suffix(style, bank)
    exact = STAGE_F_DIR / f"conflict_{core}{tail}_metrics.json"
    runs = [exact] if exact.exists() else []
    # The glob alone is not enough: for the full bank `tail` is empty, so `_px*_metrics.json` also
    # swallows `_px448_minimal_metrics.json`. Anchor the budget tag to digits and the tail to the end.
    pat = re.compile(rf"^conflict_{re.escape(core)}_px\d+{re.escape(tail)}_metrics\.json$")
    return runs + sorted(p for p in STAGE_F_DIR.glob(f"conflict_{core}_px*_metrics.json")
                         if pat.match(p.name))


def _amplification(img_ratio: float | None, ref: float) -> str:
    """Label the image-conditioned vs text-only ratio comparison (the cross-modal amplification test)."""
    if img_ratio is None or not np.isfinite(ref) or not np.isfinite(img_ratio):
        return "no base run to compare"
    if img_ratio > 1.25 * ref:
        return "CROSS-MODAL amplification (image inflates the ratio)"
    if abs(img_ratio - ref) <= 0.25 * ref:
        return "STIMULUS confound (ratios match)"
    return "image dampens (reversed)"


def run_text_only(config_path: str, model_name: str, force: bool = False, style: str = "chat",
                  bank: str = "full") -> dict:
    """The image-ablated stimulus control: every context sentence with NO image.

    Isolates whether the negativity asymmetry is a property of the SENTENCES (the banks are simply
    unbalanced in strength) or genuinely cross-modal (it needs the image). Keyed by MODEL AND BANK,
    with no token-budget tag: there is no image in this pass, so `--max-side` cannot affect it and
    running it once per resolution would burn compute re-deriving identical numbers. The bank does
    matter, because it selects which sentences are being ablated --- the matched pairs and the varied
    set are different stimulus sets and can be imbalanced to different degrees.

    Because the text-only ratio is fixed per model while the image-conditioned ratio is measured per
    token budget, the comparison is reported against EVERY base run of this model — so cross-modal
    amplification can be read as a function of the visual token budget.
    """
    load_config(config_path)
    ensure_dirs()
    key = slug(model_name, None, style, bank)   # budget-free but bank-aware; see docstring
    out_pq = STAGE_F_DIR / f"text_only_{key}.parquet"
    if out_pq.exists() and not force:
        raise FileExistsError(f"{out_pq} already exists — pass --force to replace it.")

    model, processor, family = load_any(model_name, None)
    tok_ids = emotion_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    rows = []
    for cond, cid, sentence in _conditions(bank):
        val, lp = readout(model, build_inputs(processor, None, sentence, family, style), tok_ids)  # image=None
        rows.append({"condition": cond, "context_id": cid, "context": sentence or "",
                     "text_code": TEXT_CODE[cond], "valence": val, "argmax_emotion": max(lp, key=lp.get),
                     **{f"lp_{w}": lp[w] for w in EMOTION_LABELS}})
    df = pd.DataFrame(rows)
    df.to_parquet(out_pq)

    neu = float(df[df["condition"] == "neutral"]["valence"].mean())
    none_v = float(df[df["condition"] == "none"]["valence"].mean())
    pe = float((df[df["condition"] == "positive"]["valence"] - neu).mean())
    ne = float((df[df["condition"] == "negative"]["valence"] - neu).mean())
    # raw ratio (vs 0, not vs a possibly-floored neutral) — the robust reference when a model pins
    # its no-information baseline to the scale bound, as Qwen does.
    pr = float(df[df["condition"] == "positive"]["valence"].mean())
    nr = float(df[df["condition"] == "negative"]["valence"].mean())
    raw_ratio = abs(nr) / abs(pr) if pr else float("nan")
    text_ratio = abs(ne) / abs(pe) if pe else float("nan")
    ref = raw_ratio if np.isfinite(raw_ratio) else text_ratio

    per_base = []
    for bp in _base_runs_for(model_name, style, bank):
        j = json.loads(bp.read_text())
        a = j.get("asymmetry_vs_floor", {})
        dn, dp = a.get("drop_pos_img_neg_ctx"), a.get("congruent_pos_img_pos_ctx")
        img_ratio = abs(dn) / abs(dp) if dn is not None and dp else None
        per_base.append({"base_run": bp.name, "max_side": j.get("max_side"),
                         "image_tokens": j.get("image_tokens", {}).get("image_tokens"),
                         "image_conditioned_ratio": img_ratio,
                         "verdict": _amplification(img_ratio, ref)})

    metrics = {"run": run_stamp(), "git": git_hash(), "model": model_name, "bank": bank,
               "keyed_by": "model + bank (no image in this pass; --max-side cannot affect it)",
               "neutral_baseline": neu, "none_baseline": none_v, "pos_effect": pe, "neg_effect": ne,
               "pos_raw": pr, "neg_raw": nr, "text_only_ratio_vs_neutral": text_ratio,
               "text_only_ratio_raw": raw_ratio, "reference_ratio": ref,
               "per_base_run": per_base, "tokenization_multi_token": multi}
    save_json(metrics, STAGE_F_DIR / f"text_only_{key}_metrics.json")

    print(f"\nStage F text-only [{model_name}] bank={bank} — {len(rows)} forwards (no images).")
    if multi:
        print(f"  [!] multi-token labels (first sub-token scored): {list(multi)}")
    for _, r in df.iterrows():
        print(f"    {r['context_id']:5s} {r['valence']:+6.3f}  {r['argmax_emotion']:9s}  "
              f"\"{r['context'][:42]}\"")
    print(f"  baselines: neutral {neu:+.3f}  none {none_v:+.3f}")
    print(f"  vs-neutral: pos {pe:+.3f}  neg {ne:+.3f}  |neg|/|pos| = {text_ratio:.2f}")
    print(f"  RAW (vs 0): pos {pr:+.3f}  neg {nr:+.3f}  |neg|/|pos| = {raw_ratio:.2f}  <- reference")
    if per_base:
        print("  cross-modal amplification vs each base run:")
        for b in per_base:
            ir = b["image_conditioned_ratio"]
            print(f"    max_side={str(b['max_side']):>5s}  img_tokens={str(b['image_tokens']):>5s}  "
                  f"image-conditioned {('%.2f' % ir) if ir else '  n/a'}  vs text-only {ref:.2f}"
                  f"  -> {b['verdict']}")
    else:
        print("  (no base run for this model yet — run the base pass to enable the comparison)")
    print(f"  data -> {out_pq}")
    return metrics


def reanalyze(model_name: str, max_side: int | None, bank: str = "full") -> dict:
    """Recompute metrics from a saved parquet (CPU, no model load).

    Takes `bank` because it is part of the run key: without it, `--reanalyze --bank minimal` would
    silently recompute the full-bank parquet and overwrite the full-bank metrics.
    """
    ensure_dirs()
    key = slug(model_name, max_side, bank=bank)
    pq = STAGE_F_DIR / f"conflict_{key}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run the base pass first.")
    mpath = STAGE_F_DIR / f"conflict_{key}_metrics.json"
    tokens = json.loads(mpath.read_text()).get("image_tokens", {}) if mpath.exists() else {}
    m = _analyze(pd.read_parquet(pq), model_name, tokens, max_side)
    save_json(m, mpath)
    _print(m)
    return m


# --------------------------------------------------------------------------- aggregation
def _trends(tab: pd.DataFrame) -> dict:
    """Within-model and cross-model evidence on the token-budget hypothesis, kept strictly separate.

    A single correlation over every row is worse than useless here, for three reasons this function
    exists to avoid:

      1. POOLING. A within-model resolution sweep (same weights, budget varied) and a cross-model
         comparison (everything varies at once) answer different questions. Pooling them lets three
         clustered points from one model plus one distant point from another produce r = -0.99 that
         means nothing.
      2. IGNORED UNCERTAINTY. Correlation uses point estimates only. Four gaps that sit inside
         mutually overlapping 95% CIs are a flat line, however neatly ordered they happen to be.
      3. SILENT EXCLUSION. Runs from the older fixed-path runner carry no image-token count and drop
         out of any correlation without comment — and those were exactly the models that break the
         pattern. Missing rows are now named, not dropped quietly.

    Within-model verdict is CI-based: if every sweep CI shares a common intersection, no effect of the
    budget is detectable regardless of the ordering of the point estimates.
    """
    out: dict = {}
    have = tab.dropna(subset=["image_tokens", "override_gap"])
    missing = tab[tab["image_tokens"].isna()]
    if len(missing):
        out["excluded_missing_image_tokens"] = [
            {"model": r["model"], "source": r["source"], "override_gap": r["override_gap"]}
            for _, r in missing.iterrows()]

    # ---- within-model: one entry per model that was swept over >=2 distinct budgets
    within = []
    for model, g in have.groupby("model"):
        if g["image_tokens"].nunique() < 2:
            continue
        lo, hi = float(g["ci_lo"].max()), float(g["ci_hi"].min())   # common intersection of all CIs
        flat = lo <= hi
        within.append({
            "model": model, "n_runs": int(len(g)),
            "tokens_min": float(g["image_tokens"].min()), "tokens_max": float(g["image_tokens"].max()),
            "fold_range": float(g["image_tokens"].max() / g["image_tokens"].min()),
            "gap_min": float(g["override_gap"].min()), "gap_max": float(g["override_gap"].max()),
            "all_cis_overlap": bool(flat),
            "discriminability_auc_range": [float(g["auc"].min()), float(g["auc"].max())]
                                          if g["auc"].notna().any() else None,
            "verdict": ("FLAT — every CI overlaps, so no effect of the token budget is detectable "
                        "over this range" if flat else
                        "MOVES — at least two CIs are disjoint across the budget range")})
    if within:
        out["within_model"] = within

    # ---- cross-model: exactly one representative run per model (the largest budget measured)
    reps = have.sort_values("image_tokens").groupby("model", as_index=False).last()
    if len(reps) >= 3:
        t, gp = reps["image_tokens"].to_numpy(float), reps["override_gap"].to_numpy(float)
        r = float(np.corrcoef(t, gp)[0, 1])
        loo = [float(np.corrcoef(np.delete(t, i), np.delete(gp, i))[0, 1]) for i in range(len(t))]
        out["cross_model"] = {
            "n_models": int(len(reps)), "models": reps["model"].tolist(),
            "pearson_tokens_vs_gap": r, "leave_one_out_range": [min(loo), max(loo)],
            "caveat": (f"n={len(reps)} models: a correlation this small is descriptive only, ignores "
                       f"every CI, and confounds the budget with everything else that differs "
                       f"between checkpoints."
                       + (" Runs are missing image-token counts (see excluded_missing_image_tokens), "
                          "so this omits models that may break the pattern."
                          if "excluded_missing_image_tokens" in out else ""))}
    elif len(reps):
        out["cross_model"] = {"n_models": int(len(reps)),
                              "note": "fewer than 3 models with measured image tokens — no trend reported"}
    return out


def _print_trends(t: dict) -> None:
    for w in t.get("within_model", []):
        auc = w["discriminability_auc_range"]
        print(f"\n  WITHIN {w['model']}  ({w['n_runs']} runs, {w['tokens_min']:.0f}->{w['tokens_max']:.0f} "
              f"tokens = {w['fold_range']:.1f}x)")
        print(f"    gap {w['gap_min']:+.0%}..{w['gap_max']:+.0%}   {w['verdict']}")
        if auc:
            print(f"    discriminability AUC {auc[0]:.3f}..{auc[1]:.3f} "
                  f"({'held — clean test' if auc[0] > 0.9 else 'DEGRADED — budget confounded with image quality'})")
    c = t.get("cross_model")
    if c and "pearson_tokens_vs_gap" in c:
        print(f"\n  CROSS-MODEL  n={c['n_models']} models: r = {c['pearson_tokens_vs_gap']:+.3f}  "
              f"(leave-one-out {c['leave_one_out_range'][0]:+.3f}..{c['leave_one_out_range'][1]:+.3f})")
        print(f"    {c['caveat']}")
    elif c:
        print(f"\n  CROSS-MODEL  {c['note']}")
    for e in t.get("excluded_missing_image_tokens", []):
        print(f"  [!] excluded (no image-token count): {e['model']} gap {e['override_gap']:+.0%} "
              f"-- re-run with this runner to include it")


def aggregate() -> dict:
    """Collect every token-budget run into the trend table: image tokens vs override gap.

    Also pulls in the three published base runs (Gemma / Qwen / LLaVA-1.5) when their metrics files
    are present, so the cross-model panel and the within-model resolution sweep sit in one table.
    """
    ensure_dirs()
    rows = []
    for mp in sorted(STAGE_F_DIR.glob("conflict_*_metrics.json")):
        try:
            j = json.loads(mp.read_text())
        except json.JSONDecodeError:
            continue
        f, t, d = j.get("flip_override", {}), j.get("image_tokens", {}), j.get("image_discriminability", {})
        if not f:
            continue
        # the model's text-only reference ratio, if that control has been run. Must be looked up at
        # the SAME bank: reading the full-bank control for a minimal-bank run would compare a ratio
        # to sentences it was never measured on. Older metrics files predate the field, so fall back
        # to the filename, which is the run key and always carries the suffix.
        bank = j.get("bank") or ("minimal" if mp.name.endswith("_minimal_metrics.json") else "full")
        tpath = STAGE_F_DIR / (
            f"text_only_{slug(j.get('model', ''), None, j.get('prompt_style', 'chat'), bank)}"
            "_metrics.json")
        text_ratio = None
        if tpath.exists():
            try:
                text_ratio = json.loads(tpath.read_text()).get("reference_ratio")
            except json.JSONDecodeError:
                pass
        rows.append({"source": mp.name, "model": j.get("model", "?"), "bank": bank,
                     "max_side": j.get("max_side"),
                     "image_tokens": t.get("image_tokens"),
                     "image_token_fraction": t.get("image_token_fraction"),
                     "discriminability_gap": d.get("discriminability_gap"), "auc": d.get("auc"),
                     "text_only_ratio": text_ratio,
                     "override_gap": f.get("dominance_gap"),
                     "ci_lo": f.get("dominance_gap_ci95", [None, None])[0],
                     "ci_hi": f.get("dominance_gap_ci95", [None, None])[1]})
    cols = ["source", "model", "bank", "max_side", "image_tokens", "image_token_fraction",
            "discriminability_gap", "auc", "text_only_ratio", "override_gap", "ci_lo", "ci_hi"]
    tab = pd.DataFrame(rows, columns=cols)
    if len(tab):
        tab = tab.sort_values(["model", "image_tokens"], na_position="last")
    out = {"run": run_stamp(), "git": git_hash(), "n_runs": len(tab),
           "rows": tab.to_dict(orient="records")}

    out.update(_trends(tab))
    save_json(out, STAGE_F_DIR / "token_budget_trend.json")
    if len(tab):
        print(tab.to_string(index=False))
    _print_trends(out)
    print(f"\n  data -> {STAGE_F_DIR/'token_budget_trend.json'}")
    return out


def show_prompt(model_name: str, max_side: int | None = None) -> dict:
    """Print the exact prompt each style produces, side by side. Processor only — no model, no GPU.

    The fast first step when a model fails to reproduce its published numbers: it shows whether the
    two paths differ in scaffolding, image-token count, or total length before spending a GPU session
    on a full 2,250-forward run.
    """
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    family = "qwen" if "qwen" in model_name.lower() else "llava"
    img = Image.new("RGB", (640, 480), (127, 127, 127))
    if max_side:
        img = _prep_image(img, max_side)
    sentence = NEGATIVE_CONTEXTS[0]

    out = {"model": model_name, "family": family, "styles": {}}
    for style in ("chat", "legacy"):
        try:
            enc = build_inputs(processor, img, sentence, family, style)
            text = processor.tokenizer.decode(enc["input_ids"][0])
            tok = count_image_tokens(processor, img, family, style)
            rec = {"total_prompt_tokens": int(enc["input_ids"].shape[-1]),
                   "image_tokens": tok["image_tokens"],
                   "text_scaffold_tokens": int(enc["input_ids"].shape[-1]) - tok["image_tokens"],
                   "decoded": text}
        except Exception as e:                       # legacy is Gemma-specific; other models will fail
            rec = {"error": f"{type(e).__name__}: {e}"}
        out["styles"][style] = rec
        print(f"\n=== style={style} ===")
        if "error" in rec:
            print(f"  unavailable: {rec['error']}")
            continue
        print(f"  total {rec['total_prompt_tokens']} tokens = {rec['image_tokens']} image "
              f"+ {rec['text_scaffold_tokens']} text scaffold")
        shown = re.sub(r"(<image_soft_token>|<start_of_image>)(\1)+", r"\1...[image]...",
                       rec["decoded"])
        print(f"  {shown[:600]}")

    a, b = out["styles"]["chat"], out["styles"]["legacy"]
    if "error" not in a and "error" not in b:
        print(f"\n  DIFF  image tokens {a['image_tokens']} vs {b['image_tokens']}   |   "
              f"text scaffold {a['text_scaffold_tokens']} vs {b['text_scaffold_tokens']}   |   "
              f"total {a['total_prompt_tokens']} vs {b['total_prompt_tokens']}")
        print("  If the scaffolds differ, re-run the base pass with --prompt-style legacy to test "
              "whether the template explains a reproduction gap.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — visual token budget vs textual override")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max-side", type=int, default=None,
                    help="downscale images so the long side <= N (moves the token budget on "
                         "dynamic-resolution models); omit for the model's native handling")
    ap.add_argument("--limit", type=int, default=None, help="EMOTIC image count")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run for this key")
    ap.add_argument("--text-only", action="store_true",
                    help="image-ablated stimulus control (keyed by model and --bank; ignores "
                         "--max-side)")
    ap.add_argument("--bank", choices=("full", "minimal"), default="full",
                    help="context bank: 'full' (6 pos / 6 neg / 2 neutral) or 'minimal' (6 "
                         "token-matched valence-only pairs, the valence-vs-event-content control). "
                         "The bank is part of the run key, so the two never overwrite each other.")
    ap.add_argument("--prompt-style", choices=("chat", "legacy"), default="chat",
                    help="chat = processor.apply_chat_template (portable); legacy = the hand-written "
                         "Gemma scaffold used for the PUBLISHED Gemma run (Gemma only)")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print both prompt styles side by side and exit (processor only, no GPU)")
    ap.add_argument("--reanalyze", action="store_true", help="recompute from the saved parquet (CPU)")
    ap.add_argument("--aggregate", action="store_true", help="build the cross-run trend table (CPU)")
    a = ap.parse_args()
    if a.show_prompt:
        show_prompt(a.model, a.max_side)
    elif a.aggregate:
        aggregate()
    elif a.text_only:
        if a.max_side:
            print("  [note] --max-side ignored: the text-only control has no image.")
        run_text_only(a.config, a.model, force=a.force, style=a.prompt_style, bank=a.bank)
    elif a.reanalyze:
        reanalyze(a.model, a.max_side, bank=a.bank)
    else:
        run_base(a.config, a.model, a.max_side, limit_override=a.limit, force=a.force,
                 style=a.prompt_style, bank=a.bank)


if __name__ == "__main__":
    main()
