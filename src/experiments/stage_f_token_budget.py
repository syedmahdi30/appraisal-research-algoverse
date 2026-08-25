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
  python -m src.experiments.stage_f_token_budget --model <id> --text-only --bank minimal --reanalyze
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
from ..paths import PROCESSED_DIR, STAGE_F_DIR, ensure_dirs
from .common import load_config, run_stamp, save_json
from .shared.artifacts import (artifact_metadata, ensure_output_available, run_key_suffix,
                               token_budget_key, token_budget_metric_paths)
from .shared.hf_runtime import load_vlm, resize_long_side
from .shared.readouts import first_content_token_ids, model_readout, user_text
from .shared.reporting import (
    asymmetry_vs_floor,
    flip_override,
    image_discriminability,
    text_only_readouts,
    token_budget_trends,
)
from .shared.sampling import select_extreme_rows

_text_only_readouts = text_only_readouts
_trends = token_budget_trends


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


_key_suffix = run_key_suffix
slug = token_budget_key


# --------------------------------------------------------------------------- model dispatch
load_any = load_vlm


def _prep_image(img: Image.Image, max_side: int | None) -> Image.Image:
    return resize_long_side(img, max_side)


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
        content.append({"type": "text", "text": user_text(context_sentence)})
    else:
        content = ([{"type": "image"}] if image is not None else [])
        content.append({"type": "text", "text": user_text(context_sentence)})
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


def _analyze(df, model_name, tokens: dict, max_side, multi=None, n_skipped=0) -> dict:
    return artifact_metadata(
        model=model_name, max_side=max_side,
        read_out="behavioral_valence", n_images=int(df["image_path"].nunique()) if len(df) else 0,
        n_rows=int(len(df)), n_skipped=n_skipped,
        image_tokens=tokens,
        image_discriminability=image_discriminability(df) if len(df) else {},
        asymmetry_vs_floor=asymmetry_vs_floor(df) if len(df) else {},
        flip_override=flip_override(df) if len(df) else {},
        tokenization_multi_token=multi or {},
    )


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
    ensure_output_available(
        out_pq,
        force,
        f"{out_pq} already exists — refusing to overwrite a completed run. Pass --force to "
        f"replace it, or change --max-side / --model so the run gets its own key.",
    )

    n_images = limit_override or int(cfg.get("n_images", 150))
    frame = pd.read_parquet(PROCESSED_DIR / "emotic_test.parquet").reset_index(drop=True)
    sel = select_extreme_rows(frame, n_images)
    model, processor, family = load_vlm(model_name, max_side)
    tok_ids = first_content_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    tokens, rows, n_skip = None, [], 0
    for _, r in tqdm(list(sel.iterrows()), desc=f"token-budget {key}"):
        try:
            img = resize_long_side(Image.open(r["image_path"]).convert("RGB"), max_side)
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        if tokens is None:            # measured on the first real image, at the resolution actually used
            tokens = count_image_tokens(processor, img, family, style)
        for cond, cid, sentence in _conditions(bank):
            val, lp = model_readout(model, build_inputs(processor, img, sentence, family, style), tok_ids)
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
    return token_budget_metric_paths(STAGE_F_DIR, model_name, style, bank)


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
    ensure_output_available(out_pq, force, f"{out_pq} already exists — pass --force to replace it.")

    model, processor, family = load_vlm(model_name, None)
    tok_ids = first_content_token_ids(processor)
    multi = {w: r for w, r in verify_label_tokenization(processor.tokenizer).items()
             if not r["single_token"]}

    rows = []
    for cond, cid, sentence in _conditions(bank):
        val, lp = model_readout(model, build_inputs(processor, None, sentence, family, style), tok_ids)  # image=None
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

    metrics = artifact_metadata(
        model=model_name, bank=bank,
        keyed_by="model + bank (no image in this pass; --max-side cannot affect it)",
        neutral_baseline=neu, none_baseline=none_v, pos_effect=pe, neg_effect=ne,
        pos_raw=pr, neg_raw=nr, text_only_ratio_vs_neutral=text_ratio,
        text_only_ratio_raw=raw_ratio, reference_ratio=ref,
        per_base_run=per_base, tokenization_multi_token=multi,
    )
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


def reanalyze_text_only(model_name: str, style: str = "chat", bank: str = "full") -> dict:
    """Recompute a text-only control from its saved parquet (CPU, no model load).

    Exists because the GPU pass reports only the bounded readout, and on a saturating model that
    number is uninformative --- see `_text_only_readouts`. This rereads the same parquet, adds the
    unbounded margin, and rewrites the metrics file in place.
    """
    ensure_dirs()
    key = slug(model_name, None, style, bank)
    pq = STAGE_F_DIR / f"text_only_{key}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"{pq} missing — run --text-only first.")
    df = pd.read_parquet(pq)
    r = text_only_readouts(df)

    mpath = STAGE_F_DIR / f"text_only_{key}_metrics.json"
    m = json.loads(mpath.read_text()) if mpath.exists() else {"model": model_name, "bank": bank}
    m["readouts"] = r
    m["reanalyzed"] = run_stamp()
    save_json(m, mpath)

    b, u = r["bounded_valence"], r["unbounded_margin"]
    print(f"\nText-only readouts [{model_name}] bank={bank} — {r['n_rows']} rows")
    print(f"  bounded valence saturated on {r['saturation_frac']:.0%} of rows"
          + ("   <- the bounded ratio below is an artifact; read the margin instead"
             if r["saturation_frac"] >= 0.5 else ""))
    for name, x in (("bounded valence", b), ("unbounded margin", u)):
        print(f"  {name:17s} pos {x['pos_raw']:+8.3f}  neg {x['neg_raw']:+8.3f}  neutral "
              f"{x['neutral_baseline']:+8.3f}")
        print(f"  {'':17s} raw |neg|/|pos| = {x['ratio_raw']:.2f}   "
              f"vs-neutral |neg|/|pos| = {x['ratio_vs_neutral']:.2f}")
    print("  reading these: >1 means the negative sentences are stronger in isolation, which is the "
          "stimulus")
    print("  confound the control tests for; ~1 means matched; <1 means the positives are stronger.")
    print("  Neither ratio is unconditionally the right one, so both confounds are stated:")
    if r["saturation_frac"] >= 0.5:
        print(f"    - the bounded ratios are artifacts here ({r['saturation_frac']:.0%} of rows sit on "
              f"a bound); use the margin")
    if abs(b["neutral_baseline"]) >= 0.2:
        print(f"    - the neutral baseline is off-centre ({b['neutral_baseline']:+.2f} of a possible "
              f"+/-1), so with no image the model")
        print(f"      already leans "
              f"{'negative' if b['neutral_baseline'] < 0 else 'positive'} and the vs-neutral ratios "
              f"understate that side. Prefer the raw ones.")
    if r["saturation_frac"] < 0.5 and abs(b["neutral_baseline"]) < 0.2:
        print("    - neither applies: the scale is unsaturated and the neutral baseline is near "
              "centre, so")
        print("      the bounded vs-neutral ratio is the directly comparable one.")
    print(f"  data -> {mpath}")
    return m


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
    out = artifact_metadata(n_runs=len(tab), rows=tab.to_dict(orient="records"))

    out.update(token_budget_trends(tab))
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
        img = resize_long_side(img, max_side)
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
    ap.add_argument("--reanalyze", action="store_true",
                    help="recompute from the saved parquet (CPU). With --text-only, recomputes the "
                         "stimulus control on both the bounded and the unbounded scale.")
    ap.add_argument("--aggregate", action="store_true", help="build the cross-run trend table (CPU)")
    a = ap.parse_args()
    if a.show_prompt:
        show_prompt(a.model, a.max_side)
    elif a.aggregate:
        aggregate()
    elif a.text_only:
        if a.max_side:
            print("  [note] --max-side ignored: the text-only control has no image.")
        if a.reanalyze:
            reanalyze_text_only(a.model, style=a.prompt_style, bank=a.bank)
        else:
            run_text_only(a.config, a.model, force=a.force, style=a.prompt_style, bank=a.bank)
    elif a.reanalyze:
        reanalyze(a.model, a.max_side, bank=a.bank)
    else:
        run_base(a.config, a.model, a.max_side, limit_override=a.limit, force=a.force,
                 style=a.prompt_style, bank=a.bank)


if __name__ == "__main__":
    main()
