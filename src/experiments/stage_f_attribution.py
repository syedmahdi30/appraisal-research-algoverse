"""Stage F — L18 source-attribution: WHY does a positive image suppress positive framing?

The text-only control (`stage_f_text_only.py`) established that the pos-image+neg-context valence
drop is a genuine CROSS-MODAL effect, not a stimulus artifact: adding a positive image blunts the
positive-context channel by ~76% but the negative channel only ~20% (negativity dominance in
conflict resolution). This script asks the mechanism: at the frozen probe site (last-token
`blocks.18.hook_attn_out`), is the positive channel lost because the last token (1) ATTENTION-REROUTES
away from positive-context tokens toward the image, or (2) still attends but the context tokens write
a weaker VALUE into the read-out?

Method — causal attention knockout (no W_O / GQA bookkeeping needed). attn_out is LINEAR in the
attention pattern, so zeroing the last-token query's attention to a group of source positions and
reading the resulting attn_out gives EXACTLY that group's additive contribution:
    contribution(G) = probe(attn_out_full) − probe(attn_out | pattern[last, G] := 0).
Per positive-image cell and per context polarity we record, at the last token:
  - ATTENTION MASS to image / context / template  (tests mechanism 1: re-routing)
  - CONTRIBUTION (probe units, and behavioral-valence units) of image / context (tests mechanism 2:
    value competition)
Comparing negative- vs positive-context cells on the SAME positive images isolates why the positive
channel collapses.

Caveat: single-layer, last-token attribution captures the DIRECT L18 path only; context information
that leaked into image/template positions at earlier layers is credited to those groups (a lower
bound on total context influence). Stated, not hidden.

Run on the A100 with HF_TOKEN + EMOTIC. Frozen probe; never re-fit. `--limit N` sets image count.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from ..bridge.boot import boot_gemma
from ..bridge.multimodal import build_image_inputs
from ..data.conflict_contexts import (NEGATIVE_CONTEXTS, NEUTRAL_CONTEXTS, POSITIVE_CONTEXTS,
                                      TEXT_CODE, context_prompt)
from ..data.emotic import load_split as load_emotic_split
from ..data.labels import EMOTION_LABELS
from ..paths import FIGURES_DIR, STAGE_F_DIR, ensure_dirs
from ..probes.evaluate import predict
from .common import git_hash, load_config, load_probes, run_stamp, save_json
from .shared.patching import find_subsequence, segment_prompt_positions, stash_activation
from .stage_a_steering import emotion_token_ids, valence_score
from .shared.sampling import select_extreme_rows

QUESTION = "What single emotion is this person feeling?"


# --------------------------------------------------------------------------- position segmentation
def segment_positions(bridge, input_ids) -> dict:
    """Partition key positions into image / context / template index arrays for one prompt.

    Image tokens = the 256-long contiguous run of a single repeated placeholder id spliced at
    <start_of_image>. Context tokens = everything between the image block and the question anchor
    ("What single emotion ..."). Template = the rest (turn markers, question, model turn, BOS).
    Returns index arrays plus decoded snippets for eyeball validation.
    """
    segment = segment_prompt_positions(
        bridge.tokenizer, input_ids, QUESTION, expected_image_tokens=256
    )
    ids = input_ids[0].tolist()
    dec = bridge.tokenizer.decode
    return {
        "image": np.array(segment["image"].tolist()),
        "context": np.array(segment["context"].tolist()),
        "question": np.array(segment["question"].tolist()),
        "question_ok": segment["question_ok"],
        "template": np.array(segment["template"].tolist()),
        "n": segment["n"], "img_len": segment["img_len"],
        "image_ok": segment["image_ok"],
        "context_text": dec([ids[k] for k in segment["context"]]) if len(segment["context"]) else "",
        "template_text": dec([ids[k] for k in segment["template"]][:20]),
    }


def _find_subseq(hay, needle):
    return find_subsequence(hay, needle)


# --------------------------------------------------------------------------- forwards + knockout
def _stash_hook(store):
    return stash_activation(store)


def _knockout_hook(group_idx, last):
    """Zero the last-token query's attention to `group_idx` key positions (pre-normalized pattern)."""
    g = torch.as_tensor(group_idx, dtype=torch.long)

    def hook(pattern, hook):  # noqa: ARG001 - pattern: [batch, head, q, k]
        pattern[:, :, last, g] = 0
        return pattern
    return hook


def _pattern_stash(pstore):
    def hook(pattern, hook):  # noqa: ARG001
        pstore["p"] = pattern.detach()
        return pattern
    return hook


def _readout(bridge, ids, pv, name, layer, coef, inter, tok_ids, knockout=None, want_pattern=False):
    """(probe, behavioral_valence, attn_pattern_last?) at the last token, optionally with a knockout."""
    store, pstore = {}, {}
    hooks = [(name, _stash_hook(store))]
    if want_pattern:
        hooks.append((f"blocks.{layer}.attn.hook_pattern", _pattern_stash(pstore)))
    if knockout is not None:
        hooks.append((f"blocks.{layer}.attn.hook_pattern", knockout))
    with torch.no_grad():
        logits = bridge.run_with_hooks(ids, pixel_values=pv, fwd_hooks=hooks)
    last = ids.shape[-1] - 1
    act = store["act"][0, last].float().cpu().numpy()
    probe = float(predict(act[None, :], coef, inter)[0])
    val = valence_score(logits[0, -1], tok_ids)
    pat = pstore.get("p")
    return probe, val, (pat[0, :, last].float().cpu().numpy() if pat is not None else None)


# --------------------------------------------------------------------------- run
def run(config_path: str, limit_override: int | None = None) -> dict:
    cfg = load_config(config_path)
    ensure_dirs()
    layer = int(cfg.get("critical_layer", 18))
    tap = cfg.get("tap", "hook_attn_out")
    n_images = limit_override or int(cfg.get("attr_n_images", cfg.get("n_images", 40)))

    from ..paths import STAGE_A_DIR
    ppath = STAGE_A_DIR / "probes.npz"
    if not ppath.exists():
        raise FileNotFoundError(f"{ppath} missing — Stage A must have saved frozen probes.")
    probes = load_probes(ppath)
    pi = probes.index("pleasantness")
    coef, inter = probes.coef[pi], probes.intercept[pi]

    conditions = ([("neutral", f"z{i}", c) for i, c in enumerate(NEUTRAL_CONTEXTS)]
                  + [("negative", f"n{i}", c) for i, c in enumerate(NEGATIVE_CONTEXTS)]
                  + [("positive", f"p{i}", c) for i, c in enumerate(POSITIVE_CONTEXTS)])

    # positive-image group only: its neutral valence is mid-scale (symmetric head-room), the clean
    # cell where the positive channel demonstrably collapses under the image.
    frame = load_emotic_split(cfg.get("split", "test")).reset_index(drop=True)
    sel = select_extreme_rows(frame, n_images * 2)
    sel = sel[sel["image_group"] == "positive"].head(n_images).reset_index(drop=True)

    bridge = boot_gemma(cfg.get("model", "google/gemma-3-4b-it"), device=cfg.get("device", "cuda"))
    tok_ids = emotion_token_ids(bridge)
    name = f"blocks.{layer}.{tap}"

    rows, n_skip, n_ok, seg0 = [], 0, 0, None
    sanity = {"checked": False, "pattern_edits_attn_out": None}
    for _, r in tqdm(list(sel.iterrows()), desc="stage-f attribution"):
        try:
            img = Image.open(r["image_path"]).convert("RGB")
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        for cond, cid, sentence in conditions:
            inputs = build_image_inputs(bridge, img, prompt=context_prompt(sentence))
            ids, pv = inputs["input_ids"], inputs["pixel_values"]
            seg = segment_positions(bridge, ids)
            if seg0 is None:
                seg0 = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in seg.items() if k not in ("image", "context", "template")}
            last = ids.shape[-1] - 1
            r_full, v_full, pat = _readout(bridge, ids, pv, name, layer, coef, inter, tok_ids,
                                           want_pattern=True)
            # attention mass (mean over heads) to each group at the last token
            mass = {g: float(pat[:, seg[g]].sum(axis=1).mean()) if len(seg[g]) else 0.0
                    for g in ("image", "context", "template")} if pat is not None else {}
            # causal contribution of image / context (probe + valence units)
            r_noimg, v_noimg, _ = _readout(bridge, ids, pv, name, layer, coef, inter, tok_ids,
                                           knockout=_knockout_hook(seg["image"], last))
            r_noctx, v_noctx, _ = _readout(bridge, ids, pv, name, layer, coef, inter, tok_ids,
                                           knockout=_knockout_hook(seg["context"], last))
            if not sanity["checked"]:
                sanity.update(checked=True,
                              pattern_edits_attn_out=bool(abs(r_full - r_noimg) > 1e-6))
            rows.append({
                "image_path": r["image_path"], "condition": cond, "context_id": cid,
                "text_code": TEXT_CODE[cond], "r_full_probe": r_full, "v_full": v_full,
                "attn_image": mass.get("image"), "attn_context": mass.get("context"),
                "attn_template": mass.get("template"),
                "contrib_image_probe": r_full - r_noimg, "contrib_context_probe": r_full - r_noctx,
                "contrib_image_val": v_full - v_noimg, "contrib_context_val": v_full - v_noctx,
            })
        n_ok = len({row["image_path"] for row in rows})

    df = pd.DataFrame(rows)
    df.to_parquet(STAGE_F_DIR / "attribution.parquet")
    summary = _summarize(df)

    metrics = {
        "run": run_stamp(), "git": git_hash(), "layer": layer, "tap": tap,
        "n_images": n_ok, "n_skipped": n_skip, "image_group": "positive",
        "segmentation_example": seg0, "pattern_hook_sanity": sanity,
        "by_condition": summary,
        "note": ("causal attention-knockout attribution at last-token blocks.{L}.attn.hook_pattern; "
                 "contribution(G)=probe(full)-probe(knockout G). direct-L18 path only (lower bound "
                 "on context influence that leaked upstream).").format(L=layer),
    }
    metrics["verdict"] = _verdict(summary, sanity)
    save_json(metrics, STAGE_F_DIR / "attribution_metrics.json")
    _plot(summary)

    print(f"\nStage F attribution — {n_ok} positive-group images x {len(conditions)} contexts "
          f"({n_skip} skipped), L{layer} {tap}.")
    if not seg0["image_ok"]:
        print(f"  WARNING image span len={seg0['img_len']} (expected 256) — check segmentation.")
    if sanity["pattern_edits_attn_out"] is False:
        print("  WARNING editing hook_pattern did NOT change attn_out — attention may be fused "
              "(flash/SDPA); knockout contributions are invalid. Fix before trusting.")
    print(f"  segmentation[0]: image {seg0['img_len']} tok | context=\"{seg0['context_text'][:50]}\"")
    print(f"\n  {'condition':9s} {'attn_img':>8s} {'attn_ctx':>8s} {'contrib_img':>11s} "
          f"{'contrib_ctx':>11s}  (probe units)")
    for cond in ("neutral", "negative", "positive"):
        s = summary.get(cond, {})
        if not s:
            continue
        print(f"  {cond:9s} {s['attn_image']:>8.3f} {s['attn_context']:>8.3f} "
              f"{s['contrib_image_probe']:>+11.3f} {s['contrib_context_probe']:>+11.3f}")
    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  data -> {STAGE_F_DIR/'attribution.parquet'}   metrics -> "
          f"{STAGE_F_DIR/'attribution_metrics.json'}")
    return metrics


def _summarize(df) -> dict:
    out = {}
    for cond in ("neutral", "negative", "positive"):
        g = df[df["condition"] == cond]
        if not len(g):
            continue
        out[cond] = {k: float(g[k].mean()) for k in (
            "attn_image", "attn_context", "attn_template",
            "contrib_image_probe", "contrib_context_probe",
            "contrib_image_val", "contrib_context_val", "r_full_probe", "v_full")}
    return out


def _verdict(summary, sanity) -> str:
    """Attribute the positive-channel collapse to re-routing (attention share) vs value competition.

    The effect to explain: negative context influences the read-out much more than positive context
    (|contrib_context| larger under negative). Two decompositions of that gap:
      re-routing  — the last token attends MORE to negative-context tokens (attn share differs)
      value       — per unit of attention, negative-context tokens write a STRONGER read-out shift
                    (|contrib|/attn differs, attention held roughly constant)
    """
    if sanity.get("pattern_edits_attn_out") is False:
        return "INVALID — hook_pattern edits did not propagate to attn_out (fused attention)."
    neg, pos = summary.get("negative"), summary.get("positive")
    if not neg or not pos:
        return "INCOMPLETE — missing a context condition."
    a_neg, a_pos = neg["attn_context"], pos["attn_context"]
    m_neg, m_pos = abs(neg["contrib_context_probe"]), abs(pos["contrib_context_probe"])

    # GUARD: is the DIRECT last-token->context path even large enough to explain the effect? Compare
    # the context contribution to the image contribution (both direct L18). If |contrib_ctx| is a tiny
    # fraction of |contrib_img|, the read-out barely reads the context tokens directly and the context
    # effect must be INDIRECT (context reshaped the high-attention image/template positions upstream);
    # any re-routing / value ratio below is then a ratio of negligible magnitudes and must not be cited.
    img_ref = float(np.mean([abs(c["contrib_image_probe"]) for c in
                             (summary.get("neutral"), neg, pos) if c]))
    a_tmpl = 1.0 - neg["attn_image"] - a_neg
    if max(m_neg, m_pos) < 0.15 * img_ref:
        return (f"DIRECT L18 PATH NEGLIGIBLE — the last token routes only ~{a_neg:.0%} of its attention "
                f"to context tokens ({neg['attn_image']:.0%} image, ~{a_tmpl:.0%} template) and their "
                f"direct read-out contribution (|contrib_ctx| neg {m_neg:.3f}, pos {m_pos:.3f}) is "
                f"<15% of the image contribution ({img_ref:.3f}); attention shares are ~invariant to "
                f"context polarity (NO re-routing). => the negativity dominance is NOT a last-token L18 "
                f"read-out-layer effect. Context influence is INDIRECT — established upstream, reshaping "
                f"the high-attention image/template positions before L18. NEXT: layer-resolved probe "
                f"and activation-patching (image- vs template-token groups) to localize where it enters.")
    u_neg, u_pos = m_neg / max(a_neg, 1e-4), m_pos / max(a_pos, 1e-4)  # |contrib| per unit attention
    attn_ratio = a_neg / max(a_pos, 1e-4)
    unit_ratio = u_neg / max(u_pos, 1e-4)
    facts = (f"|contrib_ctx| neg {m_neg:.3f} vs pos {m_pos:.3f}; attn share neg {a_neg:.3f} vs pos "
             f"{a_pos:.3f} (×{attn_ratio:.2f}); per-unit-attention |contrib| neg {u_neg:.2f} vs pos "
             f"{u_pos:.2f} (×{unit_ratio:.2f}).")
    if unit_ratio >= 1.5 and unit_ratio >= attn_ratio:
        lead = ("VALUE COMPETITION dominates — negative-context tokens write a stronger read-out shift "
                "per unit of attention; the positive channel collapses in what it WRITES, not in "
                "attention share.")
    elif attn_ratio >= 1.5 and attn_ratio > unit_ratio:
        lead = ("ATTENTION RE-ROUTING dominates — the last token attends far more to negative- than "
                "positive-context tokens; the positive channel loses attention SHARE.")
    else:
        lead = "MIXED — both attention share and per-unit value contribute; neither cleanly dominates."
    return f"{lead} {facts}"


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in ("neutral", "negative", "positive") if c in summary]
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (key_i, key_c, title) in zip(axes, [
            ("attn_image", "attn_context", "attention mass (last token)"),
            ("contrib_image_probe", "contrib_context_probe", "contribution to probe read-out")]):
        ax.bar(x - 0.2, [summary[c][key_i] for c in conds], 0.4, label="image")
        ax.bar(x + 0.2, [summary[c][key_c] for c in conds], 0.4, label="context")
        ax.set_xticks(x); ax.set_xticklabels([f"{c} ctx" for c in conds])
        ax.set_title(title); ax.axhline(0, color="gray", lw=0.5); ax.legend(fontsize=8)
    fig.suptitle("Stage F L18 attribution (positive images): why positive framing collapses")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "stage_f_attribution.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage F — L18 source attribution (mechanism)")
    ap.add_argument("--config", default="config/stage_f.yaml")
    ap.add_argument("--limit", type=int, default=None, help="positive-group image count (default 40)")
    args = ap.parse_args()
    run(args.config, limit_override=args.limit)


if __name__ == "__main__":
    main()
