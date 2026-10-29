"""Camera-ready re-analysis: every readout x every contrast, on one image set, with crossed intervals.

Both VLM4RWD reviewers and the meta-review asked for the same thing: the matched-set result read on
every scale the paper defines, side by side, so a reader can see which conclusions depend on the
bounded score. This module produces that grid from the stored parquets (`lp_{label}` log-probs, every
row), on CPU, with no forward passes.

Readouts (per row):
  valence    P(pos) - P(neg), bounded in [-1, 1]                        (the published primary)
  sum_lo     logsumexp(lp_pos) - logsumexp(lp_neg) = log P(pos)/P(neg)  (unbounded, summed category)
  best_lo    max lp_pos - max lp_neg                                    (unbounded, best label)
  headroom   valence shift / room left toward the bound it moves to     (mirror arms only)
  category   argmax label's valence category, neutral-corrected        (mirror only; the paper's
                                                                         corrected override gap)

Contrasts (each effect is measured per photograph against that photograph's mean neutral response):
  mirror      |drop| - |rise|: negative sentence on positive images vs positive sentence on negative
              images, both CONFLICTING. Cell means first, then absolute values (published estimator).
  within_pos  mean(|d_neg| - |d_pos|) on positive images. The negative member conflicts, the positive
              member agrees, so this contrast confounds valence with conflict-vs-agreement.
  within_neg  the same on negative images, where the roles reverse. If within_pos reflects valence,
              within_neg should also be positive; if it reflects conflict, within_neg is negative.
  equal_room  mean(|d_neg| - |d_pos|) on photographs whose neutral response is near zero, where both
              members have the same room to move. The one contrast free of both confounds.

Intervals are percentile bootstraps. `photo` resamples photographs only (sentences fixed). `crossed`
also resamples sentences: matched pairs (or bank sentences) independently per image group, which is
the published scheme for the mirror contrast. `+neutral` additionally resamples the neutral
sentences that define every baseline, the reviewer's request; with the published two-sentence bank
that draw is coarse (three distinct baselines), which is why bank B exists.

    python -m src.experiments.analyze_matched_readouts                       # published matched set
    python -m src.experiments.analyze_matched_readouts --equal-room <pq>     # equal-room run
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..data.labels import EMOTION_LABELS
from ..paths import STAGE_F_DIR, ensure_dirs
from .common import git_hash, run_stamp, save_json
from .shared.reporting import NEGATIVE_LABELS, POSITIVE_LABELS

LP = [f"lp_{w}" for w in EMOTION_LABELS]
_POS_IDX = [EMOTION_LABELS.index(w) for w in POSITIVE_LABELS]
_NEG_IDX = [EMOTION_LABELS.index(w) for w in NEGATIVE_LABELS]
GRADED = ("valence", "sum_lo", "best_lo")
PUBLISHED_NEUTRAL = ("z0", "z1")
ROOM_FLOOR = 1e-3
EQUAL_ROOM_WINDOW = 0.3   # |measured valence| at which the two members' room differs by < 2x


def add_readouts(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the unbounded margins and the argmax-category indicators to every row."""
    d = df.copy()
    lp = d[LP].to_numpy(dtype=float)
    d["sum_lo"] = logsumexp(lp[:, _POS_IDX], axis=1) - logsumexp(lp[:, _NEG_IDX], axis=1)
    d["best_lo"] = lp[:, _POS_IDX].max(axis=1) - lp[:, _NEG_IDX].max(axis=1)
    top = lp.argmax(axis=1)
    d["neg_top"] = np.isin(top, _NEG_IDX).astype(float)
    d["pos_top"] = np.isin(top, _POS_IDX).astype(float)
    return d


# --------------------------------------------------------------------------- per-group arrays
def group_arrays(d: pd.DataFrame, group: str, value: str, neutral_ids) -> dict | None:
    """Photograph-by-sentence arrays for one image group.

    Rows are photographs, not person-annotations: EMOTIC can annotate several people in one photo, the
    model sees the whole photo, and those rows are byte-identical forward passes, so they are averaged
    rather than counted twice (the convention `analyze_stage_f_unbounded` already uses).
    Returns `neg` and `pos` (photos x that arm's own context ids), `neutral` (photos x neutral ids)
    and the photo index. In the matched bank both arms carry the same pair ids in the same order
    (`paired`); in the varied bank they carry different sentences, and each arm keeps only its own.
    """
    sub = d[d["image_group"] == group]
    neutral = sub[(sub["condition"] == "neutral") & sub["context_id"].isin(neutral_ids)]
    if neutral.empty:
        return None
    z = neutral.pivot_table(index="image_path", columns="context_id", values=value, aggfunc="mean")
    z = z[[c for c in neutral_ids if c in z.columns]].dropna()
    neg = sub[sub["condition"] == "negative"].pivot_table(
        index="image_path", columns="context_id", values=value, aggfunc="mean")
    pos = sub[sub["condition"] == "positive"].pivot_table(
        index="image_path", columns="context_id", values=value, aggfunc="mean")
    photos = z.index.intersection(neg.index).intersection(pos.index)
    if not len(photos):
        return None
    neg_ids, pos_ids = sorted(neg.columns), sorted(pos.columns)
    return {"photos": list(photos),
            "neg": neg.loc[photos, neg_ids].to_numpy(dtype=float),
            "pos": pos.loc[photos, pos_ids].to_numpy(dtype=float),
            "neutral": z.loc[photos].to_numpy(dtype=float),
            "neg_ids": neg_ids, "pos_ids": pos_ids, "paired": neg_ids == pos_ids,
            "neutral_ids": list(z.columns)}


def _effects(g: dict, rows, cols, ncols, headroom: bool = False):
    """Per-cell shifts of the negative and positive members against each photo's neutral mean."""
    base = g["neutral"][np.ix_(rows, ncols)].mean(axis=1, keepdims=True)
    neg = g["neg"][np.ix_(rows, cols[0])] - base
    pos = g["pos"][np.ix_(rows, cols[1])] - base
    if headroom:
        neg = neg / np.clip(base + 1.0, ROOM_FLOOR, None)
        pos = pos / np.clip(1.0 - base, ROOM_FLOOR, None)
    return neg, pos


def _stat(name: str, P, N, headroom: bool = False) -> float:
    """Evaluate contrast `name` on resampled positive-group (P) and negative-group (N) selections."""
    if name == "mirror":
        p_neg, _ = _effects(*P, headroom=headroom)
        _, n_pos = _effects(*N, headroom=headroom)
        return abs(np.nanmean(p_neg)) - abs(np.nanmean(n_pos))
    g = P if name in ("within_pos", "equal_room", "at_zero_baseline") else N
    neg, pos = _effects(*g)
    if name == "at_zero_baseline":
        # Screening narrows the baseline but cannot fix it, so regress each photograph's contrast on
        # its own measured baseline and read the contrast where both members have equal room (b=0).
        grp, rows, _cols, ncols = g
        b = grp["neutral"][np.ix_(rows, ncols)].mean(axis=1)
        d = np.nanmean(np.abs(neg) - np.abs(pos), axis=1)
        if np.ptp(b) == 0:
            return float(np.nanmean(d))
        return float(np.polyfit(b, d, 1)[1])
    return float(np.nanmean(np.abs(neg) - np.abs(pos)))


def _draw(rng, g, cols, ncols):
    """One replicate's selection for group `g`: photographs always resampled, the rest as given."""
    n_rows = g["neg"].shape[0]
    return (g, rng.integers(0, n_rows, n_rows), cols, ncols)


def _replicate(rng, P: dict | None, N: dict | None, resample_cols: bool, resample_neutral: bool,
               shared_pairs: bool = True):
    """Draw one bootstrap replicate for both image groups.

    Photographs are resampled within each group. The neutral bank is one population shared by both
    groups, so a replicate draws ONE neutral sample for both. When both groups were shown the same
    sentences (the matched set: the same six pairs, `mp0`..`mp5`, appear in both), the pairs are
    likewise one shared population and are drawn once per replicate (`shared_pairs`). The varied set
    shows different sentences to the two groups (negative sentences to positive photos, positive to
    negative), so its sentence draws are independent.
    """
    ref = P if P is not None else N
    n_neu = ref["neutral"].shape[1]
    ncols = rng.integers(0, n_neu, n_neu) if resample_neutral else np.arange(n_neu)
    same = (P is not None and N is not None and P["paired"] and N["paired"]
            and P["neg_ids"] == N["neg_ids"])

    def draw(n):
        return rng.integers(0, n, n) if resample_cols else np.arange(n)

    def cols_for(g):
        # A matched pair is one unit: its two members are resampled together.
        if g["paired"]:
            c = draw(g["neg"].shape[1])
            return (c, c)
        return (draw(g["neg"].shape[1]), draw(g["pos"].shape[1]))

    if shared_pairs and same:
        cols = cols_for(P)
        return _draw(rng, P, cols, ncols), _draw(rng, N, cols, ncols)
    Pd = _draw(rng, P, cols_for(P), ncols) if P is not None else None
    Nd = _draw(rng, N, cols_for(N), ncols) if N is not None else None
    return Pd, Nd


_INTERVALS = (("ci95_photo", False, False), ("ci95_crossed", True, False),
              ("ci95_crossed_neutral", True, True))


def _full(g):
    return None if g is None else (g, np.arange(g["neg"].shape[0]),
                                   (np.arange(g["neg"].shape[1]), np.arange(g["pos"].shape[1])),
                                   np.arange(g["neutral"].shape[1]))


def _intervals(statistic, P, N, n_boot: int, seed: int, shared_pairs: bool) -> dict:
    out = {}
    for label, cols, neutral in _INTERVALS:
        rng = np.random.default_rng(seed)
        boot = np.array([statistic(*_replicate(rng, P, N, cols, neutral, shared_pairs))
                         for _ in range(n_boot)])
        out[label] = [float(x) for x in np.nanpercentile(boot, [2.5, 97.5])]
    return out


def contrast(name: str, P: dict | None, N: dict | None, n_boot: int = 2000, seed: int = 0,
             headroom: bool = False, shared_pairs: bool = True) -> dict:
    """Point estimate plus photo, crossed and crossed+neutral intervals for one contrast."""
    needs_n = name in ("mirror", "within_neg")
    needs_p = name in ("mirror", "within_pos", "equal_room", "at_zero_baseline")
    if (needs_p and P is None) or (needs_n and N is None):
        return {}
    # Within-photograph contrasts compare the two members of a pair; the varied bank has no pairs.
    if name != "mirror" and not (P if needs_p else N)["paired"]:
        return {}
    stat = lambda Pd, Nd: _stat(name, Pd, Nd, headroom)
    out = {"estimate": float(stat(_full(P), _full(N))),
           **_intervals(stat, P, N, n_boot, seed, shared_pairs)}
    lo, hi = out["ci95_crossed_neutral"]
    out["crossed_neutral_excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


def arms(P: dict | None, N: dict | None, headroom: bool = False) -> dict:
    """The four shift means behind every contrast, so a reader can see what each subtracts."""
    out = {}
    for tag, g in (("positive_images", P), ("negative_images", N)):
        if g is None:
            continue
        neg, pos = _effects(*_full(g), headroom=headroom)
        out[tag] = {"neg_sentence": float(np.nanmean(neg)), "pos_sentence": float(np.nanmean(pos)),
                    "n_photos": int(g["neg"].shape[0])}
    return out


def category_gap(d: pd.DataFrame, neutral_ids, n_boot: int = 2000, seed: int = 0,
                 shared_pairs: bool = True) -> dict:
    """Neutral-corrected override gap with all three interval types.

    Corrected negative-label rate on positive images under negative sentences minus corrected
    positive-label rate on negative images under positive sentences. Matches
    `shared.reporting.corrected_override_gap` on the point estimate, and is the code path for the
    crossed intervals that the paper's corrected-gap cells previously lacked.
    """
    P = group_arrays(d, "positive", "neg_top", neutral_ids)
    N = group_arrays(d, "negative", "pos_top", neutral_ids)
    if P is None or N is None:
        return {}

    def gap(Pd, Nd):
        p_neg, _ = _effects(*Pd)
        _, n_pos = _effects(*Nd)
        return float(np.nanmean(p_neg) - np.nanmean(n_pos))

    return {"estimate": gap(_full(P), _full(N)),
            **_intervals(gap, P, N, n_boot, seed, shared_pairs)}


# --------------------------------------------------------------------------- drivers
def matched_grid(df: pd.DataFrame, neutral_ids=PUBLISHED_NEUTRAL, n_boot: int = 2000,
                 seed: int = 0, shared_pairs: bool = True) -> dict:
    """The reviewers' table: every graded readout x mirror / within_pos / within_neg, plus the
    head-room-corrected mirror and the corrected categorical gap, all against `neutral_ids`."""
    d = add_readouts(df[df["condition"] != "none"])
    grid = {}
    for value in GRADED:
        P = group_arrays(d, "positive", value, neutral_ids)
        N = group_arrays(d, "negative", value, neutral_ids)
        grid[value] = {"arms": arms(P, N),
                       **{name: contrast(name, P, N, n_boot, seed, shared_pairs=shared_pairs)
                          for name in ("mirror", "within_pos", "within_neg")}}
    P = group_arrays(d, "positive", "valence", neutral_ids)
    N = group_arrays(d, "negative", "valence", neutral_ids)
    grid["headroom"] = {"arms": arms(P, N, headroom=True),
                        "mirror": contrast("mirror", P, N, n_boot, seed, headroom=True,
                                           shared_pairs=shared_pairs)}
    grid["category"] = {"mirror": category_gap(d, neutral_ids, n_boot, seed, shared_pairs)}
    return grid


def baseline_profile(df: pd.DataFrame, neutral_ids=PUBLISHED_NEUTRAL) -> dict:
    """Distribution of each group's neutral-context valence: where the bounds sit for each arm."""
    d = df[(df["condition"] == "neutral") & df["context_id"].isin(neutral_ids)]
    out = {}
    for group, sub in d.groupby("image_group"):
        z = sub.groupby("image_path")["valence"].mean()
        out[str(group)] = {"n_photos": int(len(z)),
                           "quantiles": {str(q): float(z.quantile(q))
                                         for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                           "n_within_0.3": int((z.abs() <= 0.3).sum())}
    return out


def neutral_sentence_profile(df: pd.DataFrame, neutral_ids) -> dict:
    """Each neutral sentence's mean shift from the no-context response, per image group.

    A neutral sentence should leave the image-conditioned judgment where no sentence leaves it; this
    is the image-conditioned definition of neutrality the reviewer asked for. (Text-only scores do
    not work for Qwen: with no image it reads even a neutral sentence as sadness.)
    """
    out = {}
    for group, sub in df.groupby("image_group"):
        none = sub[sub["condition"] == "none"].groupby("image_path")["valence"].mean()
        rows = {}
        for cid in neutral_ids:
            z = sub[sub["context_id"] == cid].groupby("image_path")["valence"].mean()
            shared = z.index.intersection(none.index)
            if len(shared):
                rows[cid] = float((z[shared] - none[shared]).mean())
        out[str(group)] = rows
    return out


def equal_room(df: pd.DataFrame, neutral_ids, threshold: float = 0.3, n_boot: int = 2000,
               seed: int = 0) -> dict:
    """|d_neg| - |d_pos| on photographs whose neutral response is within `threshold` of zero.

    Every photo in `df` is treated as one group regardless of its EMOTIC label: the selection is on
    the model's reading, which is the quantity that sets the room. `threshold` is applied to the
    mean over `neutral_ids`; the screened RUN passes `threshold=inf` because its photographs were
    selected on other sentences (see `screen_photos`).

    Screening on one baseline does not guarantee room on another, so three things are reported per
    readout: the contrast on every kept photograph; the same on the subset whose MEASUREMENT baseline
    is also within `room_window` of zero; and `at_zero_baseline`, the intercept of a per-photograph
    regression of the contrast on its measured baseline, i.e. the contrast where both members have
    equal room. The measured-baseline distribution is reported so a reader can see the room directly.
    """
    d = add_readouts(df[df["condition"] != "none"]).assign(image_group="all")
    base = d[(d["condition"] == "neutral") & d["context_id"].isin(neutral_ids)]
    v0 = base.groupby("image_path")["valence"].mean()
    keep = v0[v0.abs() <= threshold].index
    d = d[d["image_path"].isin(keep)]
    v0 = v0.loc[keep]
    room_window = threshold if np.isfinite(threshold) else EQUAL_ROOM_WINDOW
    inside = v0[v0.abs() <= room_window].index
    out = {"threshold": threshold, "n_photos": int(len(keep)), "neutral_ids": list(neutral_ids),
           "measured_baseline": {"quantiles": {str(q): float(v0.quantile(q))
                                               for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
                                 "room_window": room_window,
                                 "n_within_room_window": int(len(inside))}}
    for value in GRADED:
        P = group_arrays(d, "all", value, neutral_ids)
        Pin = group_arrays(d[d["image_path"].isin(inside)], "all", value, neutral_ids)
        out[value] = {"arms": arms(P, None).get("positive_images", {}),
                      "equal_room": contrast("equal_room", P, None, n_boot, seed),
                      "within_room_window": (contrast("equal_room", Pin, None, n_boot, seed)
                                             if Pin is not None and len(inside) >= 5 else {}),
                      "at_zero_baseline": contrast("at_zero_baseline", P, None, n_boot, seed)}
    return out


def screen_photos(screen: pd.DataFrame, threshold: float, max_photos: int | None,
                  seed: int = 0) -> list[str]:
    """Photographs whose mean response over no-context + the published neutral pair is near zero.

    Deterministic: candidates are shuffled with `seed` before truncating to `max_photos`, so the kept
    set does not favour any EMOTIC sub-corpus or file order.
    """
    v = screen.groupby("image_path")["valence"].mean()
    kept = sorted(v[v.abs() <= threshold].index)
    rng = np.random.default_rng(seed)
    kept = [kept[i] for i in rng.permutation(len(kept))]
    return kept[:max_photos] if max_photos else kept


def photo_key(path: str) -> str:
    """`<corpus>/images/<file>`: the part of an EMOTIC path that survives remounting on Colab."""
    return "/".join(str(path).split("/")[-3:])


# Bounded valence saturates, so agreement on it alone can hide a large log-odds discrepancy
# (margins of 20 and 10 give nearly identical valence). Both scales are checked.
REPRODUCTION_TOLERANCE = {"valence": 0.01, "sum_lo": 0.1, "best_lo": 0.1}


def merge_neutral_bank(matched: pd.DataFrame, bank: pd.DataFrame,
                       bank_ids: tuple[str, ...] | None = None) -> tuple[pd.DataFrame, dict]:
    """Append bank-B rows to the published matched parquet, after checking the session reproduces it.

    Refuses unless (1) every published (photo, group) has both published neutral rows and every
    bank-B sentence in the bank run, (2) every bank row has finite log-probs, and (3) the re-scored
    published neutral rows agree with the published ones on bounded valence AND both log-odds
    margins. A partial or drifted session would otherwise change the sample or the zero point of
    every effect without any visible sign.
    """
    from ..data.conflict_contexts import NEUTRAL_BANK_B

    bank_ids = bank_ids or tuple(f"zb{i}" for i in range(len(NEUTRAL_BANK_B)))
    if not np.isfinite(bank[LP].to_numpy(dtype=float)).all():
        raise ValueError("bank run has non-finite log-probs")
    m = add_readouts(matched).assign(_k=matched["image_path"].map(photo_key))
    b = add_readouts(bank).assign(_k=bank["image_path"].map(photo_key))

    need = set(map(tuple, m[["_k", "image_group"]].drop_duplicates().to_numpy()))
    have = {}
    for (k, g), sub in b.groupby(["_k", "image_group"]):
        have[(k, g)] = set(sub["context_id"])
    required = set(PUBLISHED_NEUTRAL) | set(bank_ids)
    missing = sorted(f"{k}/{g}" for k, g in need if not required <= have.get((k, g), set()))
    extra = sorted(f"{k}/{g}" for k, g in have if (k, g) not in need)
    if missing or extra:
        raise ValueError(f"bank coverage mismatch: {len(missing)} published photo-groups incomplete "
                         f"(e.g. {missing[:3]}), {len(extra)} unexpected (e.g. {extra[:3]})")

    keys = ["_k", "image_group", "context_id"]
    pub = m[m["context_id"].isin(PUBLISHED_NEUTRAL)].groupby(keys)[list(REPRODUCTION_TOLERANCE)].mean()
    new = b[b["context_id"].isin(PUBLISHED_NEUTRAL)].groupby(keys)[list(REPRODUCTION_TOLERANCE)].mean()
    pub, new = pub.align(new, join="inner")
    check = {"n_checked": int(len(pub))}
    for col, tol in REPRODUCTION_TOLERANCE.items():
        check[f"max_abs_{col}_diff"] = float((pub[col] - new[col]).abs().max())
    bad = {c: check[f"max_abs_{c}_diff"] for c, t in REPRODUCTION_TOLERANCE.items()
           if not check[f"max_abs_{c}_diff"] <= t}
    if bad:
        raise ValueError(f"bank session does not reproduce the published neutral rows: {bad}")

    by_key = m.drop_duplicates("_k").set_index("_k")["image_path"]
    rows = b[b["context_id"].isin(bank_ids)].assign(image_path=lambda x: x["_k"].map(by_key))
    rows = rows[[c for c in matched.columns if c in rows.columns]]
    return pd.concat([matched, rows], ignore_index=True), check


def _fmt_row(name: str, c: dict) -> str:
    if not c:
        return f"    {name:11s} n/a"
    xn = c["ci95_crossed_neutral"]
    flag = "*" if xn[0] > 0 or xn[1] < 0 else " "
    p, x, xn = c["ci95_photo"], c["ci95_crossed"], c["ci95_crossed_neutral"]
    return (f"    {name:11s} {c['estimate']:+9.3f}  photo [{p[0]:+.3f},{p[1]:+.3f}]  "
            f"crossed [{x[0]:+.3f},{x[1]:+.3f}]  +neutral [{xn[0]:+.3f},{xn[1]:+.3f}] {flag}")


TABLE_PATH = Path("paper/tables/readouts.tex")
# Written whole: LaTeX rejects an \\input between \\midrule and \\bottomrule ("Misplaced \\noalign").
TABLE_HEAD = r"""% GENERATED by `python -m src.experiments.analyze_matched_readouts`; edit the generator.
\begin{table}[t]
  \caption{The matched set read every way the paper defines (\qwen{}, six one-word pairs). Each shift is measured against the photograph's own response averaged over twelve neutral sentences; intervals resample photographs, word pairs and neutral sentences (bold: excludes zero). \emph{Mirror}: negative sentence on positive photographs against positive sentence on negative photographs, both conflicting. \emph{Within}: $|\Delta_{\rm neg}|-|\Delta_{\rm pos}|$ on one photograph, where one member conflicts and the other agrees, so its sign follows conflict: it reverses between the two image groups on every readout. \emph{Equal room}: the same within-photograph contrast on @ER_N@ mid-valence photographs, where neither member conflicts with the image, estimated at a measured neutral baseline of zero (regression intercept), where both members have equal room to move; most measured baselines lie well above zero, so this is an extrapolation (\S\ref{sec:minimal}). Head-room and categorical readouts are defined for the mirror contrast only. Log-odds rows are in log-odds units, not comparable in magnitude to the bounded rows.}
  \label{tab:readouts}
  \centering
  \scriptsize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{lrrrr}
    \toprule
    Readout & Mirror & Within, pos.\ photos & Within, neg.\ photos & Equal room \\
    \midrule
"""
TABLE_TAIL = r"""    \bottomrule
  \end{tabular}
\end{table}
"""
_ROWS = (("valence", "Bounded valence", "{:+.3f}"), ("headroom", "Head-room-corrected", "{:+.3f}"),
         ("sum_lo", "Summed log-odds", "{:+.2f}"), ("best_lo", "Best-label log-odds", "{:+.2f}"),
         ("category", "Categorical (corrected)", "{:+.1%}"))


def _cells(c: dict | None, fmt: str, interval: str) -> tuple[str, str]:
    """(estimate, interval) LaTeX cells; an interval excluding zero is set in bold."""
    if not c:
        return "---", ""
    lo, hi = c[interval]
    f = lambda x: fmt.format(x).replace("%", "\\%")
    ci = f"[{f(lo)},{f(hi)}]"
    return f"${f(c['estimate'])}$", (f"$\\mathbf{{{ci}}}$" if lo > 0 or hi < 0 else f"${ci}$")


def latex_table(out: dict) -> str:
    """The main-text readout table body, generated so it cannot drift from matched_readouts.json.

    Two rows per readout: estimates, then 95% intervals. Uses the published pair plus bank B with
    neutral sentences resampled once the bank run has been merged, otherwise the published pair with
    photos and pairs resampled. The equal-room column uses the screened run when present, otherwise
    the post-hoc subset of the matched set.
    """
    merged = "published_plus_b" in out["matched"]
    grid = out["matched"]["published_plus_b" if merged else "published"]
    interval = "ci95_crossed_neutral" if merged else "ci95_crossed"
    er = out.get("equal_room_run") or out["equal_room_post_hoc"]
    lines = []
    for key, label, fmt in _ROWS:
        cells = [_cells(grid[key].get(n), fmt, interval)
                 for n in ("mirror", "within_pos", "within_neg")]
        cells.append(_cells(er[key]["at_zero_baseline"], fmt, interval) if key in er else ("---", ""))
        lines.append(f"    {label} & " + " & ".join(c[0] for c in cells) + r" \\")
        lines.append(r"    \quad 95\% CI & " + " & ".join(c[1] for c in cells) + r" \\")
    return "\n".join(lines) + "\n"


def _rel(path: Path) -> str:
    """Repo-relative where possible: the metrics file is released, the analysing machine is not."""
    try:
        return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return Path(path).name


def run(matched_pq: Path, equal_room_pq: Path | None, n_boot: int, seed: int,
        bank_pq: Path | None = None) -> dict:
    ensure_dirs()
    out = {"run": run_stamp(), "git": git_hash(), "n_boot": n_boot, "seed": seed,
           "matched_parquet": _rel(matched_pq), "matched": {}}
    df = pd.read_parquet(matched_pq)
    if bank_pq is not None and bank_pq.exists():
        df, out["neutral_bank_reproduction"] = merge_neutral_bank(df, pd.read_parquet(bank_pq))
        print(f"  merged neutral bank B: {out['neutral_bank_reproduction']}")
    out["baseline_profile"] = baseline_profile(df)
    banks = {"published": PUBLISHED_NEUTRAL, "z0_only": ("z0",), "z1_only": ("z1",)}
    bank_b = tuple(sorted(c for c in df["context_id"].unique()
                          if isinstance(c, str) and c.startswith("zb")))
    if bank_b:
        banks["bank_b"] = bank_b
        banks["published_plus_b"] = PUBLISHED_NEUTRAL + bank_b
        out["neutral_sentence_profile"] = neutral_sentence_profile(df, PUBLISHED_NEUTRAL + bank_b)
    for tag, ids in banks.items():
        grid = matched_grid(df, ids, n_boot, seed)
        out["matched"][tag] = grid
        print(f"\n== matched set, neutral = {tag} {ids}")
        for value in (*GRADED, "headroom", "category"):
            print(f"  {value}")
            for name in ("mirror", "within_pos", "within_neg"):
                if name in grid[value]:
                    print(_fmt_row(name, grid[value][name]))
    out["equal_room_post_hoc"] = equal_room(df, PUBLISHED_NEUTRAL, n_boot=n_boot, seed=seed)
    print(f"\n== equal room, post hoc on the matched set "
          f"(n={out['equal_room_post_hoc']['n_photos']} photos, |v|<=0.3)")
    for value in GRADED:
        print(f"  {value}")
        print(_fmt_row("equal_room", out["equal_room_post_hoc"][value]["equal_room"]))
        print(_fmt_row("at_zero_b", out["equal_room_post_hoc"][value]["at_zero_baseline"]))

    if equal_room_pq is not None and equal_room_pq.exists():
        er = pd.read_parquet(equal_room_pq)
        ids = tuple(sorted(c for c in er["context_id"].unique()
                           if isinstance(c, str) and c.startswith("zb")))
        # The run already screened; threshold=inf keeps every measured photo.
        out["equal_room_run"] = equal_room(er, ids, threshold=np.inf, n_boot=n_boot, seed=seed)
        out["equal_room_run"]["parquet"] = _rel(equal_room_pq)
        out["equal_room_run"]["neutral_sentence_profile"] = neutral_sentence_profile(er, ids)
        print(f"\n== equal-room RUN (n={out['equal_room_run']['n_photos']} screened photos, "
              f"baseline = bank B)")
        print(f"  measured baseline (bank B): {out['equal_room_run']['measured_baseline']}")
        for value in GRADED:
            print(f"  {value}")
            for k in ("equal_room", "within_room_window", "at_zero_baseline"):
                print(_fmt_row(k[:11], out["equal_room_run"][value][k]))

    path = STAGE_F_DIR / "matched_readouts.json"
    save_json(out, path)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    er = out.get("equal_room_run") or out["equal_room_post_hoc"]
    head = TABLE_HEAD.replace("@ER_N@", str(er["n_photos"]))
    TABLE_PATH.write_text(head + latex_table(out) + TABLE_TAIL)
    print(f"  table -> {TABLE_PATH}")
    print(f"\n  data -> {path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--matched", type=Path,
                    default=STAGE_F_DIR / "conflict_qwen3-vl-8b-instruct_minimal.parquet")
    ap.add_argument("--equal-room", type=Path, default=STAGE_F_DIR / "equal_room_qwen.parquet")
    ap.add_argument("--neutral-bank", type=Path,
                    default=STAGE_F_DIR / "controls_neutral_bank_qwen.parquet")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.matched, a.equal_room, a.n_boot, a.seed, a.neutral_bank)


if __name__ == "__main__":
    main()
