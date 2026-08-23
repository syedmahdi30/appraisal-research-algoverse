"""Run-key tests for the Stage F token-budget runner.

`slug` is what keeps two different experiments from writing to one path. The module's own docstring
records that a shared path destroyed three published numbers before per-run keys existed, so the
keys are pinned here: any future change to the key format has to break a test rather than silently
alias a new run onto an old one's file.
"""

from src.experiments import stage_f_token_budget as tb

QWEN = "Qwen/Qwen3-VL-8B-Instruct"
GEMMA = "google/gemma-3-4b-it"


def test_slug_pins_the_published_run_keys():
    # These exact strings name files already on disk; changing them orphans published results.
    assert tb.slug(QWEN, None) == "qwen3-vl-8b-instruct"
    assert tb.slug(QWEN, 448) == "qwen3-vl-8b-instruct_px448"
    assert tb.slug(QWEN, None, bank="minimal") == "qwen3-vl-8b-instruct_minimal"
    assert tb.slug(GEMMA, None, "legacy") == "gemma-3-4b-it_legacy"


def test_slug_orders_budget_before_style_and_bank():
    # `_base_runs_for` globs on this ordering; if it changes, the glob silently matches nothing.
    assert tb.slug(GEMMA, 896, "legacy", "minimal") == "gemma-3-4b-it_px896_legacy_minimal"


def test_every_axis_changes_the_key():
    base = tb.slug(QWEN, None)
    assert len({base, tb.slug(QWEN, 448), tb.slug(QWEN, None, "legacy"),
                tb.slug(QWEN, None, bank="minimal"), tb.slug(GEMMA, None)}) == 5


def test_key_suffix_matches_the_tail_slug_produces():
    for style in ("chat", "legacy"):
        for bank in ("full", "minimal"):
            key = tb.slug(QWEN, 448, style, bank)
            assert key.endswith(tb._key_suffix(style, bank))
            assert key == "qwen3-vl-8b-instruct_px448" + tb._key_suffix(style, bank)


def test_base_runs_for_never_crosses_banks(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "STAGE_F_DIR", tmp_path)
    for name in ("conflict_qwen3-vl-8b-instruct_metrics.json",
                 "conflict_qwen3-vl-8b-instruct_px448_metrics.json",
                 "conflict_qwen3-vl-8b-instruct_minimal_metrics.json",
                 "conflict_qwen3-vl-8b-instruct_px448_minimal_metrics.json"):
        (tmp_path / name).write_text("{}")

    full = {p.name for p in tb._base_runs_for(QWEN, "chat", "full")}
    minimal = {p.name for p in tb._base_runs_for(QWEN, "chat", "minimal")}
    assert full == {"conflict_qwen3-vl-8b-instruct_metrics.json",
                    "conflict_qwen3-vl-8b-instruct_px448_metrics.json"}
    # the budget tag precedes the bank tail, so this only matches if the glob is built from parts
    assert minimal == {"conflict_qwen3-vl-8b-instruct_minimal_metrics.json",
                       "conflict_qwen3-vl-8b-instruct_px448_minimal_metrics.json"}
    assert not full & minimal


def test_both_banks_supply_the_same_condition_structure():
    # The text-only control runs one forward per sentence, so the two banks must be comparable in
    # shape: 6 of each polarity plus the neutral baselines the corrected readouts are measured against.
    for bank in ("full", "minimal"):
        conds = [c for c, _, _ in tb._conditions(bank)]
        assert conds.count("positive") == 6
        assert conds.count("negative") == 6
        assert conds.count("neutral") == 2
        assert conds.count("none") == 1


def test_minimal_bank_is_token_matched_pairs():
    minimal = tb._conditions("minimal")
    pos = [s for c, _, s in minimal if c == "positive"]
    neg = [s for c, _, s in minimal if c == "negative"]
    # each pair differs in exactly one word — that is what makes it a valence-only contrast
    for p, n in zip(pos, neg):
        pw, nw = p.split(), n.split()
        assert len(pw) == len(nw), f"length mismatch: {p!r} vs {n!r}"
        assert sum(a != b for a, b in zip(pw, nw)) == 1, f"not one-word-apart: {p!r} vs {n!r}"
