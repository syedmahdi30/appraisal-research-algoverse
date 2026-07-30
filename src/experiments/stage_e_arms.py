"""Stage E arm definitions — the appraisal-combination steering conditions (pilot-plan-stage-e-f.md).

Pure Python (no torch / no model import) so both the GPU combo runner (stage_e_combo.py) and the
CPU analysis scripts (analyze_appraisal_profiles.py, analyze_stage_e.py) share one source of
truth for what each arm is and which SPECIFIC emotion appraisal theory predicts it should raise.

Each arm's `combo` is a list of (appraisal, sign). Congruent arms (A1-A5) carry a theory-predicted
target (+ optional alternative + a "must-not" negative prediction). Control arms:
  N1, N2  — congruent-form controls (valence-flipped / weak-target sanity).
  S1-S6   — each appraisal alone ("does the COMBINATION beat its components?").
  R       — sum of two orthogonal random unit vectors (perturbation null); built in the runner.
  A1-raw  — raw Δμ sum for A1, unrescaled (magnitude-sensitivity check); built in the runner.
"""
from __future__ import annotations

APPRAISALS = (
    "pleasantness", "unpleasantness", "suddenness",
    "predict_event", "self_responsblt", "other_responsblt",
)

# name -> {combo, target, alt, must_not, kind, note}. target/alt/must_not are emotion labels.
ARMS: dict[str, dict] = {
    # --- congruent theory arms (Smith & Ellsworth 1985 style) ---
    "A1": {"combo": [("unpleasantness", +1), ("other_responsblt", +1)],
           "target": "anger", "alt": None, "must_not": ["guilt", "shame"],
           "kind": "congruent", "note": "+unpleasant +other-responsibility -> anger"},
    "A2": {"combo": [("unpleasantness", +1), ("self_responsblt", +1)],
           "target": "guilt", "alt": "shame", "must_not": ["anger"],
           "kind": "congruent", "note": "+unpleasant +self-responsibility -> guilt (shame runner-up)"},
    "A3": {"combo": [("unpleasantness", +1), ("suddenness", +1)],
           "target": "fear", "alt": "surprise", "must_not": ["pride"],
           "kind": "congruent", "note": "+unpleasant +sudden -> fear (surprise runner-up)"},
    "A4": {"combo": [("pleasantness", +1), ("self_responsblt", +1)],
           "target": "pride", "alt": None, "must_not": ["guilt"],
           "kind": "congruent", "note": "+pleasant +self-responsibility -> pride"},
    "A5": {"combo": [("pleasantness", +1), ("suddenness", +1)],
           "target": "surprise", "alt": "joy", "must_not": ["sadness"],
           "kind": "congruent", "note": "+pleasant +sudden -> surprise or joy"},
    # --- congruent-form controls ---
    "N1": {"combo": [("pleasantness", +1), ("other_responsblt", +1)],
           "target": None, "alt": None, "must_not": ["anger"],
           "kind": "control", "note": "valence-flipped A1 — must NOT raise anger"},
    "N2": {"combo": [("unpleasantness", +1), ("predict_event", +1)],
           "target": None, "alt": None, "must_not": [],
           "kind": "control", "note": "weak/no specific target expected — sanity"},
}

# S1-S6: each appraisal alone (the "combination beats components" reference).
for _i, _a in enumerate(APPRAISALS, start=1):
    ARMS[f"S{_i}"] = {"combo": [(_a, +1)], "target": None, "alt": None, "must_not": [],
                      "kind": "single", "note": f"single appraisal: +{_a}"}

# R and A1-raw are synthesized in the runner (need the raw Δμ / RNG), listed for run order.
SPECIAL_ARMS = ("R", "A1-raw")

# The two-component congruent + control arms whose empirical target E2 cross-checks vs theory.
COMBO_ARMS = ("A1", "A2", "A3", "A4", "A5", "N1", "N2")
CONGRUENT_ARMS = ("A1", "A2", "A3", "A4", "A5")


def arm_order() -> list[str]:
    """Canonical arm order for the pilot (15 arms): congruent, controls, singles, specials."""
    return list(COMBO_ARMS) + [f"S{i}" for i in range(1, len(APPRAISALS) + 1)] + list(SPECIAL_ARMS)
