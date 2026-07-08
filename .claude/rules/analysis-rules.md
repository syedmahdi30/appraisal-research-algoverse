# Rules — analysis, metrics, reporting

- **Select on val, report on test.** Tune Ridge `alpha` and pick the critical layer on validation
  only. The final r² / accuracy / transfer numbers come from the test split, reported once.
- **Always report baselines next to effects.** No probe r² without its shuffled-label baseline; no
  steering effect without its norm-matched random-direction baseline; no Stage C claim without the
  caption baseline. A bare positive number is not a result.
- **Careful transfer language.** "Supports transfer" only if it beats the relevant baselines;
  "inconclusive" if weak/unstable/confounded; "fails to support transfer" if it doesn't survive
  controls. Never upgrade the verdict from a single metric.
- **Comparable units only.** Don't put runs in one table unless split, metric definition, hook/layer,
  data slice, and steering strength match. If they differ, separate the tables and label the mismatch.
- **Don't bury nulls.** Negative and null results are reported as prominently as positive ones. The
  transfer gap is a headline number whether large or small.
- **Figures must be self-describing.** Every plot states axes, units, split, and subset. Prefer a few
  defensible comparisons (best layer, baseline vs probe vs steering, text vs image r²) over many weak ones.
- **Reproducible artifacts.** Persist config + metrics (`results/<stage>/metrics.json`), probe objects,
  and figures so `/analyze-results` can consume them without recomputing the stage. Record the git
  state / package versions used for each run.
- **Paper-ready output.** When asked, emit LaTeX-friendly tables and a threats-to-validity summary
  (lossy mapping, domain shift, caption confound, tokenization) — do not imply causation from
  correlational summaries.
