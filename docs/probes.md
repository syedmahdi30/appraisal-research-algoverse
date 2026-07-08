# Appraisal probes and metrics

## Read-out
- Per example, cache the **last-token** hidden state at each tap on the **language-model** blocks:
  `blocks.{i}.hook_resid_post`, `blocks.{i}.hook_attn_out`, `blocks.{i}.hook_mlp_out`.
- Cast cached **bf16** activations to **fp32** before sklearn. Store training matrices as fp32 `.npy`.
- Feature matrix `X`: `[n_examples, d_model]` (d_model = 2560 for Gemma 3 4B).

## Ridge probe (per appraisal a)
Fit `Ridge` to predict the 1–5 appraisal rating from the activation:
```python
va = Ridge(alpha=1.0).fit(X, y_a).coef_    # r_a = v_a · x + b_a
```
- Primary metric: **r²** on held-out data (val for model selection, test for the final number).
- Report **layer-wise** r² per tap; expect the Tak-style **mid-layer, MHSA-dominant** peak.

## Unique-effect (steering) vector
Remove the components shared with the OTHER appraisal directions so steering isolates appraisal a:
```python
# V_other: [k, d_model] stack of the other appraisals' Ridge coef vectors
P   = V_other.T @ np.linalg.pinv(V_other @ V_other.T) @ V_other   # projector onto span(others)
z_a = (np.eye(d) - P) @ va                                        # unique-effect direction
z_a_unit = z_a / np.linalg.norm(z_a)                              # normalize for steering
```

## Steering
Add `beta * z_a_unit` to the residual stream at the last position, at the Stage-A critical layer:
```python
act[:, pos, :] = act[:, pos, :] + beta * z_a_unit.to(act.dtype)
```
Sweep **β ∈ {±1, ±2, ±4}**. Success = the emotion-label logit distribution shifts in the
appraisal-theory-predicted direction (not just any noisy change).

## Metrics summary
- Closed-vocab emotion **accuracy** (before correctness filtering).
- Per-appraisal probe **r²**, layer-wise, per tap.
- **Transfer gap** (Stage C) = text-probe r² on text activations − on image activations.
- Steering **directional effect** = signed shift in target-emotion logit/probability vs. β.

## Controls / validity
- Norm-matched **random** direction (same ‖·‖ as `z_a`) — steering effect must beat this.
- **Shuffled-label** probe — r² must beat this near-zero baseline.
- **Caption baseline** for Stage C — if caption pipeline succeeds where direct read-out fails,
  the effect is verbalization-mediated, not shared geometry.

## Discipline
- Clean train/val/test separation; select `alpha` and layer on val, report on test only.
- Never re-fit probes on image-side data in Stage C — directions are **frozen** from Stage A.
- Do not claim replication/transfer from a single positive metric or a cherry-picked example.
