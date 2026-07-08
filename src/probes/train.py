"""Ridge appraisal probes and unique-effect (steering) vectors.

Method: docs/probes.md. Cached bf16 activations must be cast to fp32 (done here) before
sklearn. Per appraisal a: fit Ridge to get v_a = coef; the unique-effect vector removes
the components shared with the other appraisals, isolating a's direction:

    z_a = (I - P_-a) v_a,   P_-a = projector onto span(other appraisal vectors)
    z_a_unit = z_a / ||z_a||
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

# Default alpha grid for RidgeCV. Activations are ~2560-dim and collinear, so a fixed
# small alpha is ill-conditioned; standardizing + CV-selecting alpha fixes both.
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0)


@dataclass
class AppraisalProbes:
    """Fitted per-appraisal Ridge probes and their steering vectors.

    - `names`: appraisal order (rows of `coef` and `z_unit` follow it)
    - `coef`:  [k, d] raw Ridge coefficient vectors v_a
    - `intercept`: [k] Ridge intercepts
    - `z_unit`: [k, d] unit-norm unique-effect steering vectors
    - `alpha`: Ridge regularization used
    """
    names: list[str]
    coef: np.ndarray
    intercept: np.ndarray
    z_unit: np.ndarray
    alpha: float = 1.0
    meta: dict = field(default_factory=dict)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def steering_vector(self, name: str) -> np.ndarray:
        return self.z_unit[self.index(name)]


def _fold_scaler(coef_std, intercept_std, scaler) -> tuple[np.ndarray, float]:
    """Fold a StandardScaler into (coef, intercept) so they apply to RAW X.

    Ridge fits on z=(x-mean)/scale: pred = z·w + b0. In raw space that is
    x·(w/scale) + (b0 - mean·(w/scale)). Returning raw-space coefficients keeps the probe
    interface (apply to raw activations) and keeps steering vectors in activation space.
    """
    scale = scaler.scale_.copy()
    scale[scale == 0] = 1.0  # constant feature -> zero contribution
    coef_raw = (coef_std / scale).astype(np.float32)
    intercept_raw = float(intercept_std - np.sum(scaler.mean_ * coef_std / scale))
    return coef_raw, intercept_raw


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """Standardize, fit Ridge at fixed `alpha`, return raw-space (coef [d], intercept)."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y)
    return _fold_scaler(model.coef_, float(model.intercept_), scaler)


def fit_ridge_cv(X: np.ndarray, y: np.ndarray, alphas=DEFAULT_ALPHAS) -> tuple[np.ndarray, float, float]:
    """Standardize + RidgeCV (alpha chosen by efficient LOO/GCV). Returns (coef, intercept, alpha).

    Standardization + a data-driven alpha removes the ill-conditioning that a fixed small
    alpha causes on high-dim collinear activations.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    model = RidgeCV(alphas=list(alphas)).fit(scaler.transform(X), y)
    coef, intercept = _fold_scaler(model.coef_, float(model.intercept_), scaler)
    return coef, intercept, float(model.alpha_)


def unique_effect_vector(v_a: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Remove from v_a the components spanned by `others` ([k, d]); return unit vector.

    Uses the pseudo-inverse projector P = O^T (O O^T)^+ O onto span(others), robust to
    rank-deficient / collinear appraisal directions.
    """
    v_a = np.asarray(v_a, dtype=np.float32)
    others = np.asarray(others, dtype=np.float32)
    d = v_a.shape[0]
    if others.size == 0:
        z = v_a
    else:
        P = others.T @ np.linalg.pinv(others @ others.T) @ others  # [d, d]
        z = (np.eye(d, dtype=np.float32) - P) @ v_a
    norm = np.linalg.norm(z)
    if norm == 0:
        raise ValueError("unique-effect vector collapsed to zero (v_a fully in span of others)")
    return (z / norm).astype(np.float32)


def unique_effect_vectors(coef: np.ndarray) -> np.ndarray:
    """Vectorized: for each row a of `coef` [k, d], build its unit unique-effect vector."""
    coef = np.asarray(coef, dtype=np.float32)
    k = coef.shape[0]
    out = np.empty_like(coef)
    for a in range(k):
        others = np.delete(coef, a, axis=0)
        out[a] = unique_effect_vector(coef[a], others)
    return out


def fit_appraisal_probes(
    X: np.ndarray,
    Y: dict[str, np.ndarray],
    alphas=DEFAULT_ALPHAS,
) -> AppraisalProbes:
    """Fit one RidgeCV probe per appraisal and assemble unique-effect steering vectors.

    `X`: [n, d] last-token activations. `Y`: {appraisal_name: [n] ratings}. Alpha is chosen
    per appraisal by CV. The steering vectors for appraisal a exclude the other appraisals
    in `Y` (the unique effect).
    """
    names = list(Y.keys())
    coefs, intercepts, chosen = [], [], []
    for name in names:
        c, b, a = fit_ridge_cv(X, Y[name], alphas=alphas)
        coefs.append(c)
        intercepts.append(b)
        chosen.append(a)
    coef = np.stack(coefs)
    z_unit = unique_effect_vectors(coef)
    return AppraisalProbes(
        names=names,
        coef=coef,
        intercept=np.asarray(intercepts, dtype=np.float32),
        z_unit=z_unit,
        alpha=float(np.median(chosen)),
        meta={"alphas_per_appraisal": dict(zip(names, chosen))},
    )
