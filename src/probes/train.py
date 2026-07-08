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
from sklearn.linear_model import Ridge


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


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """Fit Ridge, returning (coef [d], intercept). Casts X, y to fp32."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    model = Ridge(alpha=alpha).fit(X, y)
    return model.coef_.astype(np.float32), float(model.intercept_)


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
    alpha: float = 1.0,
) -> AppraisalProbes:
    """Fit one Ridge per appraisal and assemble unique-effect steering vectors.

    `X`: [n, d] last-token activations. `Y`: {appraisal_name: [n] ratings}. The steering
    vectors for appraisal a exclude the other appraisals in `Y` (the unique effect).
    """
    names = list(Y.keys())
    coefs, intercepts = [], []
    for name in names:
        c, b = fit_ridge(X, Y[name], alpha=alpha)
        coefs.append(c)
        intercepts.append(b)
    coef = np.stack(coefs)
    z_unit = unique_effect_vectors(coef)
    return AppraisalProbes(
        names=names,
        coef=coef,
        intercept=np.asarray(intercepts, dtype=np.float32),
        z_unit=z_unit,
        alpha=alpha,
    )
