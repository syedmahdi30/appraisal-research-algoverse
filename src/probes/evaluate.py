"""Probe evaluation: r2 and layer-wise sweeps (docs/probes.md).

A frozen probe (coef, intercept) is applied to a new activation matrix — this is exactly
the Stage C read-out, where text-trained probes score image-conditioned activations.
"""
from __future__ import annotations

import numpy as np


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Returns 0.0 when the target has zero variance."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Apply a linear probe: X @ coef + intercept."""
    X = np.asarray(X, dtype=np.float32)
    return X @ np.asarray(coef, dtype=np.float32) + np.float32(intercept)


def probe_r2(X: np.ndarray, y: np.ndarray, coef: np.ndarray, intercept: float) -> float:
    """r2 of a fixed (frozen) probe on (X, y). Used for the Stage C transfer gap."""
    return r2(y, predict(X, coef, intercept))


def layerwise_r2(
    per_layer_X: dict[int, np.ndarray],
    y: np.ndarray,
    fit_fn,
) -> dict[int, float]:
    """Fit+score a probe per layer; return {layer: r2}.

    `per_layer_X`: {layer_index: [n, d]}. `fit_fn(X, y) -> (coef, intercept)` (e.g. a
    partial of `probes.train.fit_ridge`). Intended for held-out data so the reported r2
    reflects generalization, not fit.
    """
    out: dict[int, float] = {}
    for layer, X in sorted(per_layer_X.items()):
        coef, intercept = fit_fn(X, y)
        out[layer] = probe_r2(X, y, coef, intercept)
    return out


def best_layer(layer_r2: dict[int, float]) -> tuple[int, float]:
    """Return (layer, r2) of the highest-scoring layer."""
    layer, score = max(layer_r2.items(), key=lambda kv: kv[1])
    return layer, score
