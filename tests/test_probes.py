import numpy as np
import pytest

from src.probes.train import (
    fit_appraisal_probes,
    fit_ridge,
    unique_effect_vector,
    unique_effect_vectors,
)
from src.probes.evaluate import best_layer, layerwise_r2, probe_r2, r2


def test_fit_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    w = rng.standard_normal(8).astype(np.float32)
    y = X @ w + 0.5
    coef, intercept = fit_ridge(X, y, alpha=1e-3)
    assert probe_r2(X, y, coef, intercept) > 0.99


def test_unique_effect_vector_is_orthogonal_to_others():
    # v_a has a component along `other`; the unique-effect vector must be orthogonal to it.
    other = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_a = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    z = unique_effect_vector(v_a, other[None, :])
    assert np.isclose(np.linalg.norm(z), 1.0, atol=1e-5)
    assert abs(float(z @ other)) < 1e-5


def test_unique_effect_vector_collapse_raises():
    other = np.array([[1.0, 0.0]], dtype=np.float32)
    v_a = np.array([2.0, 0.0], dtype=np.float32)  # fully in span(other)
    with pytest.raises(ValueError):
        unique_effect_vector(v_a, other)


def test_unique_effect_vectors_shape_and_unit_norm():
    coef = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    Z = unique_effect_vectors(coef)
    assert Z.shape == coef.shape
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-5)


def test_fit_appraisal_probes_end_to_end():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((150, 6)).astype(np.float32)
    Y = {
        "pleasantness": X @ rng.standard_normal(6),
        "suddenness": X @ rng.standard_normal(6),
    }
    probes = fit_appraisal_probes(X, Y, alphas=(0.01, 0.1, 1.0))
    assert probes.names == ["pleasantness", "suddenness"]
    assert probes.coef.shape == (2, 6)
    assert probes.z_unit.shape == (2, 6)
    assert np.allclose(np.linalg.norm(probes.z_unit, axis=1), 1.0, atol=1e-5)
    assert probes.steering_vector("pleasantness").shape == (6,)


def test_r2_zero_variance_target():
    assert r2(np.ones(5), np.ones(5)) == 0.0


def test_layerwise_r2_and_best_layer():
    rng = np.random.default_rng(2)
    w = rng.standard_normal(5).astype(np.float32)
    # layer 1 is informative; layer 0 is noise.
    X_good = rng.standard_normal((120, 5)).astype(np.float32)
    y = X_good @ w
    per_layer = {0: rng.standard_normal((120, 5)).astype(np.float32), 1: X_good}
    from functools import partial

    lr2 = layerwise_r2(per_layer, y, partial(fit_ridge, alpha=1e-3))
    layer, score = best_layer(lr2)
    assert layer == 1 and score > 0.9
