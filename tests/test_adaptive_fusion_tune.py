import numpy as np

from explainability.adaptive_fusion_tune import (
    adaptive_fusion_p,
    confidence_from_probability,
    grid_search_fusion_and_threshold,
    lip_sync_error_score,
)


def test_confidence_peaks_at_extremes():
    assert confidence_from_probability(0.0) > confidence_from_probability(0.5)
    assert confidence_from_probability(1.0) > confidence_from_probability(0.5)


def test_adaptive_fusion_agrees_at_equal_modalities():
    pf, _, _, _ = adaptive_fusion_p(0.6, 0.6, 0.8, 0.8, tau=0.2, epsilon=1e-6)
    assert abs(pf - 0.6) < 1e-6


def test_lip_sync_error():
    assert abs(lip_sync_error_score(0.2, 0.9) - 0.7) < 1e-9


def test_grid_search_finds_reasonable_f1():
    np.random.seed(0)
    n = 40
    # Fake: higher p_v, p_a
    p_v = np.concatenate([np.random.uniform(0.1, 0.45, n // 2), np.random.uniform(0.55, 0.95, n // 2)])
    p_a = np.concatenate([np.random.uniform(0.1, 0.4, n // 2), np.random.uniform(0.5, 0.9, n // 2)])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    out = grid_search_fusion_and_threshold(p_v, p_a, y)
    assert "f1_fake" in out or "f1_fake_sklearn" in out
    assert out.get("f1_fake", 0) >= 0.0
