import os
import sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from explainability.gradcam_avh import normalize_cam_volume, compute_windowed_fusion
from explainability.gradcam_selection import select_top_cam_frames

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def test_normalize_cam_volume_squeezes_singleton_batch():
    cam_in = np.random.rand(1, 8, 16, 16).astype(np.float32)
    cam, err = normalize_cam_volume(cam_in)
    assert err is None
    assert cam is not None
    assert cam.shape == (8, 16, 16)


def test_normalize_cam_volume_rejects_non_3d():
    cam_in = np.random.rand(16, 16).astype(np.float32)
    cam, err = normalize_cam_volume(cam_in)
    assert cam is None
    assert isinstance(err, str)
    assert err.startswith("bad_cam_shape:")


def test_select_top_cam_frames_diverse_enforces_gap():
    vals = np.array([0.9, 0.85, 0.8, 0.4, 0.3, 0.2], dtype=float)
    idx = select_top_cam_frames(vals, top_k=2, mode="diverse_topk", min_temporal_gap=2)
    assert len(idx) == 2
    assert abs(idx[0] - idx[1]) >= 2


@pytest.mark.skipif(not HAS_CV2, reason="opencv unavailable")
def test_compute_windowed_fusion_covers_all_frames():
    rng = np.random.default_rng(0)
    cam = rng.normal(size=(21, 12, 12)).astype(np.float32)
    frames = (rng.uniform(0, 255, size=(21, 24, 24))).astype(np.uint8)
    fused, freq_stats = compute_windowed_fusion(cam, frames, window_size=8)
    assert fused.shape == cam.shape
    assert "high_freq_energy" in freq_stats
    assert len(freq_stats["high_freq_energy"]) == cam.shape[0]
