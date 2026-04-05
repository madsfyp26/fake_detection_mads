import numpy as np

from explainability.temporal_corroboration import (
    aggregate_saliency_to_noma_bins,
    cam_idx_to_saliency_timeseries,
    compute_temporal_corroboration,
    compute_tension_index,
)


def test_tension_index():
    assert abs(compute_tension_index(0.8, 0.2) - 0.6) < 1e-9


def test_cam_idx_from_cam_per_frame():
    cam_idx = {
        "cam_per_frame": [0.1, 0.9, 0.5],
        "cam_to_roi_index": [0, 25, 50],
        "roi_fps": 25.0,
    }
    ts = cam_idx_to_saliency_timeseries(cam_idx)
    assert ts is not None
    t, s = ts
    assert len(t) == 3
    assert s.min() >= 0 and s.max() <= 1 + 1e-6


def test_aggregate_saliency_bins():
    secs = np.array([0.0, 1.0])
    t_sal = np.array([0.1, 0.5, 1.2])
    sal = np.array([0.2, 1.0, 0.3])
    out = aggregate_saliency_to_noma_bins(secs, t_sal, sal, block_width=1.0)
    assert out.shape == (2,)
    assert out[0] >= 1.0  # max in [0,1)
    assert out[1] >= 0.3


def test_compute_corroboration_ok():
    cam_idx = {
        "cam_per_frame": [0.0, 0.0, 1.0, 1.0],
        "cam_to_roi_index": [0, 10, 25, 35],
        "roi_fps": 25.0,
    }
    secs = np.array([0.0, 1.0])
    p = np.array([0.9, 0.2])
    r = compute_temporal_corroboration(
        noma_seconds=secs,
        p_fake_calibrated=p,
        cam_idx=cam_idx,
        p_threshold=0.5,
        sal_threshold=0.3,
    )
    assert r["status"] == "ok"
    assert r["corroboration_rate"] >= 0.0
    assert len(r["bins"]) == 2
    assert "corroboration" in r["bins"][0]
