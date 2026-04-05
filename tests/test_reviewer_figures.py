"""Tests for explainability/reviewer_figures.py (no real media files)."""

import numpy as np


def test_figure_calibration_png_bytes():
    from explainability.reviewer_figures import figure_calibration_png_bytes

    b = figure_calibration_png_bytes()
    assert isinstance(b, bytes) and len(b) > 500


def test_figure_cmid_png_bytes():
    from explainability.reviewer_figures import figure_cmid_png_bytes

    b = figure_cmid_png_bytes(
        {"similarity": [0.9, 0.8, 0.85], "cmid": [0.01, 0.05, 0.02]},
    )
    assert len(b) > 500


def test_figure_attention_cam_png_bytes():
    from explainability.reviewer_figures import figure_attention_cam_png_bytes

    b = figure_attention_cam_png_bytes(
        {
            "attention_per_frame": [0.2, 0.5, 0.4],
            "cam_per_frame": [0.1, 0.6, 0.3],
            "roi_fps": 25.0,
        }
    )
    assert len(b) > 500


def test_inferno_and_triptych_synthetic(tmp_path):
    """Triptych with tiny synthetic video requires opencv."""
    import cv2
    from explainability.reviewer_figures import figure_triptych_png_bytes

    roi = tmp_path / "roi.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(roi), fourcc, 5.0, (64, 48))
    for _ in range(5):
        fr = np.zeros((48, 64, 3), dtype=np.uint8)
        fr[:] = (40, 80, 120)
        vw.write(fr)
    vw.release()

    fused = np.random.randn(5, 8, 8).astype(np.float32)
    fused_path = tmp_path / "f.npy"
    np.save(fused_path, fused)

    b = figure_triptych_png_bytes(
        roi_path=str(roi),
        frame_idx=2,
        fused_npy_path=str(fused_path),
        cam_npy_path=None,
        overlay_dir=None,
    )
    assert len(b) > 200
