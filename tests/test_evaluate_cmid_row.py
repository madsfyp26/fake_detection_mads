"""Unit tests for evaluate_video_folder CMID / Grad-CAM row helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.evaluate_video_folder import _cmid_cam_extra, _row_from_result


def test_cmid_cam_extra_from_list():
    res = {
        "cmid_status": "ok",
        "cmid": {"cmid": [0.1, 0.2, 0.3]},
        "cam_ok": True,
        "cam_overlays_dir": "/tmp/cam",
        "late_fusion_mode": "full",
    }
    extra = _cmid_cam_extra(res)
    assert extra["cmid_mean"] == pytest.approx(0.2)
    assert extra["cmid_max"] == pytest.approx(0.3)
    assert extra["cam_ok"] is True
    assert extra["cam_overlays_dir"] == "/tmp/cam"


def test_row_from_result_merges_cmid_columns():
    res = {
        "avh_ok": True,
        "avh_error": None,
        "avh_score": 0.5,
        "p_avh_cal": 0.4,
        "p_audio_mean_raw": 0.3,
        "p_audio_mean": 0.35,
        "p_fused": 0.38,
        "fusion_verdict": "Uncertain",
        "fusion_regime": "blend",
        "cmid_status": "ok",
        "cmid": {"cmid": [1.0, 2.0]},
        "cam_ok": False,
        "cam_overlays_dir": None,
        "late_fusion_mode": "full",
        "noma_df": None,
    }
    row = _row_from_result("/v/a.mp4", None, res)
    assert row["cmid_status"] == "ok"
    assert row["cmid_mean"] == pytest.approx(1.5)
    assert row["late_fusion_mode"] == "full"
    assert "p_fused" in row
