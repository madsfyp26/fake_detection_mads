"""Tests for Gemini UI guide payload builder."""

import pandas as pd
import pytest

from ui.report_explain_payload import (
    UI_GLOSSARY,
    build_combined_report_guide_payload,
)


def test_glossary_non_empty():
    assert "gradcam" in UI_GLOSSARY
    assert len(UI_GLOSSARY) >= 5


def test_build_payload_minimal():
    p = build_combined_report_guide_payload({"cmid_status": "missing_embeddings"}, None, use_unsup_avh=False)
    assert p["run_flags"]["cmid_status"] == "missing_embeddings"
    assert "run_values" in p and "ui_glossary" in p


def test_build_xai_standalone_payload_audio():
    from ui.report_explain_payload import build_xai_standalone_payload

    p = build_xai_standalone_payload(
        "audio",
        {"noma_confidence_instability": {"CII": 0.01, "variance_per_time": [0.1, 0.2]}},
        None,
    )
    assert p["page"] == "audio_explainability"
    assert p["run_values"]["has_cii_timeline"] is True


def test_build_xai_standalone_payload_video():
    from ui.report_explain_payload import build_xai_standalone_payload

    p = build_xai_standalone_payload("video", {}, {"cam_per_frame": [0.1, 0.2], "roi_fps": 25})
    assert p["run_flags"]["has_cam_intensity_line"] is True


def test_build_payload_with_noma_df():
    df = pd.DataFrame({"Seconds": [0, 1], "p_fake": [0.2, 0.4], "Prediction": ["Real", "Fake"]})
    p = build_combined_report_guide_payload({"noma_df": df, "p_fused": 0.55}, {}, use_unsup_avh=True)
    assert p["run_flags"]["use_unsup_avh"] is True
    assert p["run_values"]["noma_blocks"] == 2
    assert p["run_values"]["noma_mean_p_fake"] == pytest.approx(0.3)
