"""Tests for slide/demo proof snippets."""

from explainability.panel_proof import (
    build_panel_proof_bundle,
    build_panel_proof_markdown,
    corroboration_proof_snippet,
    fusion_proof_chain,
    gradcam_status_proof,
)


def test_fusion_proof_chain_ok():
    res = {
        "p_audio_mean": 0.7,
        "p_avh_cal": 0.4,
        "fusion_tau": 0.08,
        "fusion_tension": 0.3,
        "fusion_w_audio": 0.023,
        "p_fused": 0.41,
        "fusion_verdict": "Uncertain",
        "late_fusion_mode": "full",
    }
    out = fusion_proof_chain(res)
    assert out["status"] == "ok"
    assert "steps_plain" in out and len(out["steps_plain"]) >= 4


def test_fusion_proof_chain_simple_mean():
    res = {
        "late_fusion_mode": "mean",
        "p_audio_mean": 0.6,
        "p_avh_cal": 0.4,
        "fusion_tau": 0.1,
        "fusion_tension": 0.2,
        "p_fused": 0.5,
        "fusion_verdict": "Uncertain",
    }
    out = fusion_proof_chain(res)
    assert out["status"] == "ok"
    assert "simple mean" in out["steps_plain"][0].lower()
    assert out["formula"].startswith("p_fused = 0.5")


def test_corroboration_snippet():
    res = {
        "temporal_corroboration": {
            "status": "ok",
            "corroboration_rate": 0.2,
            "conflict_rate": 0.1,
            "bins": [
                {"second": 1.0, "p_fake": 0.8, "saliency": 0.9, "corroboration": True, "conflict": False},
                {"second": 3.0, "p_fake": 0.85, "saliency": 0.1, "corroboration": False, "conflict": True},
            ],
            "p_threshold": 0.5,
            "sal_threshold": 0.5,
        }
    }
    c = corroboration_proof_snippet(res)
    assert c["status"] == "ok"
    assert "Example corroboration" in c["headline"] or c.get("best_corroboration_bin")


def test_gradcam_status_proof():
    g = gradcam_status_proof({"xai_status": {"temporal_inconsistency": "computed"}, "T_use": 50})
    assert g["status"] == "ok"
    assert "Temporal inconsistency" in g["headline"]


def test_build_bundle_markdown_non_empty():
    res = {"p_audio_mean": 0.5, "p_avh_cal": 0.5, "fusion_tau": 0.1, "fusion_tension": 0.0, "p_fused": 0.5}
    b = build_panel_proof_bundle(res, None)
    assert "markdown_appendix" in b
    md = build_panel_proof_markdown(res, None)
    assert "Reliability" in md or "fusion" in md.lower()
