"""Unit tests for reliability-weighted late fusion."""

from explainability.reliability_fusion import (
    compute_reliability_fusion,
    compute_simple_late_fusion,
)


def test_fusion_equal_branches_identity():
    tau = 0.12
    out = compute_reliability_fusion(0.7, 0.7, tau)
    assert out["fusion_tension"] == 0.0
    assert out["fusion_w_audio"] == 1.0
    assert abs(out["p_fused"] - 0.7) < 1e-9
    assert out["fusion_regime"] == "reliability_blend"


def test_w_audio_decreases_with_tension():
    tau = 0.12
    low = compute_reliability_fusion(0.5, 0.51, tau)
    high = compute_reliability_fusion(0.45, 0.62, tau)
    assert low["fusion_tension"] < high["fusion_tension"]


def test_regime_saturation_trust_audio():
    tau = 0.12
    out = compute_reliability_fusion(0.32, 0.998, tau)
    assert out["fusion_regime"] == "saturation_trust_audio"
    assert out["p_fused"] < 0.42


def test_regime_lip_mismatch_trust_video():
    tau = 0.12
    out = compute_reliability_fusion(0.22, 0.687, tau)
    assert out["fusion_regime"] == "lip_mismatch_trust_video"
    assert out["fusion_verdict"] == "Likely FAKE"
    assert out["p_fused"] > 0.60


def test_regime_inverted_avh_noma_dog_style():
    """Lip model very confident 'real', NOMA moderately suspicious — flag fake."""
    tau = 0.12
    out = compute_reliability_fusion(0.294, 0.103, tau)
    assert out["fusion_regime"] == "inverted_avh_noma_mismatch"
    assert out["fusion_verdict"] == "Likely FAKE"


def test_regime_journalist_real():
    """High AVH but not saturated; mid NOMA — typical broadcast talking head."""
    tau = 0.12
    out = compute_reliability_fusion(0.515, 0.966, tau)
    assert out["fusion_regime"] == "high_avh_mid_audio_journalist_real"
    assert out["fusion_verdict"] == "Likely REAL"
    assert out["p_fused"] < 0.40


def test_regime_both_elevated_trust_video_mid_band():
    """Mid–high NOMA with saturated AVH uses trust-video path (not complement)."""
    tau = 0.12
    out = compute_reliability_fusion(0.599, 0.9998, tau)
    assert out["fusion_regime"] == "both_elevated_trust_video_mid"
    assert out["p_fused"] > 0.85


def test_regime_both_elevated_complement_high_noma():
    """Complement when NOMA mean is clearly high (>= 0.60)."""
    tau = 0.12
    out = compute_reliability_fusion(0.65, 0.9998, tau)
    assert out["fusion_regime"] == "both_elevated_complement"
    assert out["p_fused"] < 0.45


def test_borderline_noma_synthetic_not_complement():
    """~0.52 NOMA + saturated AVH uses reliability blend (often FAKE), not complement REAL."""
    tau = 0.12
    out = compute_reliability_fusion(0.519, 0.998, tau)
    assert out["fusion_regime"] != "both_elevated_complement"


def test_verdict_uncertain_in_band():
    tau = 0.12
    out = compute_reliability_fusion(0.5, 0.505, tau)
    assert out["fusion_verdict"] == "Uncertain"


def test_ambiguous_audio_high_avh_whatsapp():
    tau = 0.12
    out = compute_reliability_fusion(0.486, 0.923, tau)
    assert out["fusion_verdict"] == "Uncertain"


def test_simple_mean():
    tau = 0.12
    out = compute_simple_late_fusion("mean", 0.6, 0.4, tau)
    assert out["fusion_regime"] == "simple_mean"
    assert abs(out["p_fused"] - 0.5) < 1e-9
    assert out["fusion_w_audio"] == 0.5


def test_simple_audio_primary():
    tau = 0.12
    out = compute_simple_late_fusion("audio_primary", 0.72, 0.1, tau)
    assert out["fusion_regime"] == "audio_primary"
    assert out["p_fused"] == 0.72
    assert out["fusion_w_audio"] == 1.0


def test_simple_video_primary():
    tau = 0.12
    out = compute_simple_late_fusion("video_primary", 0.2, 0.88, tau)
    assert out["fusion_regime"] == "video_primary"
    assert out["p_fused"] == 0.88
    assert out["fusion_w_audio"] == 0.0
