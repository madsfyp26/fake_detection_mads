import json
import math
import os

from config import PROJECT_ROOT


DEFAULTS = {
    # AVH score -> p(fake) mapping is logistic(sigmoid) by default.
    # p_fake = sigmoid((score - bias) / temperature)
    "avh_temperature": 1.0,
    "avh_bias": 0.0,
    # Unsupervised AVH (test_video_unsupervised.py) uses a different score scale than
    # supervised Fusion logits — do NOT reuse avh_temperature/avh_bias for that path.
    # p_fake ≈ sigmoid((score - center) / scale); center ~ typical "neutral" mismatch.
    "avh_unsup_center": 1.15,
    "avh_unsup_scale": 0.85,
    # Uncertainty band for three-way verdict: p_fused in (0.5±tau) => "Uncertain".
    # Slightly tighter than 0.12 so fewer clips stay in the uncertain bucket (more decisive labels).
    "avh_uncertainty_margin": 0.085,
    "noma_uncertainty_margin": 0.085,
    # Reliability fusion (see explainability/reliability_fusion.py):
    # Pull AVH p(fake) toward 0.5 before blending to reduce saturation at ~1.0 on raw logits.
    "avh_fusion_shrink_gamma": 0.84,
    # When |p_avh_soft − p_audio| exceeds this, blend in equal-weight mean (modalities disagree).
    "fusion_high_tension_threshold": 0.26,
    # Weight on (p_audio + p_avh_soft)/2 when high tension (rest = standard reliability blend).
    "fusion_mean_blend_weight": 0.62,
    # Widen verdict uncertain band when tension is high (reduces "everything Likely FAKE").
    "fusion_tension_tau_boost_scale": 0.26,
    "fusion_tension_tau_boost_cap": 0.09,
    # NOMA recalibration is applied in logit-space to p(fake).
    # p_cal = sigmoid((logit(p_raw) + noma_bias) / noma_temperature)
    "noma_temperature": 1.0,
    "noma_bias": 0.0,
}


def _load_calibration_artifacts() -> dict:
    """
    Load JSON calibration overrides. If env `CALIBRATION_ARTIFACTS_PATH` is set and the file
    exists, it wins (for tuned eval runs without overwriting repo root `calibration_artifacts.json`).
    """
    override = os.environ.get("CALIBRATION_ARTIFACTS_PATH", "").strip()
    if override and os.path.isfile(override):
        try:
            with open(override, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass

    path_candidates = [
        os.path.join(PROJECT_ROOT, "calibration_artifacts.json"),
        os.path.join(PROJECT_ROOT, "artifacts", "calibration_artifacts.json"),
    ]
    for p in path_candidates:
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
    return {}


def get_uncertainty_margins() -> tuple[float, float]:
    art = _load_calibration_artifacts()
    return (
        float(art.get("avh_uncertainty_margin", DEFAULTS["avh_uncertainty_margin"])),
        float(art.get("noma_uncertainty_margin", DEFAULTS["noma_uncertainty_margin"])),
    )


def shrink_probability_toward_half(p: float, gamma: float) -> float:
    """Reduce overconfidence for fusion: gamma in (0, 1], 1 = no change."""
    return 0.5 + (float(p) - 0.5) * float(gamma)


def get_fusion_hyperparameters() -> dict[str, float]:
    """Defaults + optional calibration_artifacts.json overrides for reliability fusion."""
    art = _load_calibration_artifacts()

    def _f(key: str) -> float:
        return float(art.get(key, DEFAULTS[key]))

    return {
        "avh_fusion_shrink_gamma": _f("avh_fusion_shrink_gamma"),
        "fusion_high_tension_threshold": _f("fusion_high_tension_threshold"),
        "fusion_mean_blend_weight": _f("fusion_mean_blend_weight"),
        "fusion_tension_tau_boost_scale": _f("fusion_tension_tau_boost_scale"),
        "fusion_tension_tau_boost_cap": _f("fusion_tension_tau_boost_cap"),
    }


def avh_score_to_p_fake(score: float) -> float:
    """Map **supervised** AVH-Align fusion logits/score to calibrated p(fake)."""
    art = _load_calibration_artifacts()
    T = float(art.get("avh_temperature", DEFAULTS["avh_temperature"]))
    b = float(art.get("avh_bias", DEFAULTS["avh_bias"]))
    if T == 0:
        T = 1.0
    x = (float(score) - b) / T
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def avh_unsupervised_score_to_p_fake(score: float) -> float:
    """
    Map **unsupervised** lip–audio mismatch score (AVH/test_video_unsupervised.py) to p(fake).

    That score is a weighted sum of lag / off-zero alignment cues, not the supervised Fusion
    logit — applying `avh_score_to_p_fake` to it wrongly treats small positive scores like
    strong fake evidence and biases everything toward p(fake)≈1.
    """
    art = _load_calibration_artifacts()
    c = float(art.get("avh_unsup_center", DEFAULTS["avh_unsup_center"]))
    s = float(art.get("avh_unsup_scale", DEFAULTS["avh_unsup_scale"]))
    if s == 0:
        s = 0.85
    x = (float(score) - c) / s
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def avh_score_to_calibrated_p_fake(score: float, *, use_unsup_avh: bool) -> float:
    """Dispatch: supervised fusion vs unsupervised heuristic."""
    if use_unsup_avh:
        return avh_unsupervised_score_to_p_fake(score)
    return avh_score_to_p_fake(score)


def noma_p_fake_to_calibrated(p_fake):
    """
    Recalibrate NOMA p(fake) values using a logit-space temperature scaling model.

    If calibration artifacts are missing, this becomes an identity transform.
    """
    import numpy as np

    art = _load_calibration_artifacts()
    T = float(art.get("noma_temperature", DEFAULTS["noma_temperature"]))
    b = float(art.get("noma_bias", DEFAULTS["noma_bias"]))
    if T == 0:
        T = 1.0

    p = np.asarray(p_fake, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    x = (logit + b) / T
    # stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    if np.isscalar(p_fake):
        return float(out.item())
    return out

