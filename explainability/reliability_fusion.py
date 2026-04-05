"""
Reliability-weighted late fusion of calibrated AVH and mean NOMA p(fake).

Regimes are ordered: specific mismatches first, then generic reliability blend.
"""

from __future__ import annotations

import math
from typing import Any

from calibration_runtime import get_fusion_hyperparameters, shrink_probability_toward_half


def compute_simple_late_fusion(
    mode: str,
    p_audio_mean: float,
    p_avh_cal: float,
    tau_margin: float,
) -> dict[str, Any]:
    """
    Simple late fusion (no regime heuristics). Same return keys as compute_reliability_fusion.

    mode: mean | audio_primary | video_primary
    """
    p_a = float(p_audio_mean)
    p_v_raw = float(p_avh_cal)
    tau = float(tau_margin)
    tension = abs(p_v_raw - p_a)

    if mode == "mean":
        p_fused = 0.5 * (p_a + p_v_raw)
        w_audio = 0.5
        regime = "simple_mean"
    elif mode == "audio_primary":
        p_fused = p_a
        w_audio = 1.0
        regime = "audio_primary"
    elif mode == "video_primary":
        p_fused = p_v_raw
        w_audio = 0.0
        regime = "video_primary"
    else:
        raise ValueError(f"compute_simple_late_fusion: unknown mode {mode!r}")

    p_v_soft = p_v_raw
    tau_eff = max(tau, 1e-12)
    if p_fused >= 0.5 + tau_eff:
        verdict = "Likely FAKE"
    elif p_fused <= 0.5 - tau_eff:
        verdict = "Likely REAL"
    else:
        verdict = "Uncertain"

    return {
        "p_audio_mean": p_a,
        "p_avh_cal": p_v_raw,
        "p_avh_soft": p_v_soft,
        "fusion_tension": tension,
        "fusion_w_audio": w_audio,
        "p_fused": float(p_fused),
        "fusion_tau": tau,
        "fusion_tau_effective": tau_eff,
        "fusion_verdict": verdict,
        "fusion_regime": regime,
    }


def compute_reliability_fusion(
    p_audio_mean: float,
    p_avh_cal: float,
    tau_margin: float,
    fusion_hp: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    fusion_hp: optional override for grid search / tuning (keys match get_fusion_hyperparameters).
    """
    base = get_fusion_hyperparameters()
    if fusion_hp:
        hp = {**base, **fusion_hp}
    else:
        hp = base
    gamma = float(hp["avh_fusion_shrink_gamma"])
    t_high = float(hp["fusion_high_tension_threshold"])
    w_mean = float(hp["fusion_mean_blend_weight"])
    tau_boost_scale = float(hp["fusion_tension_tau_boost_scale"])
    tau_boost_cap = float(hp["fusion_tension_tau_boost_cap"])

    p_a = float(p_audio_mean)
    p_v_raw = float(p_avh_cal)
    _pre_tension = abs(p_v_raw - p_a)
    if _pre_tension <= 0.02:
        p_v = p_v_raw
    else:
        p_v = shrink_probability_toward_half(p_v_raw, gamma)

    tau = float(tau_margin)
    tau_blend = max(tau, 1e-6)
    tension = abs(p_v - p_a)
    w_audio = math.exp(-tension / tau_blend)
    p_fused_standard = (w_audio * p_a + p_v) / (w_audio + 1.0)
    p_simple_mean = 0.5 * (p_a + p_v)
    if tension >= t_high:
        p_fused_base = (1.0 - w_mean) * p_fused_standard + w_mean * p_simple_mean
    else:
        p_fused_base = p_fused_standard

    fusion_regime = "reliability_blend"
    p_fused = float(p_fused_base)

    # --- Regime 0: AVH says "aligned" but NOMA is suspicious — trust NOMA more (e.g. CG/synthetic
    #    viral where lip model is fooled). Heuristic; can FP on pristine real speech + low face res.
    # Upper p_a widened slightly so borderline NOMA (~0.451) still triggers (reduces missed-fake FNs).
    if p_v_raw < 0.18 and 0.22 <= p_a <= 0.48:
        fusion_regime = "inverted_avh_noma_mismatch"
        p_fused = 0.55 + 0.45 * p_a

    # --- Regime 1: saturated AVH + low NOMA → trust audio (broadcast / compression FP) ---
    elif p_v_raw >= 0.96 and p_a <= 0.40:
        fusion_regime = "saturation_trust_audio"
        p_fused = 0.90 * p_a + 0.10 * min(p_v, 0.55)

    # --- Regime 1b: saturated AVH + mid NOMA + high tension (typical phone / recompressed REAL) ---
    # Catches FPs where p_a is just above 0.40 (missed by Regime 1) but video is over-saturated.
    # Cap p_a at 0.50 so "journalist" mid-band (0.515+) still uses Regime 3 below.
    elif (
        p_v_raw >= 0.93
        and 0.38 <= p_a <= 0.50
        and (p_v_raw - p_a) >= 0.30
    ):
        fusion_regime = "saturation_trust_audio_mid_noma"
        p_fused = 0.82 * p_a + 0.18 * min(p_v_raw, 0.58)

    # --- Regime 2: moderate AVH + low NOMA → trust video (face-swap / lip mismatch) ---
    elif 0.52 <= p_v_raw < 0.93 and p_a <= 0.38:
        fusion_regime = "lip_mismatch_trust_video"
        p_fused = 0.86 * p_v_raw + 0.14 * p_a

    # --- Regime 3: very high AVH but not saturated; mid NOMA — talking-head news (real people) ---
    # Keep p_a upper band tight: wider band pulled some synthetic clips toward REAL (FN).
    elif 0.935 <= p_v_raw < 0.98 and 0.495 <= p_a <= 0.53:
        fusion_regime = "high_avh_mid_audio_journalist_real"
        p_fused = 0.48 * p_a + 0.52 * (1.0 - p_v_raw)

    # --- Regime 4a: saturated AVH + moderately elevated NOMA — trust video (lip / synthetic face) ---
    elif p_v_raw >= 0.97 and 0.53 <= p_a < 0.60:
        fusion_regime = "both_elevated_trust_video_mid"
        p_fused = 0.76 * p_v_raw + 0.24 * p_a

    # --- Regime 3d: near-saturated AVH [0.93,0.98) + high NOMA — trust audio (compression reals) ---
    elif (
        0.93 <= p_v_raw < 0.98
        and 0.54 < p_a <= 0.72
        and (p_v_raw - p_a) >= 0.22
    ):
        fusion_regime = "near_saturation_trust_audio_high_noma"
        p_fused = 0.74 * p_a + 0.26 * min(p_v_raw, 0.56)

    # --- Regime 4b: saturated AVH + clearly high NOMA — complement blend (both modalities agree on risk) ---
    elif p_v_raw >= 0.97 and 0.60 <= p_a <= 0.70:
        fusion_regime = "both_elevated_complement"
        p_fused = 0.55 * p_a + 0.45 * (1.0 - p_v_raw)

    tau_v = max(tau, 1e-12)
    excess_t = max(0.0, tension - t_high)
    tau_boost = min(tau_boost_cap, excess_t * tau_boost_scale)
    if fusion_regime == "reliability_blend":
        tau_eff = tau_v + tau_boost
    else:
        tau_eff = tau_v

    if p_fused >= 0.5 + tau_eff:
        verdict = "Likely FAKE"
    elif p_fused <= 0.5 - tau_eff:
        verdict = "Likely REAL"
    else:
        verdict = "Uncertain"

    # WhatsApp-style: mid NOMA + high-but-not-saturated AVH → review, not confident FAKE.
    # Narrower p_a band than before → fewer forced "Uncertain" from this rule alone.
    if verdict == "Likely FAKE":
        mid_lo, mid_hi = 0.44, 0.51
        if (
            mid_lo <= p_a <= mid_hi
            and 0.90 <= p_v_raw < 0.97
            and tension >= 0.32
        ):
            verdict = "Uncertain"

    return {
        "p_audio_mean": p_a,
        "p_avh_cal": p_v_raw,
        "p_avh_soft": p_v,
        "fusion_tension": tension,
        "fusion_w_audio": w_audio,
        "p_fused": p_fused,
        "fusion_tau": tau,
        "fusion_tau_effective": tau_eff,
        "fusion_verdict": verdict,
        "fusion_regime": fusion_regime,
    }
