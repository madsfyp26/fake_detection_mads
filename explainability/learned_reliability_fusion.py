"""
Learned reliability-weighted fusion using only calibrated p_v, p_a and derived signals.

Uses lip-sync disagreement and temporal NOMA inconsistency — never filenames or paths.

Formula (tunable alpha, beta, tau, epsilon):
  reliability_v = confidence_v + alpha * lip_sync_error + beta * temporal_inconsistency
  reliability_a = confidence_a
  tension = abs(p_v - p_a)
  w_v = reliability_v * exp(-tension / tau)
  w_a = reliability_a * exp(-tension / tau)
  p_fused = (w_v * p_v + w_a * p_a) / (w_v + w_a + epsilon)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from explainability.adaptive_fusion_tune import confidence_from_probability

# Defaults when not in calibration JSON / artifacts file
DEFAULT_LEARNED_FUSION: dict[str, float] = {
    "learned_fusion_alpha": 0.0,
    "learned_fusion_beta": 0.0,
    "learned_fusion_tau": 0.12,
    "learned_fusion_epsilon": 1e-6,
    "learned_fusion_decision_threshold": 0.5,
}


def get_learned_fusion_hyperparameters() -> dict[str, float]:
    """Merge defaults, calibration_artifacts.json keys, and LEARNED_FUSION_PARAMS_PATH."""
    from calibration_runtime import _load_calibration_artifacts

    out = dict(DEFAULT_LEARNED_FUSION)
    art = _load_calibration_artifacts()
    for k in DEFAULT_LEARNED_FUSION:
        if k in art:
            try:
                out[k] = float(art[k])
            except (TypeError, ValueError):
                pass
    env_path = os.environ.get("LEARNED_FUSION_PARAMS_PATH", "").strip()
    if env_path:
        loaded = load_learned_fusion_params_from_json(env_path)
        out.update({k: loaded[k] for k in DEFAULT_LEARNED_FUSION if k in loaded})
    return out


def load_learned_fusion_params_from_json(path: str | None) -> dict[str, float]:
    """Load alpha, beta, tau, epsilon, decision_threshold from JSON (merged with defaults)."""
    out = dict(DEFAULT_LEARNED_FUSION)
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for k in DEFAULT_LEARNED_FUSION:
            if k in data:
                out[k] = float(data[k])
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        pass
    return out


def compute_learned_reliability_fusion(
    p_v: float,
    p_a: float,
    lip_sync_error: float,
    temporal_inconsistency: float,
    *,
    alpha: float,
    beta: float,
    tau: float,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """
    p_v, p_a: calibrated visual / audio p(fake) in [0,1].
    lip_sync_error: |p_v - p_a| or related (>= 0).
    temporal_inconsistency: e.g. std of per-block NOMA p(fake) (>= 0).
    """
    p_v = float(np.clip(p_v, 0.0, 1.0))
    p_a = float(np.clip(p_a, 0.0, 1.0))
    conf_v = confidence_from_probability(p_v)
    conf_a = confidence_from_probability(p_a)
    lip = max(0.0, float(lip_sync_error))
    ti = max(0.0, float(temporal_inconsistency))

    rel_v = float(conf_v) + float(alpha) * lip + float(beta) * ti
    rel_a = float(conf_a)
    rel_v = max(rel_v, 1e-9)
    rel_a = max(rel_a, 1e-9)

    tension = abs(p_v - p_a)
    tau = max(float(tau), 1e-9)
    w_v = rel_v * math.exp(-tension / tau)
    w_a = rel_a * math.exp(-tension / tau)
    eps = max(float(epsilon), 1e-12)
    den = w_v + w_a + eps
    p_fused = (w_v * p_v + w_a * p_a) / den
    p_fused = float(np.clip(p_fused, 0.0, 1.0))

    return {
        "p_audio_mean": p_a,
        "p_avh_cal": p_v,
        "p_avh_soft": p_v,
        "fusion_tension": tension,
        "fusion_w_audio": float(w_a / (w_v + w_a + eps)),
        "fusion_w_video": float(w_v / (w_v + w_a + eps)),
        "p_fused": p_fused,
        "fusion_tau": tau,
        "fusion_tau_effective": tau,
        "fusion_verdict": "",  # filled by caller using margins
        "fusion_regime": "learned_reliability",
        "learned_w_v": w_v,
        "learned_w_a": w_a,
        "learned_rel_v": rel_v,
        "learned_rel_a": rel_a,
    }


def apply_verdict_three_way(
    p_fused: float,
    *,
    tau_margin: float,
) -> str:
    """Same band logic as reliability fusion: uncertain between 0.5 ± tau."""
    t = max(float(tau_margin), 1e-12)
    if p_fused >= 0.5 + t:
        return "Likely FAKE"
    if p_fused <= 0.5 - t:
        return "Likely REAL"
    return "Uncertain"


def binary_predict_fake(p_fused: float, threshold: float) -> int:
    """1 = fake, 0 = real (for F1 / accuracy vs GT)."""
    return 1 if float(p_fused) >= float(threshold) else 0
