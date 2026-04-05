"""
Time-aligned audio (NOMA) vs video saliency corroboration — no new fusion model.

Maps Grad-CAM / fused-heatmap saliency onto NOMA block times and flags agreement
(high p(fake) + high saliency) vs conflict (high p(fake), low saliency).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    return (x - lo) / (hi - lo)


def cam_idx_to_saliency_timeseries(cam_idx: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build (t_sec, saliency_normalized) from Grad-CAM index JSON dict.

    Prefers fused heatmap spatial max per frame when `fused_heatmap_path` exists;
    else uses `cam_per_frame` with ROI time mapping.
    """
    if not cam_idx or not isinstance(cam_idx, dict):
        return None

    roi_fps = cam_idx.get("roi_fps")
    fps = float(roi_fps) if roi_fps else 0.0

    fused_path = cam_idx.get("fused_heatmap_path")
    if fused_path and isinstance(fused_path, str):
        try:
            import os

            if os.path.isfile(fused_path):
                fused = np.load(fused_path)
                if fused.ndim == 3:
                    sal = np.max(fused, axis=(1, 2))
                    t = np.arange(len(sal), dtype=float)
                    if fps > 1e-6:
                        t = t / fps
                    return t, _normalize_01(sal)
        except Exception:
            pass

    cam_per = cam_idx.get("cam_per_frame")
    cam_to_roi = cam_idx.get("cam_to_roi_index")
    if not isinstance(cam_per, list) or len(cam_per) == 0:
        return None

    t_list: list[float] = []
    s_list: list[float] = []
    for cam_i, cam_val in enumerate(cam_per):
        roi_i = None
        if isinstance(cam_to_roi, list) and cam_i < len(cam_to_roi):
            roi_i = cam_to_roi[cam_i]
        if roi_i is None:
            roi_i = cam_i
        if fps > 1e-6:
            t_sec = float(roi_i) / fps
        else:
            t_sec = float(roi_i)
        t_list.append(t_sec)
        s_list.append(float(cam_val))

    t = np.asarray(t_list, dtype=float)
    s = _normalize_01(np.asarray(s_list, dtype=float))
    return t, s


def aggregate_saliency_to_noma_bins(
    noma_seconds: np.ndarray,
    t_sal: np.ndarray,
    sal: np.ndarray,
    *,
    block_width: float = 1.0,
) -> np.ndarray:
    """
    For each NOMA block center time, take max saliency of frames with t in [sec, sec + block_width).

    `noma_seconds` are block start times (e.g. 0, 1, 2, ...).
    """
    noma_seconds = np.asarray(noma_seconds, dtype=float)
    t_sal = np.asarray(t_sal, dtype=float)
    sal = np.asarray(sal, dtype=float)
    out = np.zeros(len(noma_seconds), dtype=float)
    for i, sec in enumerate(noma_seconds):
        lo = float(sec)
        hi = lo + float(block_width)
        mask = (t_sal >= lo) & (t_sal < hi)
        if not np.any(mask):
            # nearest frame fallback
            if t_sal.size == 0:
                out[i] = 0.0
            else:
                j = int(np.argmin(np.abs(t_sal - (lo + hi) / 2)))
                out[i] = sal[j] if j < len(sal) else 0.0
        else:
            out[i] = float(np.max(sal[mask]))
    return out


def compute_temporal_corroboration(
    *,
    noma_seconds: np.ndarray,
    p_fake_calibrated: np.ndarray,
    cam_idx: dict[str, Any] | None,
    p_threshold: float = 0.5,
    sal_threshold: float = 0.5,
    block_seconds: float = 1.0,
) -> dict[str, Any]:
    """
    Returns JSON-serializable summary for UI and RunManifest.

    - corroboration: high audio suspicion AND high video saliency (same time bin)
    - conflict: high audio suspicion AND low video saliency
    """
    p = np.asarray(p_fake_calibrated, dtype=float)
    secs = np.asarray(noma_seconds, dtype=float)
    if secs.shape != p.shape:
        raise ValueError("noma_seconds and p_fake must align")

    ts = cam_idx_to_saliency_timeseries(cam_idx)
    if ts is None:
        return {
            "status": "no_video_saliency",
            "corroboration_rate": None,
            "conflict_rate": None,
            "bins": [],
            "p_threshold": p_threshold,
            "sal_threshold": sal_threshold,
        }

    t_sal, sal = ts
    sal_bins = aggregate_saliency_to_noma_bins(secs, t_sal, sal, block_width=block_seconds)
    sal_bins_norm = _normalize_01(sal_bins)

    high_p = p >= p_threshold
    high_s = sal_bins_norm >= sal_threshold
    low_s = sal_bins_norm < sal_threshold

    corroborate = high_p & high_s
    conflict = high_p & low_s

    n = max(len(p), 1)
    bins: list[dict[str, Any]] = []
    for i in range(len(p)):
        bins.append(
            {
                "second": float(secs[i]) if i < len(secs) else float(i),
                "p_fake": float(p[i]),
                "saliency": float(sal_bins_norm[i]) if i < len(sal_bins_norm) else 0.0,
                "corroboration": bool(corroborate[i]) if i < len(corroborate) else False,
                "conflict": bool(conflict[i]) if i < len(conflict) else False,
            }
        )

    return {
        "status": "ok",
        "corroboration_rate": float(np.sum(corroborate) / n),
        "conflict_rate": float(np.sum(conflict) / n),
        "high_p_rate": float(np.sum(high_p) / n),
        "bins": bins,
        "p_threshold": p_threshold,
        "sal_threshold": sal_threshold,
    }


def compute_tension_index(p_fake_avh_cal: float, noma_p_fake_mean: float) -> float:
    """Absolute disagreement between global AVH and mean NOMA calibrated p(fake)."""
    return abs(float(p_fake_avh_cal) - float(noma_p_fake_mean))
