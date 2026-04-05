"""
Lightweight proxies for dataset analysis (not production detectors).

- Optical-flow stability on uniformly sampled frames → temporal inconsistency proxy [0, 1].
- Librosa stats on decoded audio → audio feature vector (normalized later in batch).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def optical_flow_temporal_proxy(
    video_path: str,
    *,
    max_pairs: int = 10,
    target_size: tuple[int, int] = (256, 144),
) -> dict[str, Any]:
    """
    Mean Farneback flow magnitude between consecutive sampled frames; map to [0,1] via tanh scale.

    Returns dict with keys: ok, error, temporal_flow_raw, temporal_inconsistency_score (0-1).
    """
    try:
        import cv2
    except ImportError as e:
        return {"ok": False, "error": f"opencv: {e}", "temporal_inconsistency_score": None}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"ok": False, "error": "VideoCapture failed", "temporal_inconsistency_score": None}

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    if n_frames < 3:
        cap.release()
        return {"ok": False, "error": "too_few_frames", "temporal_inconsistency_score": None}

    # Uniform indices: skip borders
    n_samples = min(max_pairs + 2, n_frames)
    idx = np.linspace(1, n_frames - 2, num=n_samples, dtype=int)
    frames_gray: list[np.ndarray] = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        fr = cv2.resize(fr, target_size)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        frames_gray.append(g)
    cap.release()

    if len(frames_gray) < 2:
        return {"ok": False, "error": "decode_fail", "temporal_inconsistency_score": None}

    mags: list[float] = []
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
        mags.append(float(mag))

    raw = float(np.mean(mags)) if mags else 0.0
    # Map to [0,1] — typical raw magnitudes are small; scale with tanh
    score = float(np.tanh(raw * 8.0))
    return {
        "ok": True,
        "error": None,
        "temporal_flow_raw": raw,
        "temporal_inconsistency_score": score,
    }


def librosa_audio_proxies(
    audio_path: str | None,
    *,
    sr: int = 16000,
    max_seconds: float = 120.0,
) -> dict[str, Any]:
    """MFCC / ZCR / spectral contrast aggregates; returns None fields if missing or decode fails."""
    if not audio_path or not os.path.isfile(audio_path):
        return {
            "ok": False,
            "error": "no_audio_path",
            "mfcc_energy_mean": None,
            "zcr_mean": None,
            "spectral_contrast_mean": None,
        }
    try:
        import librosa
    except ImportError as e:
        return {"ok": False, "error": str(e), "mfcc_energy_mean": None, "zcr_mean": None, "spectral_contrast_mean": None}

    try:
        y, _sr = librosa.load(audio_path, sr=sr, mono=True, duration=max_seconds)
    except Exception as e:
        return {"ok": False, "error": str(e), "mfcc_energy_mean": None, "zcr_mean": None, "spectral_contrast_mean": None}

    if y is None or len(y) < 512:
        return {"ok": False, "error": "too_short", "mfcc_energy_mean": None, "zcr_mean": None, "spectral_contrast_mean": None}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    zcr = librosa.feature.zero_crossing_rate(y)
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)

    return {
        "ok": True,
        "error": None,
        "mfcc_energy_mean": float(np.mean(np.abs(mfcc))),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_contrast_mean": float(np.mean(sc)),
    }
