"""
Optional SyncNet-style lip-sync score (placeholder integration).

Install a SyncNet implementation + weights and set SYNCNET_WEIGHTS_PATH, or implement
`run_syncnet_score` to return a calibrated sync confidence in [0, 1].
"""

from __future__ import annotations

import os
from typing import Any


def run_syncnet_score(
    video_path: str,
    *,
    audio_path: str | None = None,
    roi_path: str | None = None,
) -> dict[str, Any]:
    """
    Returns dict with keys: ok, sync_score (0-1 or None), error (optional).

    Default: not configured — ok False with guidance message.
    """
    weights = os.environ.get("SYNCNET_WEIGHTS_PATH", "").strip()
    if not weights or not os.path.isfile(weights):
        return {
            "ok": False,
            "sync_score": None,
            "error": "SYNCNET_WEIGHTS_PATH not set or missing; add SyncNet weights and inference code.",
        }
    return {
        "ok": False,
        "sync_score": None,
        "error": "SyncNet inference not wired; set SYNCNET_WEIGHTS_PATH and implement run_syncnet_score.",
    }
