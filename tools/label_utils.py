"""
Shared helpers for joining per-video binary labels (0=real, 1=fake).

Prefer a hand-verified `labels.csv`; filename-based rules are optional proxies only.
"""

from __future__ import annotations

import os

import pandas as pd


def load_labels_csv(path: str) -> dict[str, int]:
    """Load video_name -> label (0/1). Basenames must match raw_results video_name."""
    if not path or not os.path.isfile(path):
        raise SystemExit(f"Labels file not found: {path}")
    df = pd.read_csv(path)
    if not {"video_name", "label"}.issubset(df.columns):
        raise SystemExit("labels_csv must have columns: video_name, label (0=real, 1=fake)")
    out: dict[str, int] = {}
    for _, r in df.iterrows():
        name = str(r["video_name"]).strip()
        out[name] = int(r["label"])
    return out


def heuristic_label_whatsapp_proxy(video_name: str) -> int:
    """Legacy proxy: whatsapp in name -> REAL (0), else FAKE (1). Not ground truth."""
    return 0 if "whatsapp" in str(video_name).lower() else 1
