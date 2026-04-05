"""Smoke test for fusion_posthoc_analysis on synthetic CSV."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def test_fusion_posthoc_runs():
    rows = []
    for i in range(12):
        pv, pa = 0.4 + 0.01 * i, 0.5 - 0.02 * i
        rows.append(
            {
                "video_name": f"WhatsApp_x_{i}.mp4" if i % 2 == 0 else f"other_{i}.mp4",
                "avh_ok": True,
                "p_avh_cal": pv,
                "p_audio_mean": pa,
                "p_fused": 0.5 * (pv + pa),
                "model_disagreement_abs": abs(pv - pa),
                "noma_p_fake_std": 0.05 + 0.01 * (i % 3),
            }
        )
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "in.csv")
        out_dir = os.path.join(d, "out")
        df.to_csv(csv_path, index=False)
        subprocess.check_call(
            [
                sys.executable,
                str(_ROOT / "tools" / "fusion_posthoc_analysis.py"),
                "--csv",
                csv_path,
                "--out_dir",
                out_dir,
                "--gt_case_insensitive",
            ],
            cwd=str(_ROOT),
            env={**os.environ, "PYTHONPATH": str(_ROOT)},
        )
        with open(os.path.join(out_dir, "before_after_metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert "baseline" in m and "after_learned_fusion" in m
