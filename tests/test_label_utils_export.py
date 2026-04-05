"""Tests for real-label helpers and calibration export."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.export_calibration_from_eval_csv import export_from_raw_results
from tools.label_utils import heuristic_label_whatsapp_proxy, load_labels_csv


def test_load_labels_csv_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.csv")
        pd.DataFrame(
            {"video_name": ["a.mp4", "b.mp4"], "label": [0, 1]},
        ).to_csv(p, index=False)
        m = load_labels_csv(p)
    assert m["a.mp4"] == 0 and m["b.mp4"] == 1


def test_export_prefers_labels_column():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "raw.csv")
        pd.DataFrame(
            {
                "video_name": ["a.mp4", "b.mp4", "c.mp4", "d.mp4"],
                "avh_ok": [True, True, True, True],
                "avh_score": [0.1, 0.2, 0.3, 0.4],
                "p_audio_mean_raw": [0.5, 0.5, 0.5, 0.5],
                "label": [0, 1, 0, 1],
            }
        ).to_csv(raw, index=False)
        out = os.path.join(d, "out")
        meta = export_from_raw_results(raw, out, test_size=0.5, random_state=0)
        assert meta["n_rows_ok"] == 4
        assert meta["label_origin"] == "raw_results_column"
        with open(os.path.join(out, "export_meta.json"), encoding="utf-8") as f:
            assert json.load(f)["label_origin"] == "raw_results_column"


def test_heuristic_is_opt_in_proxy():
    assert heuristic_label_whatsapp_proxy("WhatsApp Video x.mp4") == 0
    assert heuristic_label_whatsapp_proxy("other.mp4") == 1
