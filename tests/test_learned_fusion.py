"""Learned fusion: numeric behavior and no-filename contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explainability.learned_reliability_fusion import (
    compute_learned_reliability_fusion,
    get_learned_fusion_hyperparameters,
)


def test_learned_fusion_bounds():
    r = compute_learned_reliability_fusion(
        0.7,
        0.4,
        lip_sync_error=0.3,
        temporal_inconsistency=0.1,
        alpha=0.5,
        beta=0.2,
        tau=0.15,
        epsilon=1e-6,
    )
    assert 0.0 <= r["p_fused"] <= 1.0
    assert r["fusion_regime"] == "learned_reliability"


def test_get_learned_params_has_keys():
    hp = get_learned_fusion_hyperparameters()
    assert "learned_fusion_alpha" in hp
    assert "learned_fusion_tau" in hp


def test_fusion_fn_has_no_filename_param():
    import inspect

    sig = inspect.signature(compute_learned_reliability_fusion)
    names = list(sig.parameters)
    assert "video_name" not in names
    assert "filename" not in names
