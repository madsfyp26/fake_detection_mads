"""
Dataset-driven adaptive fusion: tension-weighted blend with learnable tau / threshold.

Maps to multimodal setup where p_v ≈ AVH calibrated p(fake) and p_a ≈ NOMA mean p(fake).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def confidence_from_probability(p: float) -> float:
    """Decisiveness: high when p is near 0 or 1; low near 0.5 (uncertain modality output)."""
    p = float(np.clip(p, 0.0, 1.0))
    return float(max(1e-3, 2.0 * abs(p - 0.5)))


def lip_sync_error_score(p_v: float, p_a: float) -> float:
    """Normalized cross-modality disagreement in [0, 1]."""
    return float(abs(float(p_v) - float(p_a)))


def adaptive_fusion_p(
    p_v: float,
    p_a: float,
    conf_v: float,
    conf_a: float,
    *,
    tau: float,
    epsilon: float = 1e-6,
    tension_boost_beta: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    p_fused = (w_v * p_v + w_a * p_a) / (w_v + w_a + eps)
    w_* = conf_* * exp(-tension/tau); optional boost on w_v when lip-sync error is high.
    Returns (p_fused, w_v, w_a, tension).
    """
    tension = abs(float(p_v) - float(p_a))
    tau = max(float(tau), 1e-9)
    w_v = float(conf_v) * math.exp(-tension / tau)
    w_a = float(conf_a) * math.exp(-tension / tau)
    if tension_boost_beta > 0:
        w_v *= 1.0 + float(tension_boost_beta) * tension
    den = w_v + w_a + float(epsilon)
    p_fused = (w_v * float(p_v) + w_a * float(p_a)) / den
    return float(np.clip(p_fused, 0.0, 1.0)), w_v, w_a, tension


def predict_fake_from_p_fused(p_fused: float, threshold: float) -> int:
    """1 = fake, 0 = real."""
    return 1 if float(p_fused) >= float(threshold) else 0


def grid_search_fusion_and_threshold(
    p_v: np.ndarray,
    p_a: np.ndarray,
    y_true: np.ndarray,
    *,
    tau_grid: np.ndarray | None = None,
    threshold_grid: np.ndarray | None = None,
    epsilon: float = 1e-6,
    tension_boost_betas: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    y_true: 0 = real, 1 = fake (binary labels from your evaluation set, e.g. --labels_csv).
    Maximizes F1 for the fake class.
    """
    from sklearn.metrics import f1_score

    p_v = np.asarray(p_v, dtype=float)
    p_a = np.asarray(p_a, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"error": "need_at_least_two_samples_and_two_classes", "n": int(n)}

    if tau_grid is None:
        tau_grid = np.linspace(0.04, 0.45, 28)
    if threshold_grid is None:
        threshold_grid = np.linspace(0.35, 0.75, 33)
    if tension_boost_betas is None:
        tension_boost_betas = np.array([0.0, 0.5, 1.0])

    conf_v = np.array([confidence_from_probability(float(x)) for x in p_v])
    conf_a = np.array([confidence_from_probability(float(x)) for x in p_a])

    best_f1 = -1.0
    best: dict[str, Any] = {}
    for beta in tension_boost_betas:
        for tau in tau_grid:
            for thr in threshold_grid:
                preds = np.zeros(n, dtype=int)
                for i in range(n):
                    pf, _, _, _ = adaptive_fusion_p(
                        p_v[i],
                        p_a[i],
                        conf_v[i],
                        conf_a[i],
                        tau=float(tau),
                        epsilon=epsilon,
                        tension_boost_beta=float(beta),
                    )
                    preds[i] = predict_fake_from_p_fused(pf, thr)
                f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best = {
                        "f1_fake": f1,
                        "tau": float(tau),
                        "threshold": float(thr),
                        "tension_boost_beta": float(beta),
                        "epsilon": float(epsilon),
                    }

    # Refine accuracy / precision / recall at best point
    if not best:
        return {"error": "grid_empty"}

    beta = best["tension_boost_beta"]
    tau = best["tau"]
    thr = best["threshold"]
    preds = np.zeros(n, dtype=int)
    p_fused_arr = np.zeros(n, dtype=float)
    for i in range(n):
        pf, _, _, _ = adaptive_fusion_p(
            p_v[i],
            p_a[i],
            conf_v[i],
            conf_a[i],
            tau=tau,
            epsilon=epsilon,
            tension_boost_beta=beta,
        )
        p_fused_arr[i] = pf
        preds[i] = predict_fake_from_p_fused(pf, thr)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    acc = float(accuracy_score(y_true, preds))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", pos_label=1, zero_division=0
    )
    best["accuracy"] = acc
    best["precision_fake"] = float(pr)
    best["recall_fake"] = float(rc)
    best["f1_fake_sklearn"] = float(f1)
    best["p_fused_adaptive"] = p_fused_arr.tolist()
    best["y_pred"] = preds.tolist()
    return best


def best_threshold_for_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold_grid: np.ndarray | None = None,
    pos_label: int = 1,
) -> dict[str, Any]:
    """Find threshold on `scores` that maximizes F1 for fake (pos_label=1)."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if threshold_grid is None:
        threshold_grid = np.linspace(0.25, 0.85, 49)
    best_f1 = -1.0
    best_thr = 0.5
    best_pred = None
    for thr in threshold_grid:
        pred = (scores >= thr).astype(int)
        f1 = float(f1_score(y_true, pred, pos_label=pos_label, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_pred = pred
    out: dict[str, Any] = {"threshold": best_thr, "f1_fake": best_f1}
    if best_pred is not None:
        out.update(metrics_binary(y_true, best_pred, pos_label=pos_label))
    return out


def metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, *, pos_label: int = 1) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(accuracy_score(y_true, y_pred))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
    }
