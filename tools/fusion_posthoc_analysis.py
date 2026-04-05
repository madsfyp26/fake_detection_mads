#!/usr/bin/env python3
"""
Post-hoc fusion analysis: GT from filename rule (evaluation ONLY), feature importance, grid tuning.

Inference must never use filenames — this script only labels rows AFTER reading model outputs from CSV.

Ground truth rule (analysis): video_name.startswith("WhatsApp") -> REAL (0), else FAKE (1).
  Use --gt_case_insensitive to treat "whatsapp..." as REAL (optional).

Outputs under --out_dir:
  - feature_importance.json
  - error_analysis_baseline.json
  - grid_search_best.json
  - before_after_metrics.json
  - safety_check.json (no filename in fusion inputs; procedural check)
  - per_video_with_learned.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from explainability.learned_reliability_fusion import compute_learned_reliability_fusion
from explainability.adaptive_fusion_tune import confidence_from_probability


def _gt_label(video_name: str, *, case_insensitive: bool) -> int:
    """0 = real, 1 = fake."""
    name = str(video_name).strip()
    if case_insensitive:
        return 0 if name.lower().startswith("whatsapp") else 1
    return 0 if name.startswith("WhatsApp") else 1


def _importance(x: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> float:
    """|mean_fake - mean_real| / (std_fake + std_real + eps). y in {0,1}."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    r = x[y == 0]
    f = x[y == 1]
    if len(r) < 1 or len(f) < 1:
        return float("nan")
    return float(abs(f.mean() - r.mean()) / (f.std(ddof=0) + r.std(ddof=0) + eps))


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_fake": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_fake": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "n": int(len(y_true)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc learned fusion analysis (GT from filename rule).")
    ap.add_argument("--csv", type=str, required=True, help="evaluate_video_folder results.csv")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--gt_case_insensitive",
        action="store_true",
        help="REAL if name starts with 'whatsapp' (any case). Default: strict 'WhatsApp' prefix.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    req = {"video_name", "p_avh_cal", "p_audio_mean", "p_fused", "model_disagreement_abs", "noma_p_fake_std"}
    missing = req - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    ok = df[df.get("avh_ok") == True].copy()  # noqa: E712
    ok = ok[ok["p_avh_cal"].notna() & ok["p_audio_mean"].notna()].copy()
    ok["noma_p_fake_std"] = pd.to_numeric(ok["noma_p_fake_std"], errors="coerce").fillna(0.0)
    ok["y_true"] = ok["video_name"].map(lambda n: _gt_label(str(n), case_insensitive=args.gt_case_insensitive))
    if ok["y_true"].nunique() < 2:
        raise SystemExit(
            "Ground truth has a single class only; cannot compute F1. Check --gt_case_insensitive or labels."
        )

    p_v = ok["p_avh_cal"].astype(float).values
    p_a = ok["p_audio_mean"].astype(float).values
    lip = ok["model_disagreement_abs"].astype(float).values
    ti = ok["noma_p_fake_std"].astype(float).values
    conf_v = np.array([confidence_from_probability(float(x)) for x in p_v])
    conf_a = np.array([confidence_from_probability(float(x)) for x in p_a])
    y = ok["y_true"].astype(int).values

    feat_rows = {
        "lip_sync_error_score": _importance(lip, y),
        "temporal_inconsistency_score_noma_std": _importance(ti, y),
        "p_v": _importance(p_v, y),
        "p_a": _importance(p_a, y),
        "confidence_v": _importance(conf_v, y),
        "confidence_a": _importance(conf_a, y),
    }
    with open(os.path.join(args.out_dir, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "formula": "|mean_fake - mean_real| / (std_fake + std_real + eps)",
                "features": feat_rows,
                "ranked": sorted(feat_rows.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1), kv[0])),
            },
            f,
            indent=2,
        )

    # Baseline binary: threshold on p_fused (search best thr on grid for fair compare)
    pf_base = ok["p_fused"].astype(float).values
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.35, 0.72, 38):
        pred = (pf_base >= thr).astype(int)
        f1 = f1_score(y, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    pred_base = (pf_base >= best_thr).astype(int)
    fp = ok.loc[(y == 0) & (pred_base == 1), "video_name"].tolist()
    fn = ok.loc[(y == 1) & (pred_base == 0), "video_name"].tolist()

    baseline_metrics = _metrics_binary(y, pred_base)
    with open(os.path.join(args.out_dir, "error_analysis_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "gt_rule": "startswith('WhatsApp') -> REAL (0) else FAKE (1)"
                if not args.gt_case_insensitive
                else "startswith('whatsapp') case-insensitive -> REAL (0) else FAKE (1)",
                "baseline_best_threshold_on_p_fused": best_thr,
                "metrics": baseline_metrics,
                "false_positives_predicted_fake": fp,
                "false_negatives_missed_fake": fn,
            },
            f,
            indent=2,
        )

    # Grid search learned fusion
    best: dict[str, Any] = {}
    best_f1_grid = -1.0
    for alpha in np.linspace(-0.5, 2.0, 11):
        for beta in np.linspace(-0.5, 2.0, 11):
            for tau in np.linspace(0.06, 0.35, 12):
                out_pf = np.empty(len(ok))
                for i in range(len(ok)):
                    r = compute_learned_reliability_fusion(
                        float(p_v[i]),
                        float(p_a[i]),
                        float(lip[i]),
                        float(ti[i]),
                        alpha=float(alpha),
                        beta=float(beta),
                        tau=float(tau),
                        epsilon=1e-6,
                    )
                    out_pf[i] = r["p_fused"]
                for thr in np.linspace(0.35, 0.72, 20):
                    pred = (out_pf >= thr).astype(int)
                    f1 = f1_score(y, pred, pos_label=1, zero_division=0)
                    if f1 > best_f1_grid:
                        best_f1_grid = f1
                        best = {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "tau": float(tau),
                            "decision_threshold": float(thr),
                            "f1_fake": float(f1),
                            "metrics": _metrics_binary(y, pred),
                        }

    with open(os.path.join(args.out_dir, "grid_search_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # Apply best params to full table
    alpha = best.get("alpha", 0.0)
    beta = best.get("beta", 0.0)
    tau = best.get("tau", 0.12)
    thr_learned = best.get("decision_threshold", 0.5)
    learned_pf = []
    for i in range(len(ok)):
        r = compute_learned_reliability_fusion(
            float(p_v[i]),
            float(p_a[i]),
            float(lip[i]),
            float(ti[i]),
            alpha=alpha,
            beta=beta,
            tau=tau,
            epsilon=1e-6,
        )
        learned_pf.append(r["p_fused"])
    ok = ok.copy()
    ok["p_fused_learned"] = learned_pf
    ok["y_pred_learned"] = (ok["p_fused_learned"] >= thr_learned).astype(int)
    ok["y_pred_baseline_thr"] = pred_base

    pred_learned = ok["y_pred_learned"].astype(int).values
    fp_l = ok.loc[(y == 0) & (pred_learned == 1), "video_name"].tolist()
    fn_l = ok.loc[(y == 1) & (pred_learned == 0), "video_name"].tolist()

    after_metrics = _metrics_binary(y, pred_learned)
    with open(os.path.join(args.out_dir, "before_after_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": {**baseline_metrics, "best_threshold": best_thr},
                "after_learned_fusion": after_metrics,
                "learned_hyperparameters": {
                    "learned_fusion_alpha": alpha,
                    "learned_fusion_beta": beta,
                    "learned_fusion_tau": tau,
                    "learned_fusion_epsilon": 1e-6,
                    "learned_fusion_decision_threshold": thr_learned,
                },
                "fp_reduction": len(fp) - len(fp_l),
                "fn_reduction": len(fn) - len(fn_l),
            },
            f,
            indent=2,
        )

    params_out = os.path.join(args.out_dir, "learned_fusion_params.json")
    with open(params_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "learned_fusion_alpha": alpha,
                "learned_fusion_beta": beta,
                "learned_fusion_tau": tau,
                "learned_fusion_epsilon": 1e-6,
                "learned_fusion_decision_threshold": thr_learned,
            },
            f,
            indent=2,
        )

    ok.to_csv(os.path.join(args.out_dir, "per_video_with_learned.csv"), index=False)

    with open(os.path.join(args.out_dir, "safety_check.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "inference_must_not_use_filename_as_feature": True,
                "this_script_uses_filename_only_for_gt_column_after_inference": True,
                "learned_fusion_inputs_are_only": [
                    "p_avh_cal",
                    "p_audio_mean",
                    "lip_sync_error",
                    "temporal_inconsistency",
                ],
                "correlation_filename_vs_p_fused_learned": float(
                    np.corrcoef(
                        ok["video_name"].str.len().values,
                        ok["p_fused_learned"].values,
                    )[0, 1]
                )
                if len(ok) > 2
                else None,
                "note": "Low correlation with name length does not prove absence of leakage; "
                "audit AVH/NOMA inputs to ensure only pixels/audio are used.",
            },
            f,
            indent=2,
        )

    print(json.dumps({"wrote": args.out_dir, "best_f1": best.get("f1_fake"), "params": params_out}, indent=2))


if __name__ == "__main__":
    main()
