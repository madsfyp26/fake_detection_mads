#!/usr/bin/env python3
"""
Run Combined (AVH -> NOMA + reliability fusion) over a manifest and report metrics.

Expects a CSV with columns: video_path,label (label: 0=real, 1=fake).

Example:
  PYTHONPATH=. python tools/evaluate_fakeavceleb.py \\
    --manifest_csv fakeavceleb_manifest.csv \\
    --python_exe /path/to/avh/env/bin/python \\
    --max_videos 600 \\
    --out_dir eval_runs/fakeavceleb_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labels[mask]))
        ece += (hi - lo) * abs(acc - conf)
    return float(ece)


def _metrics_block(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if len(y) < 2 or len(np.unique(y)) < 2:
        out["note"] = "insufficient_samples_or_single_class"
        return out
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    out["roc_auc"] = float(roc_auc_score(y, p))
    out["average_precision"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["ece"] = float(_ece(p, y))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Combined pipeline on a FakeAVCeleb manifest.")
    ap.add_argument("--manifest_csv", type=str, required=True)
    ap.add_argument("--max_videos", type=int, default=600)
    ap.add_argument(
        "--python_exe",
        type=str,
        required=True,
        help="Python interpreter for AVH subprocess (conda env).",
    )
    ap.add_argument("--noma_model_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=os.path.join(_ROOT, "eval_runs", "fakeavceleb"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--use_unsup_avh", action="store_true")
    ap.add_argument(
        "--smart_crop",
        type=str,
        default="auto",
        choices=["off", "auto", "reel", "face"],
        help="AVH spatial pre-crop before mouth ROI (reels / on-screen UI).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--write_calibration_csvs",
        action="store_true",
        help="Write avh_for_calib.csv and noma_for_calib.csv (raw NOMA p(fake)) for tools/calibration_fit.py",
    )
    args = ap.parse_args()

    from detectors.noma import get_noma_model_path
    from orchestrator.combined_runner import run_combined_avh_to_noma

    noma_path = args.noma_model_path or get_noma_model_path()
    if not noma_path:
        raise SystemExit("NOMA model not found; set --noma_model_path or install model under model/noma-1")

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.csv")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    dfm = pd.read_csv(args.manifest_csv)
    if not {"video_path", "label"}.issubset(dfm.columns):
        raise SystemExit("manifest_csv must have columns: video_path,label")

    dfm = dfm.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    dfm = dfm.head(int(args.max_videos))

    rows: list[dict[str, Any]] = []
    for _, r in dfm.iterrows():
        video_path = str(r["video_path"])
        label = int(r["label"])
        video_name = os.path.basename(video_path)
        row: dict[str, Any] = {
            "video_path": video_path,
            "label": label,
            "avh_ok": False,
            "avh_error": None,
            "avh_score": None,
            "p_audio_mean_raw": None,
            "p_audio_mean": None,
            "p_avh_cal": None,
            "p_fused": None,
            "fusion_tension": None,
            "fusion_tau": None,
            "fusion_verdict": None,
        }
        if not os.path.isfile(video_path):
            row["avh_error"] = "missing_file"
            rows.append(row)
            continue

        res = run_combined_avh_to_noma(
            video_path=video_path,
            video_name=video_name,
            use_unsup_avh=bool(args.use_unsup_avh),
            python_exe=args.python_exe,
            smart_crop=str(args.smart_crop),
            run_forensics_cam=False,
            forensics_top_k=2,
            forensics_selection_mode="top_k",
            forensics_min_temporal_gap=24,
            forensics_max_fusion_frames=200,
            region_track_stride=1,
            run_robustness_delta=False,
            adv_ckpt_path="",
            capture_attention=False,
            export_bundle=False,
            noma_model_path=str(noma_path),
            timeout=int(args.timeout),
            persist_run_dir=None,
            cleanup_volatile_after_persist=False,
            dump_embeddings_for_cmid=False,
            noma_permutation_max_blocks=None,
        )
        row["avh_ok"] = bool(res.get("avh_ok"))
        row["avh_error"] = res.get("avh_error")
        if res.get("avh_score") is not None:
            row["avh_score"] = float(res["avh_score"])
        if res.get("p_audio_mean_raw") is not None:
            row["p_audio_mean_raw"] = float(res["p_audio_mean_raw"])
        if res.get("p_audio_mean") is not None:
            row["p_audio_mean"] = float(res["p_audio_mean"])
        if res.get("p_avh_cal") is not None:
            row["p_avh_cal"] = float(res["p_avh_cal"])
        if res.get("p_fused") is not None:
            row["p_fused"] = float(res["p_fused"])
        if res.get("fusion_tension") is not None:
            row["fusion_tension"] = float(res["fusion_tension"])
        if res.get("fusion_tau") is not None:
            row["fusion_tau"] = float(res["fusion_tau"])
        if res.get("fusion_verdict") is not None:
            row["fusion_verdict"] = str(res["fusion_verdict"])
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(results_path, index=False)

    ok = out_df[out_df["avh_ok"]].copy()
    y = ok["label"].to_numpy(dtype=float)
    metrics: dict[str, Any] = {
        "n_manifest": int(len(dfm)),
        "n_attempted": int(len(out_df)),
        "n_avh_ok": int(len(ok)),
        "scores": {},
    }
    for name, col in [
        ("p_fused", "p_fused"),
        ("p_avh_cal", "p_avh_cal"),
        ("p_audio_mean", "p_audio_mean"),
    ]:
        if col not in ok.columns:
            continue
        sub = ok[[col, "label"]].dropna()
        if len(sub) == 0:
            continue
        metrics["scores"][name] = _metrics_block(sub["label"].to_numpy(dtype=float), sub[col].to_numpy(dtype=float))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    print(f"Wrote {results_path}")
    print(f"Wrote {metrics_path}")

    if args.write_calibration_csvs and len(ok):
        # Raw AVH score + label for AVH calibration; raw mean NOMA p(fake) for NOMA calibration.
        avh_rows = []
        noma_rows = []
        for _, r in ok.iterrows():
            if r["avh_score"] is None or (isinstance(r["avh_score"], float) and np.isnan(r["avh_score"])):
                continue
            avh_rows.append({"score": float(r["avh_score"]), "label": int(r["label"])})
            if r.get("p_audio_mean_raw") is not None and not (
                isinstance(r["p_audio_mean_raw"], float) and np.isnan(r["p_audio_mean_raw"])
            ):
                noma_rows.append({"p_fake": float(r["p_audio_mean_raw"]), "label": int(r["label"])})

        if avh_rows:
            p_avh = os.path.join(args.out_dir, "avh_for_calib.csv")
            pd.DataFrame(avh_rows).to_csv(p_avh, index=False)
            print(f"Wrote {p_avh}  (use: python tools/calibration_fit.py --avh_csv {p_avh} ...)")
        if noma_rows:
            p_nm = os.path.join(args.out_dir, "noma_for_calib.csv")
            pd.DataFrame(noma_rows).to_csv(p_nm, index=False)
            print(f"Wrote {p_nm}  (use: python tools/calibration_fit.py --noma_csv {p_nm} ...)")


if __name__ == "__main__":
    main()
