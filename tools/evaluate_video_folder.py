#!/usr/bin/env python3
"""
Batch-run the combined pipeline (AVH → NOMA → reliability fusion) on every video in a folder.

No labels required. Writes:
  - results.csv / results.json  (one row per video + summary stats)
  - per_video/<stem>.json       (optional detailed dump)

Example:
  PYTHONPATH=. python tools/evaluate_video_folder.py \\
    --video_dir path/to/videos \\
    --python_exe /path/to/avh/bin/python \\
    --out_dir eval_runs/my_batch
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

_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _discover_videos(video_dir: str, *, recursive: bool = False) -> list[str]:
    if recursive:
        out: list[str] = []
        for dirpath, _, files in os.walk(video_dir):
            for name in sorted(files):
                if os.path.splitext(name)[1].lower() in _VIDEO_EXT:
                    out.append(os.path.abspath(os.path.join(dirpath, name)))
        return sorted(out)
    out = []
    for name in sorted(os.listdir(video_dir)):
        path = os.path.join(video_dir, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _VIDEO_EXT:
            out.append(os.path.abspath(path))
    return out


def _noma_series_stats(res: dict[str, Any]) -> dict[str, Any]:
    """Aggregate per-block NOMA p(fake) for variance / noise analysis."""
    df = res.get("noma_df")
    if df is None or not hasattr(df, "columns"):
        return {
            "noma_n_blocks": None,
            "noma_p_fake_mean": None,
            "noma_p_fake_std": None,
            "noma_p_fake_min": None,
            "noma_p_fake_max": None,
        }
    if "p_fake" not in df.columns or len(df) == 0:
        return {
            "noma_n_blocks": int(len(df)),
            "noma_p_fake_mean": None,
            "noma_p_fake_std": None,
            "noma_p_fake_min": None,
            "noma_p_fake_max": None,
        }
    s = pd.to_numeric(df["p_fake"], errors="coerce").dropna()
    if s.empty:
        return {
            "noma_n_blocks": 0,
            "noma_p_fake_mean": None,
            "noma_p_fake_std": None,
            "noma_p_fake_min": None,
            "noma_p_fake_max": None,
        }
    return {
        "noma_n_blocks": int(len(s)),
        "noma_p_fake_mean": float(s.mean()),
        "noma_p_fake_std": float(s.std(ddof=0)),
        "noma_p_fake_min": float(s.min()),
        "noma_p_fake_max": float(s.max()),
    }


def _cmid_cam_extra(res: dict[str, Any]) -> dict[str, Any]:
    """Summarize CMID dict and Grad-CAM paths for batch CSV."""
    import numpy as np

    out: dict[str, Any] = {
        "cmid_status": res.get("cmid_status"),
        "cmid_mean": None,
        "cmid_max": None,
        "cam_ok": res.get("cam_ok"),
        "cam_overlays_dir": res.get("cam_overlays_dir"),
        "late_fusion_mode": res.get("late_fusion_mode"),
    }
    cmid = res.get("cmid")
    if isinstance(cmid, dict):
        arr = cmid.get("cmid")
        if isinstance(arr, list) and len(arr) > 0:
            a = np.asarray(arr, dtype=float)
            out["cmid_mean"] = float(np.nanmean(a))
            out["cmid_max"] = float(np.nanmax(a))
    return out


def _row_from_result(
    video_path: str,
    label: int | None,
    res: dict[str, Any],
) -> dict[str, Any]:
    ns = _noma_series_stats(res)
    p_avh = res.get("p_avh_cal")
    p_aud = res.get("p_audio_mean")
    disagree = None
    if p_avh is not None and p_aud is not None:
        disagree = float(abs(float(p_avh) - float(p_aud)))

    cii = res.get("noma_confidence_instability")
    cii_global = None
    if isinstance(cii, dict) and cii.get("CII") is not None:
        try:
            cii_global = float(cii["CII"])
        except (TypeError, ValueError):
            pass

    row: dict[str, Any] = {
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "label": label,
        "avh_ok": bool(res.get("avh_ok")),
        "avh_error": res.get("avh_error"),
        "avh_score": res.get("avh_score"),
        "p_avh_cal": p_avh,
        "p_audio_mean_raw": res.get("p_audio_mean_raw"),
        "p_audio_mean": p_aud,
        "fusion_tension": res.get("fusion_tension"),
        "fusion_w_audio": res.get("fusion_w_audio"),
        "p_fused": res.get("p_fused"),
        "fusion_tau": res.get("fusion_tau"),
        "fusion_tau_effective": res.get("fusion_tau_effective"),
        "p_avh_soft": res.get("p_avh_soft"),
        "fusion_verdict": res.get("fusion_verdict"),
        "fusion_regime": res.get("fusion_regime"),
        "tension_index": res.get("tension_index"),
        "model_disagreement_abs": disagree,
        **ns,
        "noma_cii": cii_global,
        "use_unsup_avh": res.get("use_unsup_avh"),
        **_cmid_cam_extra(res),
    }
    return row


def _summarize_unlabeled(df: pd.DataFrame) -> dict[str, Any]:
    ok = df[df["avh_ok"] == True]  # noqa: E712
    if len(ok) == 0:
        return {"n_total": len(df), "n_ok": 0, "note": "no successful AVH runs"}

    pf = ok["p_fused"].dropna()
    verdicts = ok["fusion_verdict"].value_counts().to_dict() if "fusion_verdict" in ok.columns else {}

    return {
        "n_total": int(len(df)),
        "n_ok": int(len(ok)),
        "n_failed": int(len(df) - len(ok)),
        "p_fused_mean": float(pf.mean()) if len(pf) else None,
        "p_fused_std": float(pf.std(ddof=0)) if len(pf) > 1 else (0.0 if len(pf) == 1 else None),
        "fusion_verdict_counts": {str(k): int(v) for k, v in verdicts.items()},
        "mean_model_disagreement": float(ok["model_disagreement_abs"].dropna().mean())
        if ok["model_disagreement_abs"].notna().any()
        else None,
        "mean_noma_p_fake_std": float(ok["noma_p_fake_std"].dropna().mean())
        if "noma_p_fake_std" in ok.columns and ok["noma_p_fake_std"].notna().any()
        else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate combined pipeline on all videos in a folder.")
    ap.add_argument("--video_dir", type=str, required=True, help="Directory containing video files.")
    ap.add_argument("--python_exe", type=str, required=True, help="Conda env Python for AVH subprocess.")
    ap.add_argument("--noma_model_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--use_unsup_avh", action="store_true", help="Unsupervised AVH (default: supervised fusion).")
    ap.add_argument("--smart_crop", type=str, default="auto", choices=["off", "auto", "reel", "face"])
    ap.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="Optional CSV with columns video_name,label (0=real, 1=fake) to join by basename.",
    )
    ap.add_argument("--write_per_video_json", action="store_true")
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Include videos in subfolders (os.walk). Default: top-level only.",
    )
    ap.add_argument(
        "--dump_cmid",
        action="store_true",
        help="Run AVH with --dump_embeddings for CMID (slower; second AVH-style pass).",
    )
    ap.add_argument(
        "--run_gradcam",
        action="store_true",
        help="Run Grad-CAM mouth ROI evidence (slow; extra subprocess per video).",
    )
    ap.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Process only the first N videos after sorting (smoke tests / partial batches).",
    )
    args = ap.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    if not os.path.isdir(video_dir):
        raise SystemExit(f"Not a directory: {video_dir}")

    videos = _discover_videos(video_dir, recursive=bool(args.recursive))
    if not videos:
        raise SystemExit(f"No video files found in {video_dir}")
    if args.max_videos is not None and int(args.max_videos) > 0:
        videos = videos[: int(args.max_videos)]

    labels_by_name: dict[str, int] = {}
    if args.labels_csv and os.path.isfile(args.labels_csv):
        ldc = pd.read_csv(args.labels_csv)
        if {"video_name", "label"}.issubset(ldc.columns):
            for _, r in ldc.iterrows():
                labels_by_name[str(r["video_name"])] = int(r["label"])

    from detectors.noma import get_noma_model_path
    from orchestrator.combined_runner import run_combined_avh_to_noma

    noma_path = args.noma_model_path or get_noma_model_path()
    if not noma_path:
        raise SystemExit("NOMA model not found; set --noma_model_path or install model under model/noma-1")

    out_dir = args.out_dir or os.path.join(_ROOT, "eval_runs", os.path.basename(video_dir.rstrip(os.sep)))
    os.makedirs(out_dir, exist_ok=True)
    per_dir = os.path.join(out_dir, "per_video")
    if args.write_per_video_json:
        os.makedirs(per_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for video_path in videos:
        name = os.path.basename(video_path)
        label = labels_by_name.get(name)

        res = run_combined_avh_to_noma(
            video_path=video_path,
            video_name=name,
            use_unsup_avh=bool(args.use_unsup_avh),
            python_exe=args.python_exe,
            smart_crop=str(args.smart_crop),
            run_forensics_cam=bool(args.run_gradcam),
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
            dump_embeddings_for_cmid=bool(args.dump_cmid),
            noma_permutation_max_blocks=None,
        )

        row = _row_from_result(video_path, label, res)
        rows.append(row)

        if args.write_per_video_json:
            dump = {
                "video": name,
                "row": row,
                "temporal_corroboration": res.get("temporal_corroboration"),
                "cmid_status": res.get("cmid_status"),
            }
            stem = os.path.splitext(name)[0]
            with open(os.path.join(per_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=2, default=str)

    out_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    out_df.to_csv(csv_path, index=False)

    summary: dict[str, Any] = {
        "video_dir": video_dir,
        "n_videos": len(videos),
        "max_videos": int(args.max_videos) if args.max_videos is not None else None,
        "recursive": bool(args.recursive),
        "dump_cmid": bool(args.dump_cmid),
        "run_gradcam": bool(args.run_gradcam),
        "use_unsup_avh": bool(args.use_unsup_avh),
        "unlabeled_summary": _summarize_unlabeled(out_df),
    }

    labeled = out_df[out_df["label"].notna()]
    if len(labeled) >= 2 and labeled["label"].nunique() > 1:
        from sklearn.metrics import roc_auc_score

        sub = labeled[labeled["avh_ok"] & labeled["p_fused"].notna()].copy()
        if len(sub) >= 2:
            try:
                summary["metrics_p_fused_auc"] = float(
                    roc_auc_score(sub["label"].astype(int), sub["p_fused"].astype(float))
                )
            except Exception as e:
                summary["metrics_p_fused_auc_error"] = str(e)

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary["unlabeled_summary"], indent=2))


if __name__ == "__main__":
    main()
