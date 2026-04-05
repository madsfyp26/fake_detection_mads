#!/usr/bin/env python3
"""
Dataset study: Combined (AVH+NOMA) signals, optional adaptive fusion tuning, metrics, plots.

**Labels (evaluation):** Use real binary labels via `--labels_csv` (columns: video_name, label;
0=real, 1=fake). Optional `--use_filename_heuristic` uses a weak WhatsApp-in-name proxy only for
quick experiments — do not treat it as ground truth.

p_v = AVH calibrated p(fake); p_a = NOMA mean p(fake) — same as production fusion inputs.

Example (recommended):
  PYTHONPATH=. python tools/dataset_multimodal_analysis.py \\
    --video_dir "/path/to/videos" \\
    --python_exe /path/to/avh/bin/python \\
    --labels_csv "/path/to/labels.csv" \\
    --out_dir eval_runs/dataset_study
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


def discover_videos_recursive(video_root: str) -> list[str]:
    out: list[str] = []
    for dirpath, _, files in os.walk(video_root):
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in _VIDEO_EXT:
                out.append(os.path.abspath(os.path.join(dirpath, name)))
    return sorted(out)


def _plot_report(df: pd.DataFrame, out_dir: str, y_true: np.ndarray, y_pred_adaptive: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    real_mask = y_true == 0
    fake_mask = y_true == 1

    # 1) lip_sync proxy (|p_v - p_a|)
    fig, ax = plt.subplots(figsize=(7, 4))
    ls = np.abs(df["p_avh_cal"].astype(float) - df["p_audio_mean"].astype(float)).values
    ax.hist(ls[real_mask], bins=15, alpha=0.6, label="REAL", color="green")
    ax.hist(ls[fake_mask], bins=15, alpha=0.6, label="FAKE", color="red")
    ax.set_xlabel("|p_v - p_a| (lip-sync disagreement proxy)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Cross-modality tension distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "hist_lip_sync_proxy.png"), dpi=120)
    plt.close()

    # 2) temporal inconsistency proxy (NOMA std)
    if "noma_p_fake_std" in df.columns and df["noma_p_fake_std"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4))
        v = df["noma_p_fake_std"].astype(float).values
        ax.hist(v[real_mask], bins=15, alpha=0.6, label="REAL", color="green")
        ax.hist(v[fake_mask], bins=15, alpha=0.6, label="FAKE", color="red")
        ax.set_xlabel("NOMA p(fake) std (temporal inconsistency proxy)")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "hist_noma_temporal_std.png"), dpi=120)
        plt.close()

    # 3) scatter p_v vs p_a
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        df.loc[real_mask, "p_avh_cal"],
        df.loc[real_mask, "p_audio_mean"],
        c="green",
        alpha=0.7,
        label="REAL",
    )
    ax.scatter(
        df.loc[fake_mask, "p_avh_cal"],
        df.loc[fake_mask, "p_audio_mean"],
        c="red",
        alpha=0.7,
        label="FAKE",
    )
    ax.set_xlabel("p_v (AVH p fake)")
    ax.set_ylabel("p_a (NOMA mean p fake)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.legend()
    ax.set_title("Modality scores")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scatter_pv_pa.png"), dpi=120)
    plt.close()

    # 4) confusion matrix (adaptive)
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(y_true, y_pred_adaptive, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["REAL", "FAKE"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Adaptive fusion (tuned threshold)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_adaptive.png"), dpi=120)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Multimodal dataset analysis with adaptive fusion tuning.")
    ap.add_argument("--video_dir", type=str, required=True)
    ap.add_argument("--python_exe", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--noma_model_path", type=str, default=None)
    ap.add_argument("--max_videos", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--use_unsup_avh", action="store_true")
    ap.add_argument("--smart_crop", type=str, default="auto", choices=["off", "auto", "reel", "face"])
    ap.add_argument("--skip_inference", action="store_true", help="Load raw_results.csv from out_dir and only tune/plot.")
    ap.add_argument("--cv_proxy", action="store_true", help="Optical-flow proxy (slow).")
    ap.add_argument("--audio_proxy", action="store_true", help="Librosa stats on extracted AVH audio.")
    ap.add_argument(
        "--dump_cmid",
        action="store_true",
        help="Run AVH with embedding dump for CMID (slower).",
    )
    ap.add_argument(
        "--run_gradcam",
        action="store_true",
        help="Run Grad-CAM mouth ROI (slow).",
    )
    ap.add_argument(
        "--syncnet",
        action="store_true",
        help="Optional SyncNet placeholder column (set SYNCNET_WEIGHTS_PATH when wired).",
    )
    ap.add_argument(
        "--wav2vec",
        action="store_true",
        help="Optional wav2vec embedding norm on extracted audio (heavy deps; set WAV2VEC_DISABLED=1 to skip).",
    )
    ap.add_argument(
        "--labels_csv",
        type=str,
        default=None,
        help="CSV with columns video_name, label (0=real, 1=fake). Required for meaningful accuracy/F1 unless using --use_filename_heuristic.",
    )
    ap.add_argument(
        "--use_filename_heuristic",
        action="store_true",
        help="Weak proxy: 'whatsapp' in filename -> REAL. Not ground truth; prefer --labels_csv.",
    )
    args = ap.parse_args()

    if args.labels_csv and args.use_filename_heuristic:
        raise SystemExit("Use only one of --labels_csv or --use_filename_heuristic.")

    video_dir = os.path.abspath(args.video_dir)
    out_dir = args.out_dir or os.path.join(_ROOT, "eval_runs", "dataset_multimodal_analysis")
    os.makedirs(out_dir, exist_ok=True)
    raw_csv = os.path.join(out_dir, "raw_results.csv")

    from detectors.noma import get_noma_model_path
    from explainability.adaptive_fusion_tune import (
        best_threshold_for_scores,
        grid_search_fusion_and_threshold,
        lip_sync_error_score,
    )
    from orchestrator.combined_runner import run_combined_avh_to_noma
    from tools.evaluate_video_folder import _row_from_result
    from tools.label_utils import heuristic_label_whatsapp_proxy, load_labels_csv

    if args.labels_csv:
        label_source = "labels_csv"
    elif args.use_filename_heuristic:
        label_source = "filename_heuristic_whatsapp_proxy"
    else:
        label_source = "none"

    if args.skip_inference:
        if not os.path.isfile(raw_csv):
            raise SystemExit(f"Missing {raw_csv}; run without --skip_inference first.")
        df = pd.read_csv(raw_csv)
        if "label" not in df.columns:
            df["label"] = np.nan
        if args.labels_csv:
            m = load_labels_csv(args.labels_csv)
            df["video_name"] = df["video_name"].astype(str).str.strip()
            df["label"] = df["video_name"].map(lambda n: m.get(n, np.nan))
        elif args.use_filename_heuristic:
            df["label"] = df["video_name"].map(heuristic_label_whatsapp_proxy)
    else:
        from detectors.cv_audio_proxies import librosa_audio_proxies, optical_flow_temporal_proxy

        noma_path = args.noma_model_path or get_noma_model_path()
        if not noma_path:
            raise SystemExit("NOMA model not found.")

        videos = discover_videos_recursive(video_dir)
        if args.max_videos is not None:
            videos = videos[: int(args.max_videos)]

        label_map: dict[str, int] | None = None
        if args.labels_csv:
            label_map = load_labels_csv(args.labels_csv)

        rows: list[dict[str, Any]] = []
        for i, video_path in enumerate(videos):
            name = os.path.basename(video_path)
            if label_map is not None:
                lv = label_map.get(name)
                label: Any = np.nan if lv is None else int(lv)
            elif args.use_filename_heuristic:
                label = heuristic_label_whatsapp_proxy(name)
            else:
                label = np.nan
            if isinstance(label, float) and np.isnan(label):
                lbl_disp = "unlabeled"
            else:
                lbl_disp = "REAL" if int(label) == 0 else "FAKE"
            print(f"[{i + 1}/{len(videos)}] {name}  (label={lbl_disp})", flush=True)

            if not os.path.isfile(video_path):
                rows.append(
                    {
                        "video_path": video_path,
                        "video_name": name,
                        "label": label,
                        "avh_ok": False,
                        "avh_error": "missing_file",
                    }
                )
                continue

            try:
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
            except Exception as e:
                rows.append(
                    {
                        "video_path": video_path,
                        "video_name": name,
                        "label": label,
                        "avh_ok": False,
                        "avh_error": str(e),
                    }
                )
                continue

            row = _row_from_result(video_path, label, res)
            row["lip_sync_error_score"] = None
            row["temporal_inconsistency_score"] = row.get("noma_p_fake_std")
            if row.get("p_avh_cal") is not None and row.get("p_audio_mean") is not None:
                row["lip_sync_error_score"] = lip_sync_error_score(
                    float(row["p_avh_cal"]),
                    float(row["p_audio_mean"]),
                )

            if args.cv_proxy:
                cv = optical_flow_temporal_proxy(video_path)
                row["cv_flow_ok"] = cv.get("ok")
                row["temporal_cv_score"] = cv.get("temporal_inconsistency_score")
            else:
                row["temporal_cv_score"] = None

            if args.audio_proxy and res.get("audio_path") and os.path.isfile(str(res.get("audio_path"))):
                apx = librosa_audio_proxies(str(res["audio_path"]))
                row["audio_proxy_ok"] = apx.get("ok")
                row["mfcc_energy_mean"] = apx.get("mfcc_energy_mean")
                row["zcr_mean"] = apx.get("zcr_mean")
                row["spectral_contrast_mean"] = apx.get("spectral_contrast_mean")
            else:
                row["mfcc_energy_mean"] = None
                row["zcr_mean"] = None
                row["spectral_contrast_mean"] = None

            if args.syncnet:
                from detectors.syncnet_score import run_syncnet_score

                sn = run_syncnet_score(video_path, audio_path=str(res.get("audio_path") or ""))
                row["syncnet_ok"] = sn.get("ok")
                row["syncnet_score"] = sn.get("sync_score")
                row["syncnet_error"] = sn.get("error")

            if args.wav2vec and res.get("audio_path") and os.path.isfile(str(res.get("audio_path"))):
                from detectors.wav2vec_audio_proxy import wav2vec_embedding_proxy

                w2 = wav2vec_embedding_proxy(str(res["audio_path"]))
                row["wav2vec_ok"] = w2.get("ok")
                row["wav2vec_embedding_norm_mean"] = w2.get("embedding_norm_mean")
                row["wav2vec_error"] = w2.get("error")

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(raw_csv, index=False)
        print(f"Wrote {raw_csv}", flush=True)

    ok = df[df["avh_ok"] == True].copy()  # noqa: E712
    ok = ok[ok["p_avh_cal"].notna() & ok["p_audio_mean"].notna()].copy()
    if len(ok) < 2:
        raise SystemExit("Need at least 2 successful rows with p_avh_cal and p_audio_mean.")

    ok["label"] = pd.to_numeric(ok["label"], errors="coerce")
    labeled = ok[ok["label"].notna() & ok["label"].isin([0, 1])].copy()
    can_eval = len(labeled) >= 2 and labeled["label"].nunique() >= 2

    if not can_eval:
        insights_skip: dict[str, Any] = {
            "label_source": label_source,
            "evaluation_skipped": True,
            "reason": (
                "Need labeled rows with both classes (0=real, 1=fake). "
                "Pass --labels_csv with real labels, or --use_filename_heuristic for a weak proxy only."
            ),
            "n_samples_ok": int(len(ok)),
            "n_labeled_for_eval": int(len(labeled)),
            "flags": {
                "dump_cmid": bool(args.dump_cmid),
                "run_gradcam": bool(args.run_gradcam),
                "syncnet": bool(args.syncnet),
                "wav2vec": bool(args.wav2vec),
            },
        }
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(insights_skip, f, indent=2, sort_keys=True)
        with open(os.path.join(out_dir, "failure_cases.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "note": "No failure list: evaluation skipped (insufficient labels or single class only).",
                    "false_positives_fake_predicted": [],
                    "false_negatives_missed_fake": [],
                },
                f,
                indent=2,
            )
        with open(os.path.join(out_dir, "IMPROVEMENTS.txt"), "w", encoding="utf-8") as f:
            f.write(
                "Add --labels_csv with columns video_name,label (0=real,1=fake) for real-label metrics.\n"
            )
        print(f"Wrote {metrics_path} (evaluation skipped — see reason in file)")
        print("Done.")
        return

    y_true = labeled["label"].astype(int).values
    p_v = labeled["p_avh_cal"].astype(float).values
    p_a = labeled["p_audio_mean"].astype(float).values

    # Baseline: production p_fused + best threshold
    baseline = {}
    if labeled["p_fused"].notna().any():
        baseline = best_threshold_for_scores(y_true, labeled["p_fused"].astype(float).values)

    tuned = grid_search_fusion_and_threshold(p_v, p_a, y_true)
    if tuned.get("error"):
        raise SystemExit(f"Adaptive fusion tuning failed: {tuned}")

    insights: dict[str, Any] = {
        "n_samples_ok": int(len(ok)),
        "n_labeled_for_eval": int(len(labeled)),
        "label_source": label_source,
        "flags": {
            "dump_cmid": bool(args.dump_cmid),
            "run_gradcam": bool(args.run_gradcam),
            "syncnet": bool(args.syncnet),
            "wav2vec": bool(args.wav2vec),
        },
        "baseline_pipeline_p_fused_threshold_search": baseline,
        "adaptive_fusion_grid_search": {k: v for k, v in tuned.items() if k not in ("p_fused_adaptive", "y_pred")},
        "recommended_fusion_params": {
            "tau": tuned.get("tau"),
            "threshold": tuned.get("threshold"),
            "tension_boost_beta": tuned.get("tension_boost_beta"),
            "epsilon": tuned.get("epsilon"),
        },
    }

    labeled = labeled.copy()
    labeled["p_fused_adaptive"] = tuned.get("p_fused_adaptive", [])
    y_pred_list = tuned.get("y_pred", [])
    if len(y_pred_list) == len(labeled):
        labeled["y_pred_adaptive"] = y_pred_list
    else:
        labeled["y_pred_adaptive"] = np.nan

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, sort_keys=True)

    tuned_csv = os.path.join(out_dir, "per_file_results.csv")
    base_cols = [
        "video_name",
        "label",
        "p_avh_cal",
        "p_audio_mean",
        "p_fused",
        "fusion_verdict",
        "fusion_regime",
        "lip_sync_error_score",
        "tension_index",
        "noma_p_fake_std",
    ]
    base_cols = [c for c in base_cols if c in labeled.columns]
    out_df = labeled[base_cols].copy()
    if "p_fused_adaptive" in tuned:
        out_df["p_fused_adaptive"] = tuned["p_fused_adaptive"]
    if len(y_pred_list) == len(out_df):
        out_df["y_pred_adaptive"] = y_pred_list
    out_df.to_csv(tuned_csv, index=False)
    print(f"Wrote {metrics_path}\nWrote {tuned_csv}")

    # Failure analysis: false positives / negatives
    y_pred = np.array(tuned["y_pred"], dtype=int)
    fp = labeled[(y_true == 0) & (y_pred == 1)]
    fn = labeled[(y_true == 1) & (y_pred == 0)]
    failures = {
        "false_positives_fake_predicted": fp["video_name"].tolist(),
        "false_negatives_missed_fake": fn["video_name"].tolist(),
    }
    with open(os.path.join(out_dir, "failure_cases.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    # Plots
    try:
        _plot_report(labeled, os.path.join(out_dir, "plots"), y_true, y_pred)
        print(f"Wrote plots under {os.path.join(out_dir, 'plots')}")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    # System improvement notes (static)
    notes = """\
SYSTEM IMPROVEMENT SUGGESTIONS (integrate as follow-up work)
- Stronger lip-sync: SyncNet / AV-HuBERT embeddings with CMID (enable dump_embeddings in combined_runner).
- Audio: wav2vec / rawnet for synthetic speech; keep NOMA for spectral features.
- Video: lightweight temporal Transformer on frame embeddings; keyframe sampling before AVH.
- Fusion: merge recommended tau/threshold from metrics.json into calibration_artifacts.json after validation on held-out data.
"""
    with open(os.path.join(out_dir, "IMPROVEMENTS.txt"), "w", encoding="utf-8") as f:
        f.write(notes)
    print("Done.")


if __name__ == "__main__":
    main()
