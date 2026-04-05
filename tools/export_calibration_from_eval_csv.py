#!/usr/bin/env python3
"""
Build calibration CSVs + stratified train/holdout from dataset_multimodal_analysis raw_results.csv.

Labels must come from real annotations by default:
  - Prefer `label` column already in raw_results (from --labels_csv run), or
  - Pass --labels_csv to merge video_name -> label, or
  - Pass --use_filename_heuristic only as a weak WhatsApp-in-name proxy (not ground truth).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tools.label_utils import heuristic_label_whatsapp_proxy, load_labels_csv


def export_from_raw_results(
    raw_csv: str,
    out_dir: str,
    *,
    labels_csv: str | None = None,
    use_filename_heuristic: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Writes:
      avh_train.csv, avh_holdout.csv  (columns: score, label)
      noma_train.csv, noma_holdout.csv (columns: p_fake, label)
      train_manifest.csv — subset of rows used for train (for fusion tuning)
      export_meta.json — counts, stratify ok flag
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)
    if "video_name" not in df.columns:
        raise SystemExit("raw_results.csv must contain video_name")

    df = df[df.get("avh_ok") == True].copy()  # noqa: E712
    if "avh_score" not in df.columns or "p_audio_mean_raw" not in df.columns:
        raise SystemExit("raw_results.csv must contain avh_score and p_audio_mean_raw")

    if labels_csv:
        m = load_labels_csv(labels_csv)
        df["video_name"] = df["video_name"].astype(str).str.strip()
        df["label"] = df["video_name"].map(lambda n: m.get(n, np.nan))
    elif use_filename_heuristic:
        df["label"] = df["video_name"].map(heuristic_label_whatsapp_proxy)
    elif "label" in df.columns and df["label"].notna().any():
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    else:
        raise SystemExit(
            "No labels: pass --labels_csv (video_name, label), or ensure raw_results.csv has a "
            "non-empty label column from a labeled run, or use --use_filename_heuristic (weak proxy only)."
        )

    df = df.dropna(subset=["avh_score", "p_audio_mean_raw", "label"])
    df["avh_score"] = pd.to_numeric(df["avh_score"], errors="coerce")
    df["p_audio_mean_raw"] = pd.to_numeric(df["p_audio_mean_raw"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["avh_score", "p_audio_mean_raw", "label"])
    df["label"] = df["label"].astype(int)

    n = len(df)
    if n < 4:
        raise SystemExit(f"Need at least 4 ok rows with labels for a holdout split; got {n}")

    y = df["label"].astype(int).values
    stratify = y if len(np.unique(y)) >= 2 and min(np.bincount(y)) >= 2 else None
    try:
        train_idx, hold_idx = train_test_split(
            np.arange(n),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        train_idx, hold_idx = train_test_split(
            np.arange(n),
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
        stratify_ok = False
    else:
        stratify_ok = stratify is not None

    train_df = df.iloc[train_idx].reset_index(drop=True)
    hold_df = df.iloc[hold_idx].reset_index(drop=True)

    avh_train = train_df[["avh_score", "label"]].rename(columns={"avh_score": "score"})
    avh_hold = hold_df[["avh_score", "label"]].rename(columns={"avh_score": "score"})
    noma_train = train_df[["p_audio_mean_raw", "label"]].rename(columns={"p_audio_mean_raw": "p_fake"})
    noma_hold = hold_df[["p_audio_mean_raw", "label"]].rename(columns={"p_audio_mean_raw": "p_fake"})

    avh_train.to_csv(os.path.join(out_dir, "avh_train.csv"), index=False)
    avh_hold.to_csv(os.path.join(out_dir, "avh_holdout.csv"), index=False)
    noma_train.to_csv(os.path.join(out_dir, "noma_train.csv"), index=False)
    noma_hold.to_csv(os.path.join(out_dir, "noma_holdout.csv"), index=False)

    df[["avh_score", "label"]].rename(columns={"avh_score": "score"}).to_csv(
        os.path.join(out_dir, "avh_all.csv"), index=False
    )
    df[["p_audio_mean_raw", "label"]].rename(columns={"p_audio_mean_raw": "p_fake"}).to_csv(
        os.path.join(out_dir, "noma_all.csv"), index=False
    )

    cols_keep = [
        "video_name",
        "label",
        "avh_score",
        "p_audio_mean_raw",
    ]
    extra = [c for c in ("p_avh_cal", "p_audio_mean", "p_fused") if c in train_df.columns]
    train_df[list(dict.fromkeys(cols_keep + extra))].to_csv(
        os.path.join(out_dir, "train_manifest.csv"),
        index=False,
    )

    label_origin = "labels_csv" if labels_csv else ("filename_heuristic" if use_filename_heuristic else "raw_results_column")

    meta = {
        "n_rows_ok": int(n),
        "n_train": int(len(train_df)),
        "n_holdout": int(len(hold_df)),
        "stratified": bool(stratify_ok),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "label_origin": label_origin,
    }
    with open(os.path.join(out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Export calibration train/holdout CSVs from raw_results.csv")
    ap.add_argument(
        "--raw_results_csv",
        type=str,
        default=os.path.join(_ROOT, "eval_runs", "untitled2_baseline_current", "raw_results.csv"),
    )
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, default=None, help="Merge labels by video_name (overrides raw column).")
    ap.add_argument(
        "--use_filename_heuristic",
        action="store_true",
        help="Weak WhatsApp-in-name proxy only; not ground truth.",
    )
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    if args.labels_csv and args.use_filename_heuristic:
        raise SystemExit("Use only one of --labels_csv or --use_filename_heuristic.")

    if not os.path.isfile(args.raw_results_csv):
        raise SystemExit(f"Missing {args.raw_results_csv}")

    meta = export_from_raw_results(
        args.raw_results_csv,
        args.out_dir,
        labels_csv=args.labels_csv,
        use_filename_heuristic=bool(args.use_filename_heuristic),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(json.dumps(meta, indent=2))
    print(f"Wrote calibration splits under {args.out_dir}")


if __name__ == "__main__":
    main()
