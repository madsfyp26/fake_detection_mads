#!/usr/bin/env python3
"""
Fit AVH + NOMA calibration on train CSVs (from export_calibration_from_eval_csv.py) and merge into a
single calibration JSON suitable for CALIBRATION_ARTIFACTS_PATH.

Example:
  PYTHONPATH=. python tools/fit_calibration_train.py \\
    --avh_train_csv artifacts/calib_export/avh_train.csv \\
    --noma_train_csv artifacts/calib_export/noma_train.csv \\
    --base_calibration calibration_artifacts.json \\
    --out_path artifacts/calibration_artifacts_tuned.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from calibration_runtime import DEFAULTS
from config import PROJECT_ROOT
from tools.calibration_fit import fit_avh_from_csv, fit_noma_from_csv


def _load_base(path: str | None) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit calibration on train splits; merge with base JSON.")
    ap.add_argument("--avh_train_csv", type=str, default=None)
    ap.add_argument("--noma_train_csv", type=str, default=None)
    ap.add_argument(
        "--base_calibration",
        type=str,
        default=None,
        help="JSON to merge (fusion hyperparams etc.). Default: repo calibration_artifacts.json if present.",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "calibration_artifacts_tuned.json"),
    )
    ap.add_argument("--uncertain_rate", type=float, default=0.2)
    args = ap.parse_args()

    if not args.avh_train_csv and not args.noma_train_csv:
        raise SystemExit("Provide at least one of --avh_train_csv or --noma_train_csv.")

    base_path = args.base_calibration
    if base_path is None:
        for cand in (
            os.path.join(PROJECT_ROOT, "calibration_artifacts.json"),
            os.path.join(PROJECT_ROOT, "artifacts", "calibration_artifacts.json"),
        ):
            if os.path.isfile(cand):
                base_path = cand
                break

    merged: dict = {**DEFAULTS, **_load_base(base_path)}

    if args.avh_train_csv:
        if not os.path.isfile(args.avh_train_csv):
            raise SystemExit(f"Missing {args.avh_train_csv}")
        merged.update(fit_avh_from_csv(args.avh_train_csv, uncertain_rate=args.uncertain_rate))
    if args.noma_train_csv:
        if not os.path.isfile(args.noma_train_csv):
            raise SystemExit(f"Missing {args.noma_train_csv}")
        merged.update(fit_noma_from_csv(args.noma_train_csv, uncertain_rate=args.uncertain_rate))

    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Only persist keys that are numbers (calibration_fit returns floats)
    serializable = {k: v for k, v in merged.items() if isinstance(v, (int, float, str, bool))}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {out_path}")
    if base_path:
        print(f"Merged base from: {base_path}")


if __name__ == "__main__":
    main()
