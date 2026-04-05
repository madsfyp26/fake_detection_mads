#!/usr/bin/env python3
"""
CLI for reviewer-facing PNG figures (same builders as explainability/reviewer_figures.py).

Examples:
  PYTHONPATH=. python tools/reviewer_figures.py triptych \\
    --roi path/to/mouth_roi.mp4 --frame 12 \\
    --fused path/to/fused_heatmap.npy \\
    --cam-volume path/to/cam_volume.npy \\
    --overlay-dir path/to/overlays \\
    --out triptych.png

  PYTHONPATH=. python tools/reviewer_figures.py mel-noma --audio wav.wav --csv noma.csv --out mel.png

  PYTHONPATH=. python tools/reviewer_figures.py calibration --out calib.png
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="Export reviewer PNG figures")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triptych", help="ROI | Grad-CAM | fused heatmap")
    t.add_argument("--roi", required=True, help="mouth_roi.mp4 path")
    t.add_argument("--frame", type=int, required=True)
    t.add_argument("--fused", default="", help="fused_heatmap.npy")
    t.add_argument("--cam-volume", default="", dest="cam_volume")
    t.add_argument("--overlay-dir", default="", dest="overlay_dir")
    t.add_argument("--out", required=True)

    m = sub.add_parser("mel-noma", help="Mel spectrogram + NOMA p(fake) line")
    m.add_argument("--audio", required=True)
    m.add_argument("--csv", required=True, help="CSV with Seconds,p_fake columns")
    m.add_argument("--out", required=True)

    c = sub.add_parser("cmid", help="Cosine + CMID vs time (JSON from combined result)")
    c.add_argument("--json", required=True, help='JSON: {"similarity":[...],"cmid":[...]}')
    c.add_argument("--out", required=True)

    a = sub.add_parser("attention-cam", help="attention_per_frame vs cam_per_frame (JSON index.json)")
    a.add_argument("--index-json", required=True)
    a.add_argument("--out", required=True)

    cal = sub.add_parser("calibration", help="AVH + NOMA calibration curves")
    cal.add_argument("--out", required=True)

    args = p.parse_args()

    if args.cmd == "triptych":
        from explainability.reviewer_figures import figure_triptych_png_bytes

        b = figure_triptych_png_bytes(
            roi_path=args.roi,
            frame_idx=args.frame,
            fused_npy_path=args.fused or None,
            cam_npy_path=args.cam_volume or None,
            overlay_dir=args.overlay_dir or None,
        )
        with open(args.out, "wb") as f:
            f.write(b)
        print(f"Wrote {args.out} ({len(b)} bytes)")

    elif args.cmd == "mel-noma":
        import pandas as pd

        from explainability.reviewer_figures import figure_mel_noma_png_bytes

        df = pd.read_csv(args.csv)
        b = figure_mel_noma_png_bytes(
            audio_path=args.audio,
            seconds=df["Seconds"].values,
            p_fake=df["p_fake"].values,
        )
        with open(args.out, "wb") as f:
            f.write(b)
        print(f"Wrote {args.out} ({len(b)} bytes)")

    elif args.cmd == "cmid":
        import json

        from explainability.reviewer_figures import figure_cmid_png_bytes

        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
        b = figure_cmid_png_bytes(data)
        with open(args.out, "wb") as f:
            f.write(b)
        print(f"Wrote {args.out} ({len(b)} bytes)")

    elif args.cmd == "attention-cam":
        import json

        from explainability.reviewer_figures import figure_attention_cam_png_bytes

        with open(args.index_json, "r", encoding="utf-8") as f:
            idx = json.load(f)
        b = figure_attention_cam_png_bytes(idx)
        with open(args.out, "wb") as f:
            f.write(b)
        print(f"Wrote {args.out} ({len(b)} bytes)")

    elif args.cmd == "calibration":
        from explainability.reviewer_figures import figure_calibration_png_bytes

        b = figure_calibration_png_bytes()
        with open(args.out, "wb") as f:
            f.write(b)
        print(f"Wrote {args.out} ({len(b)} bytes)")


if __name__ == "__main__":
    main()
