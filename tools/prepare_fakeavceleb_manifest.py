#!/usr/bin/env python3
"""
Build a manifest CSV (video_path,label) for FakeAVCeleb-style folder layouts.

Does not download or scrape the web. After you obtain FakeAVCeleb via the official
release, point --dataset_root at the extracted tree.

Label convention (binary fake detection):
  0 = real (RealVideo-RealAudio)
  1 = fake (any modality involving FakeVideo or FakeAudio per path heuristics)

If paths do not match known patterns, they are skipped unless --from_list supplies labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _normalize_rel(rel: str) -> str:
    return rel.replace("\\", "/").lower()


def infer_fakeavceleb_label_from_path(rel_path: str) -> int | None:
    """
    Map a path relative to dataset root to 0/1 using common FakeAVCeleb combo folder names.
    """
    s = _normalize_rel(rel_path)
    # Order matters: check fully-real combo first.
    if "realvideo-realaudio" in s or "real_video_real_audio" in s:
        return 0
    if (
        "realvideo-fakeaudio" in s
        or "fakevideo-realaudio" in s
        or "fakevideo-fakeaudio" in s
        or "fake_video_fake_audio" in s
        or "fake_video_real_audio" in s
        or "real_video_fake_audio" in s
    ):
        return 1
    return None


def _iter_videos(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            suf = Path(fn).suffix.lower()
            if suf in _VIDEO_EXTS:
                out.append(Path(dirpath) / fn)
    return sorted(out)


def _load_from_list(path: str, dataset_root: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                p, lab = line.split(",", 1)
                p = p.strip()
                lab_i = int(lab.strip())
            else:
                p = line
                rel = os.path.relpath(p, dataset_root) if os.path.isabs(p) else p
                lab_i = infer_fakeavceleb_label_from_path(rel)
                if lab_i is None:
                    raise SystemExit(f"Could not infer label for list entry (add ,0 or ,1): {line}")
            abs_p = p if os.path.isabs(p) else str(dataset_root / p)
            rows.append((abs_p, lab_i))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build manifest CSV for FakeAVCeleb evaluation.")
    ap.add_argument("--dataset_root", type=str, required=True, help="Root folder of extracted FakeAVCeleb.")
    ap.add_argument(
        "--out_csv",
        type=str,
        default="fakeavceleb_manifest.csv",
        help="Output CSV path with columns: video_path,label",
    )
    ap.add_argument(
        "--from_list",
        type=str,
        default=None,
        help="Optional text file: one path per line, or path,label per line (0=real, 1=fake).",
    )
    args = ap.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        raise SystemExit(1)

    entries: list[tuple[str, int]] = []
    skipped = 0

    if args.from_list:
        entries.extend(_load_from_list(args.from_list, root))
    else:
        for vp in _iter_videos(root):
            rel = str(vp.relative_to(root))
            lab = infer_fakeavceleb_label_from_path(rel)
            if lab is None:
                skipped += 1
                continue
            entries.append((str(vp.resolve()), lab))

    if not entries:
        print("No labeled videos found. Use --from_list or check folder names.", file=sys.stderr)
        raise SystemExit(2)

    out_path = Path(args.out_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "label"])
        w.writeheader()
        for p, lab in entries:
            w.writerow({"video_path": p, "label": lab})

    print(f"Wrote {len(entries)} rows to {out_path}")
    if skipped:
        print(f"Skipped {skipped} videos (unknown label heuristics).")


if __name__ == "__main__":
    main()
