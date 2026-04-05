#!/usr/bin/env python3
"""
Download LipSynthesis public character preview videos (WebM) via their public API,
convert to MP4 with ffmpeg, and save under --out-dir.

Uses https://api.lipsynthesis.com/api/characters — each item may include
videoExampleUrl (time-limited signed URL). Re-fetch the API when URLs expire.

Respect LipSynthesis terms of use; previews are exposed for the web app catalog.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://api.lipsynthesis.com/api/characters"


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"[-\s]+", "_", s)
    return s or "video"


def fetch_characters_page(page: int, per_page: int) -> dict:
    q = urllib.parse.urlencode({"page": page, "perPage": per_page})
    url = f"{API}?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "fake-audio-detection-tools/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "fake-audio-detection-tools/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)


def webm_to_mp4(webm: Path, mp4: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(webm),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(mp4),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description="Fetch LipSynthesis catalog preview WebMs and convert to MP4.")
    p.add_argument(
        "--limit",
        type=int,
        default=23,
        help="Max avatar preview videos to download (default 23; use 20--25 for a small batch)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("untitled folder 2"),
        help='Output directory (default: "untitled folder 2")',
    )
    p.add_argument("--keep-webm", action="store_true", help="Keep intermediate .webm files")
    args = p.parse_args()

    done = 0
    args.out_dir = args.out_dir.resolve()
    # API returns paginated catalog (~500 avatars); request up to 50 per page.
    per_page = min(50, max(args.limit, 1))
    page = 1
    page_count = 1

    while done < args.limit and page <= page_count:
        payload = fetch_characters_page(page=page, per_page=per_page)
        rows = payload.get("data") or []
        page_count = int(payload.get("pageCount") or 1)
        if not rows:
            break
        for row in rows:
            if done >= args.limit:
                break
            name = row.get("name") or "character"
            cid = (row.get("id") or "unknown")[:8]
            vid_url = row.get("videoExampleUrl")
            if not vid_url:
                continue
            base = _slug(f"{name}_{cid}")
            webm_path = args.out_dir / f"{base}.webm"
            mp4_path = args.out_dir / f"{base}.mp4"
            print(f"Downloading: {name} -> {webm_path.name}")
            try:
                download_file(vid_url, webm_path)
            except urllib.error.HTTPError as e:
                print(f"  HTTP error {e.code} for {name}; URL may have expired — re-run to refresh.", file=sys.stderr)
                continue
            print(f"Converting -> {mp4_path.name}")
            webm_to_mp4(webm_path, mp4_path)
            if not args.keep_webm:
                webm_path.unlink(missing_ok=True)
            done += 1
        page += 1

    if done == 0:
        print("No videoExampleUrl entries downloaded. Try again later or increase --limit.", file=sys.stderr)
        return 1
    print(f"Done: {done} file(s) in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
