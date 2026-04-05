#!/usr/bin/env python3
"""
Compare two metrics JSON files (e.g. dataset_multimodal_analysis metrics.json before/after tuning).

Prints a flat diff for common scalar keys and nested dicts (one level).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _flatten_metrics(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                if isinstance(v, dict):
                    out.update(_flatten_metrics(v, key))
                else:
                    out[key] = json.dumps(v, sort_keys=True)
            else:
                out[key] = v
    return out


def compare_metrics(a_path: str, b_path: str) -> dict[str, Any]:
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(b_path, encoding="utf-8") as f:
        b = json.load(f)

    fa = _flatten_metrics(a)
    fb = _flatten_metrics(b)
    keys = sorted(set(fa) | set(fb))
    rows: list[dict[str, Any]] = []
    for k in keys:
        va, vb = fa.get(k), fb.get(k)
        if va != vb:
            rows.append({"key": k, "a": va, "b": vb, "delta": _delta(va, vb)})
    return {
        "path_a": os.path.abspath(a_path),
        "path_b": os.path.abspath(b_path),
        "n_keys_a": len(fa),
        "n_keys_b": len(fb),
        "n_differing": len(rows),
        "diffs": rows,
    }


def _delta(va: Any, vb: Any) -> Any:
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return float(vb) - float(va)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two metrics.json files.")
    ap.add_argument("path_a", type=str)
    ap.add_argument("path_b", type=str)
    ap.add_argument("--out_json", type=str, default=None, help="Write full diff JSON here.")
    args = ap.parse_args()

    if not os.path.isfile(args.path_a):
        raise SystemExit(f"Missing {args.path_a}")
    if not os.path.isfile(args.path_b):
        raise SystemExit(f"Missing {args.path_b}")

    report = compare_metrics(args.path_a, args.path_b)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
        print(f"Wrote {args.out_json}")

    print(json.dumps({k: v for k, v in report.items() if k != "diffs"}, indent=2))
    for d in report["diffs"]:
        print(f"  {d['key']}: {d['a']!r} -> {d['b']!r}", end="")
        if d.get("delta") is not None:
            print(f" (Δ {d['delta']:+.6g})")
        else:
            print()


if __name__ == "__main__":
    main()
