from __future__ import annotations

import numpy as np


def select_top_cam_frames(
    frame_intensity: np.ndarray,
    top_k: int,
    mode: str = "top_k",
    min_temporal_gap: int = 24,
) -> list[int]:
    frame_intensity = np.asarray(frame_intensity, dtype=float).reshape(-1)
    n = int(frame_intensity.shape[0])
    if n <= 0:
        return [0]
    top_k = max(1, min(int(top_k), n))
    gap = max(1, int(min_temporal_gap))
    order = np.argsort(frame_intensity)[::-1]

    if mode == "top_k":
        picked = [int(x) for x in order[:top_k].tolist()]
        return picked if picked else [0]

    if mode == "diverse_topk":
        picked: list[int] = []
        for idx in order:
            i = int(idx)
            if all(abs(i - p) >= gap for p in picked):
                picked.append(i)
                if len(picked) >= top_k:
                    break
        if len(picked) < top_k:
            for idx in order:
                i = int(idx)
                if i not in picked:
                    picked.append(i)
                if len(picked) >= top_k:
                    break
        return picked if picked else [0]

    if mode == "temporal_peaks":
        peaks: list[int] = []
        if n == 1:
            peaks = [0]
        else:
            for i in range(n):
                left = frame_intensity[i - 1] if i > 0 else -np.inf
                right = frame_intensity[i + 1] if i < n - 1 else -np.inf
                if frame_intensity[i] >= left and frame_intensity[i] >= right:
                    peaks.append(i)
        peak_order = sorted(peaks, key=lambda i: frame_intensity[i], reverse=True)
        picked: list[int] = []
        for i in peak_order:
            if all(abs(i - p) >= gap for p in picked):
                picked.append(int(i))
                if len(picked) >= top_k:
                    break
        if len(picked) < top_k:
            for idx in order:
                i = int(idx)
                if i in picked:
                    continue
                if all(abs(i - p) >= gap for p in picked):
                    picked.append(i)
                if len(picked) >= top_k:
                    break
        if len(picked) < top_k:
            for idx in order:
                i = int(idx)
                if i not in picked:
                    picked.append(i)
                if len(picked) >= top_k:
                    break
        return picked if picked else [0]

    picked = [int(x) for x in order[:top_k].tolist()]
    return picked if picked else [0]
