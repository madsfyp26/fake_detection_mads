"""
High-impact reviewer figures: triptych (ROI | Grad-CAM | fused), mel + NOMA, CMID,
attention vs CAM, and calibration curves.

Produces PNG bytes for Streamlit, downloads, or slides. Optional deps: matplotlib,
opencv-python (for video frames / JET), librosa (mel).
"""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np


def _ensure_matplotlib_agg() -> None:
    import matplotlib

    matplotlib.use("Agg")

# ─── Triptych: ROI frame | overlay or CAM | fused heatmap ───────────────────


def _resize_rgb(img: np.ndarray, target_h: int) -> np.ndarray:
    """img: (H,W,3) uint8 RGB."""
    import cv2

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = target_h / float(h)
    nh, nw = target_h, max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _read_roi_frame_rgb(roi_path: str, frame_idx: int) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(roi_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _gray_to_jet_bgr(cam2d: np.ndarray) -> np.ndarray:
    import cv2

    c = np.asarray(cam2d, dtype=float)
    c = (c - c.min()) / (c.max() - c.min() + 1e-9)
    u8 = (c * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_JET)


def _inferno_rgb_from_2d(z: np.ndarray) -> np.ndarray:
    """Scalar field (H,W) to RGB uint8 using matplotlib inferno."""
    _ensure_matplotlib_agg()
    import matplotlib.cm as cm

    z = np.asarray(z, dtype=float)
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    rgba = cm.inferno(z)[:, :, :3]
    return (rgba * 255).astype(np.uint8)


def figure_triptych_png_bytes(
    *,
    roi_path: str,
    frame_idx: int,
    fused_npy_path: str | None = None,
    cam_npy_path: str | None = None,
    overlay_dir: str | None = None,
    target_height: int = 320,
) -> bytes:
    """
    Single horizontal PNG: [ROI frame | Grad-CAM overlay or CAM slice | fused slice].

    Overlay PNG used if ``overlay_dir/cam_frame_{idx:05d}.png`` exists; else CAM volume slice.
    """
    import cv2
    from PIL import Image

    left = _read_roi_frame_rgb(roi_path, frame_idx)
    if left is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {roi_path}")

    center_bgr = None
    if overlay_dir:
        png = os.path.join(overlay_dir, f"cam_frame_{frame_idx:05d}.png")
        if os.path.isfile(png):
            center_bgr = cv2.imread(png)

    if center_bgr is None and cam_npy_path and os.path.isfile(cam_npy_path):
        vol = np.load(cam_npy_path)
        if vol.ndim == 3 and frame_idx < vol.shape[0]:
            heat = vol[frame_idx]
            center_bgr = _gray_to_jet_bgr(heat)

    if center_bgr is None:
        # Placeholder: grayscale ROI duplicated as "no CAM"
        g = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        center_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    center_rgb = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2RGB)

    right_rgb = None
    if fused_npy_path and os.path.isfile(fused_npy_path):
        fused = np.load(fused_npy_path)
        if fused.ndim == 3 and frame_idx < fused.shape[0]:
            right_rgb = _inferno_rgb_from_2d(fused[frame_idx])

    if right_rgb is None:
        right_rgb = np.stack([left[:, :, 0]] * 3, axis=-1)

    h = target_height
    left_r = _resize_rgb(left, h)
    center_r = _resize_rgb(center_rgb, h)
    right_r = _resize_rgb(right_rgb, h)

    combined = np.hstack([left_r, center_r, right_r])
    pil = Image.fromarray(combined)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ─── Mel spectrogram + NOMA p(fake) ────────────────────────────────────────


def figure_mel_noma_png_bytes(
    *,
    audio_path: str,
    seconds: np.ndarray,
    p_fake: np.ndarray,
    sr: int | None = None,
) -> bytes:
    """Stack: mel spectrogram (time on x) + line plot of per-block p_fake."""
    _ensure_matplotlib_agg()
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    y, sr_u = librosa.load(audio_path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr_u, n_mels=80)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.25}
    )
    librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sr_u, ax=ax0, cmap="magma")
    ax0.set_title("Mel spectrogram + NOMA p(fake) (calibrated)")
    ax0.set_xlabel("")
    t_max = float(librosa.get_duration(y=y, sr=sr_u))
    ax1.set_xlim(0.0, max(t_max, float(np.max(seconds)) + 0.5 if len(seconds) else t_max))
    ax1.plot(np.asarray(seconds, dtype=float), np.asarray(p_fake, dtype=float), color="#ef4444", lw=2)
    ax1.fill_between(
        np.asarray(seconds, dtype=float),
        np.asarray(p_fake, dtype=float),
        alpha=0.15,
        color="#ef4444",
    )
    ax1.set_ylabel("p(fake)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ─── CMID / cosine similarity vs time ──────────────────────────────────────


def figure_cmid_png_bytes(cmid: dict[str, Any]) -> bytes:
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    sim = cmid.get("similarity") or []
    cmid_vals = cmid.get("cmid") or []
    if not sim and not cmid_vals:
        raise ValueError("cmid dict must contain 'similarity' or 'cmid' lists")

    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    if sim:
        t0 = np.arange(len(sim), dtype=float)
        ax1.plot(t0, np.asarray(sim, dtype=float), color="#2563eb", label="cos(audio, visual)")
        ax1.set_ylabel("similarity")
    if cmid_vals:
        t1 = np.arange(len(cmid_vals), dtype=float)
        if sim:
            ax2 = ax1.twinx()
            ax2.plot(t1, np.asarray(cmid_vals, dtype=float), color="#dc2626", label="CMID")
            ax2.set_ylabel("CMID (higher = less sync)")
            lines1, lab1 = ax1.get_legend_handles_labels()
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
        else:
            ax1.plot(t1, np.asarray(cmid_vals, dtype=float), color="#dc2626", label="CMID")
            ax1.set_ylabel("CMID")
            ax1.legend(loc="upper right")
    elif sim:
        ax1.legend(loc="upper right")
    ax1.set_xlabel("frame index (aligned embeddings)")
    ax1.set_title("Cross-modal sync: cosine similarity and CMID")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ─── Attention vs Grad-CAM intensity ────────────────────────────────────────


def figure_attention_cam_png_bytes(cam_idx: dict[str, Any]) -> bytes:
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    attn = cam_idx.get("attention_per_frame")
    cam = cam_idx.get("cam_per_frame")
    if not isinstance(attn, list) or not isinstance(cam, list):
        raise ValueError("cam_idx needs attention_per_frame and cam_per_frame lists")

    n = min(len(attn), len(cam))
    t = np.arange(n, dtype=float)
    roi_fps = float(cam_idx.get("roi_fps") or 0.0)
    if roi_fps > 1e-6:
        t = t / roi_fps

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, np.asarray(attn[:n], dtype=float), label="attention (norm)", color="#7c3aed")
    ax.plot(t, np.asarray(cam[:n], dtype=float), label="Grad-CAM intensity", color="#059669", alpha=0.85)
    ax.set_xlabel("time (s)" if roi_fps > 1e-6 else "frame")
    ax.set_ylabel("value")
    ax.set_title("Attention trace vs Grad-CAM per-frame intensity")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ─── Calibration: raw → calibrated p(fake) ──────────────────────────────────


def figure_calibration_png_bytes() -> bytes:
    """Two panels: AVH raw score → p(fake); NOMA raw p → calibrated p."""
    _ensure_matplotlib_agg()
    import matplotlib.pyplot as plt

    from config import PROJECT_ROOT

    from calibration_runtime import avh_score_to_p_fake, noma_p_fake_to_calibrated

    art_path = os.path.join(PROJECT_ROOT, "calibration_artifacts.json")
    has_file = os.path.isfile(art_path)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    xs = np.linspace(-4.0, 4.0, 400)
    ys = np.array([avh_score_to_p_fake(float(x)) for x in xs])
    ax0.plot(xs, ys, color="#2563eb", lw=2)
    ax0.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax0.axvline(0.0, color="gray", ls=":", lw=0.6)
    ax0.set_xlabel("AVH raw fusion score")
    ax0.set_ylabel("calibrated p(fake)")
    ax0.set_title("AVH calibration map")
    ax0.grid(True, alpha=0.3)

    pr = np.linspace(0.01, 0.99, 300)
    yc = noma_p_fake_to_calibrated(pr)
    ax1.plot(pr, np.asarray(yc, dtype=float), color="#dc2626", lw=2)
    ax1.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8, label="identity")
    ax1.set_xlabel("NOMA raw p(fake)")
    ax1.set_ylabel("calibrated p(fake)")
    ax1.set_title("NOMA logit calibration")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        "Runtime calibration (from calibration_artifacts.json if present)"
        + (" — file found" if has_file else " — using defaults"),
        fontsize=10,
    )
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
