"""Manual drag-to-crop on the first video frame (Streamlit + streamlit-drawable-canvas)."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import Any

# ── FFmpeg ───────────────────────────────────────────────────────────────


def resolve_ffmpeg_bin() -> str:
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/bin/ffmpeg", "ffmpeg"):
        if cand == "ffmpeg":
            p = shutil.which("ffmpeg")
            if p:
                return p
        elif os.path.isfile(cand):
            return cand
    return "ffmpeg"


def _even(x: int) -> int:
    return max(2, (int(x) // 2) * 2)


def extract_first_frame_png(video_path: str, out_png: str, ffmpeg_bin: str | None = None) -> bool:
    ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg_bin()
    try:
        r = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-ss",
                "0",
                "-i",
                video_path,
                "-frames:v",
                "1",
                out_png,
                "-loglevel",
                "quiet",
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
        return r.returncode == 0 and os.path.isfile(out_png) and os.path.getsize(out_png) > 10
    except Exception:
        return False


def ffmpeg_spatial_crop_video(
    in_path: str,
    out_path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    ffmpeg_bin: str | None = None,
) -> bool:
    """Crop all frames; copy audio from source. w/h/x/y should be even for broad codec support."""
    ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg_bin()
    x, y, w, h = _even(x), _even(y), _even(w), _even(h)
    if w < 32 or h < 32:
        return False
    vf = f"crop={w}:{h}:{x}:{y}"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        in_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        out_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        return r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 100
    except Exception:
        return False


# ── Canvas JSON → pixel rect ─────────────────────────────────────────────


def rect_from_canvas_object(o: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Map Fabric.js rect JSON to integer x, y, w, h in canvas/image pixel space."""
    if not o:
        return None
    if str(o.get("type", "")).lower() != "rect":
        return None
    left = float(o.get("left", 0))
    top = float(o.get("top", 0))
    w = float(o.get("width", 0)) * float(o.get("scaleX", 1) or 1)
    h = float(o.get("height", 0)) * float(o.get("scaleY", 1) or 1)
    if w < 4 or h < 4:
        return None
    return int(round(left)), int(round(top)), int(round(w)), int(round(h))


def pick_last_rect_from_canvas_json(json_data: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    if not json_data or not isinstance(json_data, dict):
        return None
    objects = json_data.get("objects")
    if not isinstance(objects, list):
        return None
    last: tuple[int, int, int, int] | None = None
    for o in objects:
        if not isinstance(o, dict):
            continue
        r = rect_from_canvas_object(o)
        if r is not None:
            last = r
    return last


def map_display_rect_to_original(
    x: int,
    y: int,
    w: int,
    h: int,
    orig_w: int,
    orig_h: int,
    disp_w: int,
    disp_h: int,
) -> tuple[int, int, int, int]:
    sx = orig_w / max(1, disp_w)
    sy = orig_h / max(1, disp_h)
    x0 = int(round(x * sx))
    y0 = int(round(y * sy))
    w0 = int(round(w * sx))
    h0 = int(round(h * sy))
    x0 = max(0, min(x0, orig_w - 1))
    y0 = max(0, min(y0, orig_h - 1))
    w0 = max(32, min(w0, orig_w - x0))
    h0 = max(32, min(h0, orig_h - y0))
    return x0, y0, w0, h0


def upload_signature(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


# ── Streamlit widget ───────────────────────────────────────────────────────


def render_manual_crop_ui(
    *,
    state_prefix: str,
    file_bytes: bytes | None,
    filename: str | None,
    max_display_width: int = 720,
) -> None:
    """
    Renders optional manual crop UI; stores under session_state:
      {prefix}_manual_rect — (x,y,w,h) in original video pixels
      {prefix}_manual_sig — upload signature when rect was saved
    """
    import streamlit as st

    try:
        from PIL import Image
    except ImportError:
        st.warning("Pillow is required for manual crop. `pip install Pillow`")
        return

    try:
        # streamlit-drawable-canvas expects streamlit.elements.image.image_to_url (removed in newer Streamlit).
        import streamlit.elements.image as _st_image_mod

        if not hasattr(_st_image_mod, "image_to_url"):
            from streamlit.elements.lib.image_utils import image_to_url as _image_to_url_impl
            from streamlit.elements.lib.layout_utils import LayoutConfig

            def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
                return _image_to_url_impl(
                    image,
                    LayoutConfig(width=width),
                    clamp,
                    channels,
                    output_format,
                    image_id,
                )

            _st_image_mod.image_to_url = _image_to_url_compat  # type: ignore[attr-defined]

        from streamlit_drawable_canvas import st_canvas
    except ImportError:
        st.info(
            "Install **streamlit-drawable-canvas** for drag-to-crop: `pip install streamlit-drawable-canvas`"
        )
        return

    if not file_bytes or not filename:
        return

    sig = upload_signature(file_bytes)
    enabled = st.checkbox(
        "Manual crop: draw a rectangle on the first frame",
        value=False,
        key=f"{state_prefix}_manual_enable",
        help="Focuses AVH on the speaker region; ignores automatic spatial pre-crop when a crop is saved.",
    )
    if not enabled:
        return

    st.caption(
        "Draw a **rectangle** around the face / talking-head region. Click **Save crop region**, then run the pipeline."
    )

    suffix = os.path.splitext(filename)[-1] or ".mp4"
    fd, vid_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(fd)
        with open(vid_path, "wb") as f:
            f.write(file_bytes)
        fd2, png_path = tempfile.mkstemp(suffix=".png")
        os.close(fd2)
        try:
            if not extract_first_frame_png(vid_path, png_path):
                st.warning("Could not extract the first frame (ffmpeg missing?).")
                return
            pil_full = Image.open(png_path).convert("RGB")
        finally:
            try:
                os.unlink(png_path)
            except Exception:
                pass
    finally:
        try:
            os.unlink(vid_path)
        except Exception:
            pass

    orig_w, orig_h = pil_full.size
    scale = min(1.0, max_display_width / float(orig_w))
    disp_w = max(1, int(round(orig_w * scale)))
    disp_h = max(1, int(round(orig_h * scale)))
    img_disp = pil_full.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    canvas_result = st_canvas(
        fill_color="rgba(255, 80, 80, 0.12)",
        stroke_width=2,
        stroke_color="#dc2626",
        background_image=img_disp,
        update_streamlit=True,
        height=disp_h,
        width=disp_w,
        drawing_mode="rect",
        key=f"{state_prefix}_manual_canvas",
        display_toolbar=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save crop region", key=f"{state_prefix}_save_manual_crop"):
            jd = canvas_result.json_data if canvas_result else None
            disp_rect = pick_last_rect_from_canvas_json(jd)
            if disp_rect is None:
                st.warning("Draw a rectangle on the image first.")
            else:
                dx, dy, dw, dh = disp_rect
                ox0, oy0, ow0, oh0 = map_display_rect_to_original(
                    dx, dy, dw, dh, orig_w, orig_h, disp_w, disp_h
                )
                st.session_state[f"{state_prefix}_manual_rect"] = (ox0, oy0, ow0, oh0)
                st.session_state[f"{state_prefix}_manual_sig"] = sig
                st.success(f"Saved region: {ow0}×{oh0} px at ({ox0}, {oy0}).")
    with col_b:
        if st.button("Clear saved crop", key=f"{state_prefix}_clear_manual_crop"):
            st.session_state.pop(f"{state_prefix}_manual_rect", None)
            st.session_state.pop(f"{state_prefix}_manual_sig", None)
            st.rerun()

    saved_sig = st.session_state.get(f"{state_prefix}_manual_sig")
    rect = st.session_state.get(f"{state_prefix}_manual_rect")
    if saved_sig == sig and isinstance(rect, tuple) and len(rect) == 4:
        x, y, w, h = rect
        st.caption(f"Using saved crop: **{w}×{h}** px at ({x}, {y}) on a **{orig_w}×{orig_h}** frame.")
    elif enabled:
        st.caption("No saved crop yet for this file — automatic spatial pre-crop still applies if enabled.")


def get_saved_manual_rect(state_prefix: str, file_bytes: bytes | None) -> tuple[int, int, int, int] | None:
    """Return saved rect if it matches current upload bytes."""
    import streamlit as st

    if not file_bytes:
        return None
    sig = upload_signature(file_bytes)
    if st.session_state.get(f"{state_prefix}_manual_sig") != sig:
        return None
    r = st.session_state.get(f"{state_prefix}_manual_rect")
    if isinstance(r, (list, tuple)) and len(r) == 4:
        return int(r[0]), int(r[1]), int(r[2]), int(r[3])
    return None


def prepare_video_with_optional_manual_crop(
    source_path: str,
    filename: str,
    manual_rect: tuple[int, int, int, int] | None,
) -> tuple[str, str, bool]:
    """
    If manual_rect is set, write a cropped copy and return it.

    Returns:
        (path_for_avh, path_to_delete_after_run_or_empty, use_manual)
        If use_manual, caller should pass smart_crop \"off\" to AVH.
    """
    if not manual_rect:
        return source_path, "", False
    suffix = os.path.splitext(filename)[-1] or ".mp4"
    out_path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name
    x, y, w, h = manual_rect
    if ffmpeg_spatial_crop_video(source_path, out_path, x, y, w, h):
        return out_path, out_path, True
    return source_path, "", False
