import json
import hashlib
import os
import tempfile
from functools import lru_cache
from typing import Any

from config import (
    PROJECT_ROOT,
    AVH_DIR,
    AVH_GRADCAM_SCRIPT,
    AVH_FUSION_CKPT,
    AVH_AVHUBERT_CKPT,
    GRADCAM_DEFAULT_MAX_FUSION_FRAMES,
    GRADCAM_DEFAULT_REGION_TRACK_STRIDE,
)
from logging_utils import get_logger, log_timed
from metrics import inc_counter

from detectors.avh_ckpt_paths import get_readable_ckpt_path


@lru_cache(maxsize=16)
def _sha256_file_cached(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_cam_volume(cam_like: Any) -> tuple[Any | None, str | None]:
    import numpy as np

    cam = np.asarray(cam_like)
    cam = np.squeeze(cam)
    if cam.ndim != 3:
        return None, f"bad_cam_shape:{tuple(cam.shape)}"
    return cam, None


def resize_frames_to_cam(frames_gray: Any, target_h: int, target_w: int):
    import cv2 as _cv2
    import numpy as np

    T = int(frames_gray.shape[0])
    out = np.zeros((T, target_h, target_w), dtype=frames_gray.dtype)
    for t in range(T):
        out[t] = _cv2.resize(frames_gray[t], (target_w, target_h), interpolation=_cv2.INTER_AREA)
    return out


def compute_windowed_fusion(cam: Any, frames_gray: Any, window_size: int):
    import numpy as np
    from explainability.video_fusion import (
        compute_optical_flow_error,
        compute_frequency_noise_map,
        generate_fused_heatmap,
    )

    T_use = min(int(cam.shape[0]), int(frames_gray.shape[0]))
    cam_use = cam[:T_use]
    frames_use = frames_gray[:T_use]

    if T_use <= 0:
        raise ValueError("empty inputs after alignment")

    Hc, Wc = int(cam_use.shape[1]), int(cam_use.shape[2])
    frames_resized = resize_frames_to_cam(frames_use, Hc, Wc)

    fused_chunks = []
    hfe_vals = []
    for start in range(0, T_use, window_size):
        end = min(start + window_size, T_use)
        cam_w = cam_use[start:end].astype(float)
        frm_w = frames_resized[start:end]
        flow_err = compute_optical_flow_error(frm_w)
        freq_noise = compute_frequency_noise_map(frm_w)
        fused_w = generate_fused_heatmap(cam_w, flow_err[: cam_w.shape[0]], freq_noise[: cam_w.shape[0]])
        fused_chunks.append(fused_w)

        # Frequency summary without importing scipy-heavy module in callers.
        high_freq = freq_noise.mean(axis=(1, 2))
        hfe_vals.extend([float(v) for v in high_freq])

    fused = np.concatenate(fused_chunks, axis=0)
    return fused, {"high_freq_energy": hfe_vals}


def run_gradcam_mouth_roi(
    video_path: str,
    python_exe: str,
    top_k: int = 2,
    adv_ckpt: str | None = None,
    roi_path: str | None = None,
    audio_path: str | None = None,
    max_fusion_frames: int = GRADCAM_DEFAULT_MAX_FUSION_FRAMES,
    region_track_stride: int = GRADCAM_DEFAULT_REGION_TRACK_STRIDE,
    selection_mode: str = "top_k",
    min_temporal_gap: int = 24,
    timeout: int = 300,
    keep_temp: bool = False,
    capture_attention: bool = False,
) -> tuple[bool, Any, Any]:
    """
    Runs AVH/gradcam_mouth_roi.py and returns:
      - (ok, overlays_dir_or_error, index_json_or_none)
    """
    logger = get_logger("explainability.gradcam_avh")

    if not os.path.exists(AVH_GRADCAM_SCRIPT):
        return (False, "Grad-CAM script not found. Expected gradcam_mouth_roi.py under AVH/.", None)
    if not (os.path.exists(AVH_FUSION_CKPT) and os.path.exists(AVH_AVHUBERT_CKPT)):
        return (False, "AVH checkpoints missing for Grad-CAM.", None)

    if adv_ckpt:
        if not os.path.isfile(adv_ckpt):
            return (False, f"Adversary checkpoint not found: {adv_ckpt}", None)

    from subprocess_utils import run_subprocess_capture, validate_python_exe

    try:
        py = validate_python_exe(python_exe)
    except Exception as e:
        return (False, str(e), None)

    cache_root = os.path.join(PROJECT_ROOT, ".cache", "gradcam")
    os.makedirs(cache_root, exist_ok=True)

    # Cache key: input media + relevant checkpoint hashes + runtime params.
    # We hash the inputs to make the cache stable across temp path changes.
    video_hash = _sha256_file_cached(video_path)
    fusion_hash = _sha256_file_cached(AVH_FUSION_CKPT)
    avhubert_hash = _sha256_file_cached(AVH_AVHUBERT_CKPT)
    roi_hash = _sha256_file_cached(roi_path) if roi_path and os.path.isfile(roi_path) else "none"
    audio_hash = _sha256_file_cached(audio_path) if audio_path and os.path.isfile(audio_path) else "none"
    adv_hash = _sha256_file_cached(adv_ckpt) if adv_ckpt and os.path.isfile(adv_ckpt) else "none"

    cache_key = (
        f"video={video_hash}|fusion={fusion_hash}|avhubert={avhubert_hash}|roi={roi_hash}|audio={audio_hash}|"
        f"adv={adv_hash}|topk={int(top_k)}|cap_attn={bool(capture_attention)}|"
        f"max_fusion={int(max_fusion_frames)}|track_stride={int(region_track_stride)}|"
        f"sel_mode={selection_mode}|min_gap={int(min_temporal_gap)}"
    )
    cache_key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    cache_dir = os.path.join(cache_root, cache_key_hash)

    index_cache_path = os.path.join(cache_dir, "index.json")
    overlays_cache_dir = os.path.join(cache_dir, "overlays")

    def _restore_from_cache() -> tuple[bool, Any, Any]:
        if not (os.path.isfile(index_cache_path) and os.path.isdir(overlays_cache_dir)):
            return (False, None, None)

        out_dir = tempfile.mkdtemp(prefix="gradcam_out_", dir=tempfile.gettempdir())
        try:
            import shutil

            overlays_dir = os.path.join(out_dir, "overlays")
            shutil.copytree(overlays_cache_dir, overlays_dir)

            restored_index_path = os.path.join(out_dir, "index.json")
            shutil.copy2(index_cache_path, restored_index_path)
            with open(restored_index_path, "r", encoding="utf-8") as f:
                idx = json.load(f)

            # Ensure overlay_dir points to the restored directory.
            idx["overlay_dir"] = overlays_dir
            return (True, overlays_dir, idx)
        except Exception:
            # If cache restore fails, fall back to recompute.
            try:
                import shutil

                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            return (False, None, None)

    restored_ok, restored_overlays_dir, restored_idx = _restore_from_cache()
    if restored_ok:
        inc_counter("gradcam_cache_hit", stage="gradcam")
        return (True, restored_overlays_dir, restored_idx)

    out_dir = tempfile.mkdtemp(prefix="gradcam_out_", dir=tempfile.gettempdir())
    try:
        import shutil

        fusion_path = get_readable_ckpt_path(AVH_FUSION_CKPT, force_tmp=True)
        avhubert_path = get_readable_ckpt_path(AVH_AVHUBERT_CKPT, "self_large_vox_433h.pt", force_tmp=True)

        cmd = [
            py,
            "gradcam_mouth_roi.py",
            "--video_path",
            os.path.abspath(video_path),
            "--out_dir",
            out_dir,
            "--top_k",
            str(int(top_k)),
            "--selection_mode",
            str(selection_mode),
            "--min_temporal_gap",
            str(int(min_temporal_gap)),
            "--max_fusion_frames",
            str(int(max_fusion_frames)),
            "--overwrite",
            "--device",
            "cpu",
            "--fusion_ckpt",
            fusion_path,
            "--avhubert_ckpt",
            avhubert_path,
        ]

        if roi_path and audio_path:
            cmd += ["--roi_path", os.path.abspath(roi_path), "--audio_path", os.path.abspath(audio_path)]
        if adv_ckpt:
            adv_path = get_readable_ckpt_path(adv_ckpt, tmp_name="feature_adversary_latest.pt", force_tmp=True)
            cmd += ["--adv_ckpt", adv_path]
        if keep_temp:
            cmd.append("--keep_temp")
        if capture_attention:
            cmd.append("--capture_attention")

        with log_timed(logger, "gradcam_subprocess", cache_hit=False):
            run_res = run_subprocess_capture(cmd, cwd=AVH_DIR, timeout_s=timeout)
        out = (run_res.get("stdout") or "") + (run_res.get("stderr") or "")
        timed_out = bool(run_res.get("timed_out"))
        returncode = run_res.get("returncode")

        if timed_out or returncode != 0:
            return (False, out if out else "Grad-CAM timed out or failed.", None)

        index_path = os.path.join(out_dir, "index.json")
        if not os.path.isfile(index_path):
            return (False, "Grad-CAM finished but index.json was not created.", None)

        with open(index_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        idx.setdefault("xai_status", {})
        idx.setdefault("max_fusion_frames", int(max_fusion_frames))
        idx.setdefault("region_track_stride", int(region_track_stride))
        idx.setdefault("selection_mode", str(selection_mode))
        idx.setdefault("min_temporal_gap", int(min_temporal_gap))

        # Optional: compute temporal inconsistency, region tracks, and fused heatmaps
        # if CAM volume and video are available.
        cam_volume_path = idx.get("cam_volume_path")
        video_src = idx.get("video_path") or idx.get("input_video")
        idx.setdefault("xai_errors", {})
        if cam_volume_path and os.path.isfile(cam_volume_path):
            import numpy as np
            from explainability.video_temporal import compute_temporal_inconsistency
            from explainability.video_regions import cam_to_binary_masks, track_regions_iou, summarize_region_anomalies
            from explainability.video_fusion import prepare_gray_frames_from_video

            cam_raw = np.load(cam_volume_path)
            cam, cam_shape_err = normalize_cam_volume(cam_raw)
            if cam is None:
                idx["xai_status"]["temporal_inconsistency"] = "skipped:bad_cam_shape"
                idx["xai_status"]["region_tracks"] = "skipped:bad_cam_shape"
                idx["xai_status"]["fusion"] = "skipped:bad_cam_shape"
                idx["xai_status"]["video_frequency_stats"] = "skipped:bad_cam_shape"
                idx["xai_errors"]["cam_shape"] = cam_shape_err
            else:
                # Persist normalized shape to avoid repeated failures on cache hits.
                try:
                    import numpy as np

                    np.save(cam_volume_path, cam)
                except Exception:
                    pass

                # Temporal inconsistency
                try:
                    flat = cam.reshape(cam.shape[0], -1)
                    delta_t = compute_temporal_inconsistency(flat)
                    idx["temporal_inconsistency"] = delta_t.tolist()
                    idx["xai_status"]["temporal_inconsistency"] = "computed"
                except Exception as e:
                    idx["xai_status"]["temporal_inconsistency"] = "failed"
                    idx["xai_errors"]["temporal_inconsistency"] = str(e)
                    logger.exception("temporal inconsistency enrichment failed")

                # Region tracks
                try:
                    stride = max(1, int(region_track_stride))
                    cam_tracks = cam[::stride] if stride > 1 else cam
                    masks = cam_to_binary_masks(cam_tracks)
                    tracks = track_regions_iou(masks, cam_tracks)
                    summary = summarize_region_anomalies(tracks)
                    summary["track_stride"] = stride
                    idx["region_tracks"] = summary
                    idx["xai_status"]["region_tracks"] = "computed"
                except Exception as e:
                    idx["xai_status"]["region_tracks"] = "failed"
                    idx["xai_errors"]["region_tracks"] = str(e)
                    logger.exception("region tracking enrichment failed")

                # Multi-signal fusion and frequency stats if we have the original video.
                if video_src and os.path.isfile(video_src):
                    try:
                        frames_gray, _ = prepare_gray_frames_from_video(video_src)
                        T_cam = cam.shape[0]
                        T_vid = frames_gray.shape[0]
                        T_use = min(T_cam, T_vid)
                        max_frames = max(1, int(idx.get("max_fusion_frames", max_fusion_frames)))
                        fused, freq_stats = compute_windowed_fusion(cam[:T_use], frames_gray[:T_use], max_frames)
                        idx["fused_heatmap_path"] = os.path.join(os.path.dirname(cam_volume_path), "fused_heatmap.npy")
                        np.save(idx["fused_heatmap_path"], fused)
                        idx["xai_status"]["fusion"] = "computed:windowed" if T_use > max_frames else "computed"
                        idx["video_frequency_stats"] = freq_stats
                        idx["xai_status"]["video_frequency_stats"] = "computed:windowed" if T_use > max_frames else "computed"
                        idx["fusion_window_frames"] = max_frames
                        idx["fusion_total_frames"] = T_use
                    except Exception as e:
                        idx["xai_status"]["fusion"] = "failed"
                        idx["xai_status"]["video_frequency_stats"] = "failed"
                        idx["xai_errors"]["fusion"] = str(e)
                        idx["xai_errors"]["video_frequency_stats"] = str(e)
                        logger.exception("fusion/frequency enrichment failed")
                else:
                    idx["xai_status"]["fusion"] = "skipped:no_video_src"
                    idx["xai_status"]["video_frequency_stats"] = "skipped:no_video_src"
        else:
            idx["xai_status"]["temporal_inconsistency"] = "skipped:no_cam_volume"
            idx["xai_status"]["region_tracks"] = "skipped:no_cam_volume"
            idx["xai_status"]["fusion"] = "skipped:no_cam_volume"
            idx["xai_status"]["video_frequency_stats"] = "skipped:no_cam_volume"

        overlays_dir = idx.get("overlay_dir", os.path.join(out_dir, "overlays"))
        # Best-effort cache write.
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Write enriched index back before caching.
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(idx, f, indent=2)
            shutil.copy2(index_path, os.path.join(cache_dir, "index.json"))
            # Copy overlay images.
            if os.path.isdir(overlays_dir):
                if os.path.isdir(overlays_cache_dir):
                    shutil.rmtree(overlays_cache_dir, ignore_errors=True)
                shutil.copytree(overlays_dir, overlays_cache_dir)
        except Exception:
            pass

        return (True, overlays_dir, idx)
    finally:
        if not keep_temp:
            # overlays were rendered via st.image; safe to delete after returning.
            import shutil

            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass

