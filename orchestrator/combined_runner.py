import os
import shutil
import tempfile
from typing import Any

from logging_utils import get_logger

_log = get_logger("orchestrator.combined_runner")


def _safe_temp_roots_for_cleanup(paths: list[str]) -> list[str]:
    """Return absolute paths under the system temp dir only (never / or $HOME)."""
    tmp = os.path.abspath(tempfile.gettempdir())
    out: list[str] = []
    for p in paths:
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap == tmp:
            continue
        if ap.startswith(tmp + os.sep):
            out.append(ap)
    # Deepest paths first so nested temps are removed before parents.
    return sorted(set(out), key=lambda s: len(s), reverse=True)


def _persist_combined_artifacts(
    result: dict[str, Any],
    *,
    persist_run_dir: str,
    cleanup_volatile_after_persist: bool,
) -> None:
    """
    Copy audio, ROI, Grad-CAM overlays, cam_volume / fused heatmap, embeddings into persist_run_dir
    and rewrite result paths (including cam_idx) so other pages still work after temp cleanup.
    Optionally remove original volatile temp dirs (only under tempfile.gettempdir()).
    """
    audio_path = result.get("audio_path")
    if not audio_path or not os.path.isfile(audio_path):
        return

    orig_audio_dir = os.path.dirname(os.path.abspath(audio_path))
    os.makedirs(persist_run_dir, exist_ok=True)
    volatile_roots: list[str] = []

    audio_root = os.path.dirname(os.path.abspath(audio_path))
    volatile_roots.append(audio_root)

    new_audio = os.path.join(persist_run_dir, os.path.basename(audio_path))
    shutil.copy2(audio_path, new_audio)
    result["audio_path"] = new_audio

    roi_path = result.get("roi_path")
    if roi_path and os.path.isfile(roi_path):
        new_roi = os.path.join(persist_run_dir, "mouth_roi.mp4")
        shutil.copy2(roi_path, new_roi)
        result["roi_path"] = new_roi

    overlays = result.get("cam_overlays_dir")
    old_overlay_dir = overlays
    if overlays and os.path.isdir(overlays):
        new_overlays = os.path.join(persist_run_dir, "gradcam_overlays")
        if os.path.isdir(new_overlays):
            shutil.rmtree(new_overlays, ignore_errors=True)
        shutil.copytree(overlays, new_overlays)
        result["cam_overlays_dir"] = new_overlays
        cam_parent = result.get("cam_parent_dir")
        if cam_parent and os.path.abspath(cam_parent) != audio_root:
            volatile_roots.append(cam_parent)
        result["cam_parent_dir"] = os.path.dirname(new_overlays)

        ci = result.get("cam_idx")
        if isinstance(ci, dict):
            ci["overlay_dir"] = new_overlays
            cv = ci.get("cam_volume_path")
            if cv and os.path.isfile(cv):
                dest_cv = os.path.join(persist_run_dir, "cam_volume.npy")
                shutil.copy2(cv, dest_cv)
                ci["cam_volume_path"] = dest_cv
            fh = ci.get("fused_heatmap_path")
            if fh and os.path.isfile(fh):
                dest_fh = os.path.join(persist_run_dir, "fused_heatmap.npy")
                shutil.copy2(fh, dest_fh)
                ci["fused_heatmap_path"] = dest_fh
        idx_side = os.path.join(os.path.dirname(os.path.abspath(old_overlay_dir)), "index.json")
        if os.path.isfile(idx_side):
            shutil.copy2(idx_side, os.path.join(persist_run_dir, "gradcam_index.json"))

    emb_src = os.path.join(orig_audio_dir, "avh_embeddings.npz")
    if os.path.isfile(emb_src):
        shutil.copy2(emb_src, os.path.join(persist_run_dir, "avh_embeddings.npz"))

    result["persist_run_dir"] = persist_run_dir
    result["volatile_cleanup_roots"] = list(dict.fromkeys(volatile_roots))

    if not cleanup_volatile_after_persist:
        return

    for root in _safe_temp_roots_for_cleanup(volatile_roots):
        try:
            shutil.rmtree(root, ignore_errors=True)
        except Exception:
            pass


def run_combined_avh_to_noma(
    *,
    video_path: str,
    video_name: str,
    use_unsup_avh: bool,
    python_exe: str | None,
    run_forensics_cam: bool,
    forensics_top_k: int,
    forensics_selection_mode: str,
    forensics_min_temporal_gap: int,
    forensics_max_fusion_frames: int,
    region_track_stride: int,
    run_robustness_delta: bool,
    adv_ckpt_path: str,
    capture_attention: bool,
    export_bundle: bool,
    noma_model_path: str,
    timeout: int = 900,
    persist_run_dir: str | None = None,
    cleanup_volatile_after_persist: bool = False,
    dump_embeddings_for_cmid: bool = False,
    noma_permutation_max_blocks: int | None = None,
    smart_crop: str = "auto",
    late_fusion_mode: str | None = None,
) -> dict[str, Any]:
    """
    Multi-stage pipeline:
      AVH (preprocess -> feature extraction -> score) -> optional Grad-CAM evidence
      -> extract audio from AVH temp work dir -> NOMA per-block predictions
      -> optional evidence bundle export (.zip bytes)

    Returns a dict for Streamlit rendering (raw objects + errors).

    late_fusion_mode: if set to a valid mode (see config.LATE_FUSION_MODES), overrides
    LATE_FUSION_MODE for this run only. Otherwise uses get_late_fusion_mode() (env / default full).
    """
    import hashlib
    import json

    from detectors.avh_align import run_avh_on_video, run_avh_unsupervised_on_video
    from detectors.noma import (
        _load_noma_pipeline,
        noma_p_fake_raw_confidence_and_preds_from_probas,
        run_noma_prediction,
    )
    from explainability.gradcam_avh import run_gradcam_mouth_roi
    from evidence.exporter import zip_evidence_bundle
    from config import PROJECT_ROOT, AVH_FUSION_CKPT, AVH_AVHUBERT_CKPT
    result: dict[str, Any] = {
        "avh_ok": False,
        "avh_score": None,
        "avh_error": None,
        "audio_path": None,
        "cam_ok": False,
        "cam_overlays_dir": None,
        "cam_idx": None,
        "roi_path": None,
        "noma_df": None,
        "bundle_bytes": None,
        "bundle_error": None,
        # For UI cleanup: Grad-CAM outputs are created under a temp dir.
        "cam_parent_dir": None,
        "cam_error": None,
        "cmid": None,
        "cmid_status": "not_computed",
        "tension_index": None,
        "temporal_corroboration": None,
        "noma_permutation_xai": None,
        # Reliability-weighted fusion (calibrated late fusion).
        "p_audio_mean": None,
        "p_audio_mean_raw": None,
        "p_avh_cal": None,
        "fusion_tension": None,
        "fusion_w_audio": None,
        "p_fused": None,
        "fusion_tau": None,
        "fusion_verdict": None,
        "use_unsup_avh": bool(use_unsup_avh),
    }

    work_dir = None
    # Step 1 cache: AVH stage artifacts (score + extracted audio + mouth ROI).
    def _sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    cache_root = os.path.join(PROJECT_ROOT, ".cache", "avh_stage")
    os.makedirs(cache_root, exist_ok=True)

    avh_ret = None
    cache_dir: str | None = None
    meta_path: str | None = None
    if not result["avh_ok"] and not dump_embeddings_for_cmid:
        try:
            video_hash = _sha256_file(video_path)
            if use_unsup_avh:
                fusion_hash = "none"
            elif os.path.isfile(AVH_FUSION_CKPT):
                fusion_hash = _sha256_file(AVH_FUSION_CKPT)
            else:
                fusion_hash = "missing"
            avhubert_hash = _sha256_file(AVH_AVHUBERT_CKPT)
            py_str = (python_exe or "").strip()

            sc = (smart_crop or "auto").strip().lower()
            cache_key_str = (
                f"video={video_hash}|unsup={bool(use_unsup_avh)}|py={py_str}|fusion={fusion_hash}|"
                f"avhubert={avhubert_hash}|smart_crop={sc}"
            )
            cache_key_hash = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()
            cache_dir = os.path.join(cache_root, cache_key_hash)
            meta_path = os.path.join(cache_dir, "meta.json")

            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                cached_audio_basename = meta.get("audio_basename")
                cached_audio_path = os.path.join(cache_dir, "audio", cached_audio_basename)
                cached_roi_path = os.path.join(cache_dir, "mouth_roi.mp4")

                if (
                    cached_audio_basename
                    and os.path.isfile(cached_audio_path)
                    and os.path.isfile(cached_roi_path)
                    and meta.get("score") is not None
                ):
                    work_dir = tempfile.mkdtemp(prefix="avh_restore_", dir=tempfile.gettempdir())
                    restored_audio_path = os.path.join(work_dir, cached_audio_basename)
                    restored_roi_path = os.path.join(work_dir, "mouth_roi.mp4")
                    shutil.copy2(cached_audio_path, restored_audio_path)
                    shutil.copy2(cached_roi_path, restored_roi_path)

                    result["avh_ok"] = True
                    result["avh_score"] = float(meta["score"])
                    result["audio_path"] = restored_audio_path
        except Exception:
            cache_dir = None
            meta_path = None
            work_dir = None

    # Step 1: run AVH (cache miss path)
    if not result["avh_ok"]:
        if use_unsup_avh:
            avh_ret = run_avh_unsupervised_on_video(
                video_path,
                timeout=timeout,
                python_exe=python_exe,
                keep_temp=True,
                smart_crop=smart_crop,
            )
        else:
            avh_ret = run_avh_on_video(
                video_path,
                timeout=timeout,
                python_exe=python_exe,
                keep_temp=True,
                dump_embeddings=bool(dump_embeddings_for_cmid),
                smart_crop=smart_crop,
            )

        if not avh_ret or avh_ret[0] is False:
            result["avh_ok"] = False
            result["avh_error"] = avh_ret[1] if avh_ret else "AVH failed"
            return result

        # keep_temp=True => (True, score, audio_path)
        _, avh_score, audio_path = avh_ret
        result["avh_ok"] = True
        result["avh_score"] = float(avh_score)
        result["audio_path"] = audio_path

        if not audio_path:
            return result

        # Cache AVH stage artifacts (best-effort).
        try:
            cached_roi_path = os.path.join(os.path.dirname(audio_path), "mouth_roi.mp4")
            if os.path.isfile(cached_roi_path) and cache_dir and meta_path:
                os.makedirs(os.path.join(cache_dir, "audio"), exist_ok=True)
                cached_audio_basename = os.path.basename(audio_path)
                shutil.copy2(audio_path, os.path.join(cache_dir, "audio", cached_audio_basename))
                shutil.copy2(cached_roi_path, os.path.join(cache_dir, "mouth_roi.mp4"))
                tmp_meta_path = meta_path + ".tmp"
                with open(tmp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"score": float(avh_score), "audio_basename": cached_audio_basename},
                        f,
                        indent=2,
                    )
                os.replace(tmp_meta_path, meta_path)
        except Exception:
            pass

    audio_path = result["audio_path"]
    if not audio_path:
        return result

    # Step 2: optional Grad-CAM evidence
    work_dir = os.path.dirname(audio_path)
    result["roi_path"] = os.path.join(work_dir, "mouth_roi.mp4")

    if run_forensics_cam:
        adv_for_cam = adv_ckpt_path.strip() or None if run_robustness_delta else None

        ok_cam, overlays_dir_or_err, cam_idx_or_none = run_gradcam_mouth_roi(
            video_path=video_path,
            python_exe=python_exe,
            top_k=forensics_top_k,
            selection_mode=forensics_selection_mode,
            min_temporal_gap=forensics_min_temporal_gap,
            max_fusion_frames=forensics_max_fusion_frames,
            region_track_stride=region_track_stride,
            adv_ckpt=adv_for_cam,
            roi_path=result["roi_path"],
            audio_path=audio_path,
            timeout=timeout,
            keep_temp=True,
            capture_attention=capture_attention,
        )

        result["cam_ok"] = bool(ok_cam)
        if ok_cam:
            result["cam_overlays_dir"] = overlays_dir_or_err
            result["cam_idx"] = cam_idx_or_none
            # overlays_dir is .../overlays; parent is the temp root for cleanup after UI.
            result["cam_parent_dir"] = os.path.dirname(result["cam_overlays_dir"])
            result["cam_error"] = None
        else:
            result["cam_overlays_dir"] = None
            result["cam_idx"] = None
            result["cam_error"] = (
                str(overlays_dir_or_err) if isinstance(overlays_dir_or_err, str) else repr(overlays_dir_or_err)
            )

    # Step 3: NOMA on extracted audio
    import pandas as pd

    times, probas = run_noma_prediction(noma_model_path, audio_path=audio_path)
    noma_pipe = _load_noma_pipeline(noma_model_path)
    p_fake_raw, confidences, preds_str = noma_p_fake_raw_confidence_and_preds_from_probas(
        noma_pipe,
        probas,
    )
    from calibration_runtime import noma_p_fake_to_calibrated
    p_fake = noma_p_fake_to_calibrated(p_fake_raw)
    p_real = 1.0 - p_fake
    result["p_audio_mean_raw"] = float(pd.Series(p_fake_raw.astype(float)).mean())

    noma_df = pd.DataFrame(
        {
            "Seconds": times,
            "Prediction": preds_str,
            "Confidence": confidences,
            "p_fake": p_fake,
            "p_real": p_real,
        }
    )
    result["noma_df"] = noma_df

    # Confidence Instability Index (CII) on NOMA p(fake).
    try:
        from explainability.instability import confidence_instability

        inst = confidence_instability(p_fake)
        result["noma_confidence_instability"] = inst
    except Exception:
        result["noma_confidence_instability"] = None

    # CMID: load optional AVH embedding dump (avh_embeddings.npz) or legacy ndarray hooks.
    try:
        from explainability.cross_modal import compute_cross_modal_sync

        emb_audio = result.get("avh_audio_embeddings")
        emb_visual = result.get("avh_visual_embeddings")
        if emb_audio is None or emb_visual is None:
            emb_npz = os.path.join(os.path.dirname(audio_path), "avh_embeddings.npz")
            if os.path.isfile(emb_npz):
                import numpy as np

                z = np.load(emb_npz)
                emb_audio, emb_visual = z["audio"], z["visual"]
        if emb_audio is not None and emb_visual is not None:
            result["cmid"] = compute_cross_modal_sync(emb_audio, emb_visual)
            result["cmid_status"] = "computed"
        else:
            result["cmid_status"] = "missing_embeddings"
    except Exception:
        result["cmid_status"] = "failed"

    # Late multimodal diagnostics: disagreement + time-aligned corroboration (no extra models).
    try:
        from calibration_runtime import avh_score_to_calibrated_p_fake, get_uncertainty_margins
        from config import LATE_FUSION_MODES, get_late_fusion_mode
        from explainability.reliability_fusion import (
            compute_reliability_fusion,
            compute_simple_late_fusion,
        )
        from explainability.learned_reliability_fusion import (
            apply_verdict_three_way,
            compute_learned_reliability_fusion,
            get_learned_fusion_hyperparameters,
        )
        from explainability.temporal_corroboration import (
            compute_temporal_corroboration,
            compute_tension_index,
        )

        if result.get("avh_ok") and result.get("avh_score") is not None:
            p_avh = avh_score_to_calibrated_p_fake(
                float(result["avh_score"]),
                use_unsup_avh=bool(result.get("use_unsup_avh")),
            )
            p_noma_mean = float(pd.Series(p_fake).mean())
            result["tension_index"] = compute_tension_index(p_avh, p_noma_mean)

            avh_m, noma_m = get_uncertainty_margins()
            tau_f = max(float(avh_m), float(noma_m))
            _override = (late_fusion_mode or "").strip().lower()
            late_mode = (
                _override
                if _override in LATE_FUSION_MODES
                else get_late_fusion_mode()
            )
            result["late_fusion_mode"] = late_mode
            if late_mode == "learned":
                hp = get_learned_fusion_hyperparameters()
                lip = abs(float(p_noma_mean) - float(p_avh))
                temporal = float(pd.Series(p_fake).astype(float).std(ddof=0))
                fusion_out = compute_learned_reliability_fusion(
                    float(p_avh),
                    float(p_noma_mean),
                    lip,
                    temporal,
                    alpha=float(hp["learned_fusion_alpha"]),
                    beta=float(hp["learned_fusion_beta"]),
                    tau=float(hp["learned_fusion_tau"]),
                    epsilon=float(hp["learned_fusion_epsilon"]),
                )
                fusion_out["fusion_verdict"] = apply_verdict_three_way(
                    float(fusion_out["p_fused"]),
                    tau_margin=tau_f,
                )
                result.update(fusion_out)
            elif late_mode == "full":
                fusion_out = compute_reliability_fusion(p_noma_mean, p_avh, tau_f)
                result.update(fusion_out)
            else:
                fusion_out = compute_simple_late_fusion(late_mode, p_noma_mean, p_avh, tau_f)
                result.update(fusion_out)
        cam_for_corr = result.get("cam_idx") if run_forensics_cam else None
        result["temporal_corroboration"] = compute_temporal_corroboration(
            noma_seconds=noma_df["Seconds"].values,
            p_fake_calibrated=p_fake,
            cam_idx=cam_for_corr if isinstance(cam_for_corr, dict) else None,
        )
    except Exception:
        _log.exception("reliability fusion or temporal corroboration failed")

    # Optional NOMA permutation sensitivity (capped blocks; extra inference cost).
    if noma_permutation_max_blocks is not None:
        mb = int(noma_permutation_max_blocks)
        if mb > 0:
            try:
                from detectors.noma import _load_noma_pipeline, run_noma_prediction_with_features
                from explainability.noma_feature_sensitivity import (
                    compute_noma_permutation_feature_sensitivity,
                )

                mb = min(mb, 60)
                feat_out = run_noma_prediction_with_features(noma_model_path, audio_path=audio_path)
                times_f = feat_out[0]
                X = feat_out[2]
                fn = feat_out[3]
                pipeline = _load_noma_pipeline(noma_model_path)
                result["noma_permutation_xai"] = compute_noma_permutation_feature_sensitivity(
                    feature_matrix=X,
                    pipeline=pipeline,
                    feature_names=fn,
                    block_times_seconds=times_f,
                    max_blocks=mb,
                    top_k=5,
                )
            except Exception as exc:  # noqa: BLE001
                result["noma_permutation_xai"] = {"error": str(exc)}

    # Step 4: optional evidence bundle export
    if export_bundle:
        bundle_bytes = zip_evidence_bundle(
            input_video_path=video_path,
            input_video_name=video_name,
            avh_score=result["avh_score"],
            audio_path=audio_path,
            roi_path=result["roi_path"],
            cam_idx=result["cam_idx"],
            overlays_dir=result["cam_overlays_dir"],
            noma_df=noma_df,
        )
        result["bundle_bytes"] = bundle_bytes

    if persist_run_dir and result.get("audio_path"):
        _persist_combined_artifacts(
            result,
            persist_run_dir=os.path.abspath(persist_run_dir),
            cleanup_volatile_after_persist=cleanup_volatile_after_persist,
        )

    return result

