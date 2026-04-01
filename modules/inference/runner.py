from typing import Any, Tuple

import numpy as np

from config import PROJECT_ROOT


def run_noma_on_audio_bytes(model_path: str, audio_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper to run NOMA on raw audio bytes.
    """
    import io
    from detectors.noma import run_noma_prediction

    buf = io.BytesIO(audio_bytes)
    times, probas = run_noma_prediction(model_path, audio_bytes=buf, audio_filename="upload")
    return np.asarray(times), np.asarray(probas)


def run_combined_pipeline(
    video_path: str,
    video_name: str,
    use_unsup_avh: bool,
    python_exe: str,
    run_forensics_cam: bool,
    forensics_top_k: int,
    forensics_selection_mode: str = "top_k",
    forensics_min_temporal_gap: int = 24,
    forensics_max_fusion_frames: int = 200,
    region_track_stride: int = 1,
    run_robustness_delta: bool,
    adv_ckpt_path: str,
    capture_attention: bool,
    export_bundle: bool,
    noma_model_path: str,
    timeout: int = 900,
) -> dict[str, Any]:
    """
    Thin wrapper around orchestrator.combined_runner.run_combined_avh_to_noma
    for use in the new dashboard.
    """
    from orchestrator.combined_runner import run_combined_avh_to_noma

    return run_combined_avh_to_noma(
        video_path=video_path,
        video_name=video_name,
        use_unsup_avh=use_unsup_avh,
        python_exe=python_exe,
        run_forensics_cam=run_forensics_cam,
        forensics_top_k=forensics_top_k,
        forensics_selection_mode=forensics_selection_mode,
        forensics_min_temporal_gap=forensics_min_temporal_gap,
        forensics_max_fusion_frames=forensics_max_fusion_frames,
        region_track_stride=region_track_stride,
        run_robustness_delta=run_robustness_delta,
        adv_ckpt_path=adv_ckpt_path,
        capture_attention=capture_attention,
        export_bundle=export_bundle,
        noma_model_path=noma_model_path,
        timeout=timeout,
    )

