"""
Build structured JSON for Gemini to explain Combined-report UI (metrics, plots, tables).
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd


def _df_summary(df: pd.DataFrame | None, max_rows: int = 8) -> dict[str, Any] | None:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    out: dict[str, Any] = {"columns": list(df.columns), "n_rows": len(df)}
    try:
        out["head"] = df.head(max_rows).to_dict(orient="records")
    except Exception:
        out["head"] = []
    return out


# Static glossary: Gemini uses this + live values; avoids inventing definitions.
UI_GLOSSARY: dict[str, str] = {
    "user_summary": (
        "Plain-language summary: fused p(fake) as 0–1 risk, screening verdict (Likely FAKE / REAL / Uncertain), "
        "and tension between AVH (lip–audio) vs NOMA (speech-only)."
    ),
    "avh": (
        "AVH-Align: score from a fusion model on mouth ROI video + audio; calibrated to AVH p(fake). "
        "High values suggest lip–audio mismatch or synthetic patterns, not a court verdict."
    ),
    "noma": (
        "NOMA: per 1-second blocks, hand-crafted audio features + classifier → p(fake) per second. "
        "Speech-only; does not see the face."
    ),
    "late_fusion": (
        "Late fusion: combines calibrated AVH p(fake) with mean NOMA p(fake) (mode: full reliability, mean, "
        "audio_primary, video_primary, or learned). p_fused is the blended risk; fusion_tension is |p_avh − p_noma_mean|."
    ),
    "cmid": (
        "CMID / cross-modal sync: cosine similarity between audio and visual embeddings over time (needs AVH "
        "embedding dump). missing_embeddings means embeddings were not saved for this run."
    ),
    "cii": (
        "CII (confidence instability index): variation of NOMA calibrated p(fake) across seconds; higher = more "
        "inconsistent block-level scores."
    ),
    "temporal_corroboration": (
        "Time-aligned bins: compares NOMA p(fake) per second with Grad-CAM saliency proxies; corroboration vs conflict "
        "flags when audio suspicion aligns or misaligns with salient frames."
    ),
    "gradcam": (
        "Grad-CAM: class-discriminative saliency on the mouth ROI CNN path; highlights where the model looked. "
        "Evidence of sensitivity, not proof of manipulation."
    ),
    "fused_heatmap": (
        "Fused heatmap (Altair): one 2D slice of a volume combining Grad-CAM with optical-flow and frequency cues; "
        "inferno color scale = intensity. Read: hotter = more fused attention to that spatial region at that frame."
    ),
    "gradcam_intensity_chart": (
        "Line chart: x = time (seconds), y = mean Grad-CAM intensity per frame over the ROI; shows when saliency peaks."
    ),
    "noma_permutation": (
        "NOMA permutation sensitivity: which audio features most affect p(fake) when perturbed (developer diagnostic)."
    ),
    "research_tab": (
        "Models + web research: SerpAPI web + Google Lens + NewsAPI + Gemini; separate from detection scores unless "
        "you include Combined context. API keys required for web sources."
    ),
    "export_json": (
        "Download final_combined_report.json: snapshot of verdict, scores, fusion, cmid_status, CII, temporal_corroboration."
    ),
}


def build_combined_report_guide_payload(
    res: dict[str, Any],
    cam_idx: dict[str, Any] | None,
    *,
    use_unsup_avh: bool,
) -> dict[str, Any]:
    """Structured payload for Gemini (full report or per-section)."""
    noma_df = res.get("noma_df")
    inst = res.get("noma_confidence_instability")
    cii = None
    if isinstance(inst, dict) and inst.get("CII") is not None:
        try:
            cii = float(inst["CII"])
        except (TypeError, ValueError):
            cii = None

    tc = res.get("temporal_corroboration")
    tc_ok = isinstance(tc, dict) and tc.get("status") == "ok"

    cam = cam_idx if isinstance(cam_idx, dict) else {}
    has_gradcam = bool(cam)
    fused_path = cam.get("fused_heatmap_path")
    has_fused = isinstance(fused_path, str) and os.path.isfile(fused_path)
    has_cam_line = isinstance(cam.get("cam_per_frame"), list) and len(cam.get("cam_per_frame") or []) > 0
    overlay_dir = cam.get("overlay_dir") or res.get("cam_overlays_dir")
    has_overlays = isinstance(overlay_dir, str) and os.path.isdir(overlay_dir)

    perm = res.get("noma_permutation_xai")
    has_perm = isinstance(perm, dict) and "error" not in perm and perm.get("feature_names")

    return {
        "ui_glossary": UI_GLOSSARY,
        "run_flags": {
            "use_unsup_avh": bool(use_unsup_avh),
            "has_gradcam_index": has_gradcam,
            "has_gradcam_overlays": has_overlays,
            "has_fused_heatmap_volume": has_fused,
            "has_gradcam_intensity_line": has_cam_line,
            "has_temporal_corroboration": tc_ok,
            "has_noma_permutation_xai": bool(has_perm),
            "cmid_status": res.get("cmid_status"),
        },
        "run_values": {
            "avh_score": res.get("avh_score"),
            "p_avh_cal": res.get("p_avh_cal"),
            "p_fused": res.get("p_fused"),
            "fusion_verdict": res.get("fusion_verdict"),
            "fusion_tension": res.get("fusion_tension"),
            "fusion_tau": res.get("fusion_tau"),
            "late_fusion_mode": res.get("late_fusion_mode"),
            "noma_blocks": int(len(noma_df)) if isinstance(noma_df, pd.DataFrame) else 0,
            "noma_mean_p_fake": float(noma_df["p_fake"].astype(float).mean())
            if isinstance(noma_df, pd.DataFrame) and len(noma_df) > 0 and "p_fake" in noma_df.columns
            else None,
            "cii": cii,
            "temporal_corroboration_bins_preview": (tc.get("bins")[:12] if tc_ok and isinstance(tc.get("bins"), list) else None),
            "gradcam_meta": {
                "T_use": cam.get("T_use"),
                "roi_fps": cam.get("roi_fps"),
                "xai_status": cam.get("xai_status"),
            }
            if has_gradcam
            else None,
            "noma_table_preview": _df_summary(noma_df if isinstance(noma_df, pd.DataFrame) else None),
            "permutation_top_features": (perm.get("top_features") if isinstance(perm, dict) else None),
        },
    }


XAI_STANDALONE_GLOSSARY: dict[str, str] = {
    "audio_cii": (
        "CII (confidence instability index): rolling variance of calibrated NOMA p(fake) over time; "
        "plotted as variance per index vs time. Higher CII ⇒ more disagreement across 1s blocks."
    ),
    "audio_page": (
        "Audio Explainability page shows math references and, after a Combined run, the CII variance timeline "
        "from the last session. It does not replace the full NOMA table (see Inference Demo or Final Combined Report)."
    ),
    "video_gradcam": (
        "Grad-CAM: class-discriminative saliency on the mouth ROI CNN path in AV-HuBERT; highlights where the "
        "model attends — sensitivity, not proof of manipulation."
    ),
    "video_temporal": (
        "Temporal inconsistency curve (Δ_t): frame-to-frame change in embedding or CAM-derived signal; "
        "spikes may indicate unstable frames."
    ),
    "video_frequency": (
        "High-frequency energy over time: proxy for texture/band energy in the ROI; interpret as a weak forensic cue."
    ),
    "video_fused_intensity": (
        "Mean fused heatmap intensity per frame: combines Grad-CAM with optical-flow and frequency maps into one scalar "
        "per time step — higher ⇒ stronger combined anomaly signal in that frame."
    ),
    "video_region_tracks": (
        "Region tracks: IoU-based boxes on high-CAM pixels over time; summarizes where spatial attention stayed."
    ),
}


def build_xai_standalone_payload(
    kind: str,
    res: dict[str, Any] | None,
    cam_idx: dict[str, Any] | None,
) -> dict[str, Any]:
    """Payload for Audio / Video Explainability pages (not the full Combined report)."""
    res = res if isinstance(res, dict) else {}
    cam = cam_idx if isinstance(cam_idx, dict) else {}
    if kind == "audio":
        inst = res.get("noma_confidence_instability")
        has_cii = isinstance(inst, dict) and isinstance(inst.get("variance_per_time"), list)
        return {
            "page": "audio_explainability",
            "glossary": XAI_STANDALONE_GLOSSARY,
            "run_values": {
                "has_cii_timeline": has_cii,
                "cii_scalar": float(inst.get("CII", 0.0)) if isinstance(inst, dict) and inst.get("CII") is not None else None,
                "n_blocks_implied": len(inst["variance_per_time"]) if has_cii else None,
            },
        }
    if kind == "video":
        has_cam = bool(cam)
        roi_fps = cam.get("roi_fps")
        cam_per = cam.get("cam_per_frame")
        return {
            "page": "video_explainability",
            "glossary": XAI_STANDALONE_GLOSSARY,
            "run_flags": {
                "has_gradcam_session": has_cam,
                "has_cam_intensity_line": isinstance(cam_per, list) and len(cam_per) > 0,
                "has_temporal_inconsistency": isinstance(cam.get("temporal_inconsistency"), list),
                "has_region_tracks": bool((cam.get("region_tracks") or {}).get("tracks")),
                "has_high_freq": isinstance(cam.get("video_frequency_stats"), dict),
                "has_fused_heatmap_file": isinstance(cam.get("fused_heatmap_path"), str)
                and os.path.isfile(str(cam.get("fused_heatmap_path"))),
            },
            "run_values": {
                "T_use": cam.get("T_use"),
                "roi_fps": roi_fps,
                "xai_status": cam.get("xai_status"),
            },
        }
    raise ValueError(f"Unknown xai kind: {kind!r}")


XAI_SECTION_LABELS: list[tuple[str, str]] = [
    ("xai_audio", "Audio Explainability — CII & instability"),
    ("xai_video", "Video Explainability — Grad-CAM, tracks, fused intensity"),
]


SECTION_LABELS: list[tuple[str, str]] = [
    ("full", "Entire report (all sections below)"),
    ("user_summary", "User summary — verdict, progress bar, tension"),
    ("score_table", "Developer — score table (AVH, NOMA, fusion)"),
    ("diagnostics", "Other diagnostics — CMID status, CII"),
    ("noma_per_second", "NOMA per-second table"),
    ("temporal_corroboration", "Audio–video alignment / temporal corroboration"),
    ("gradcam", "Grad-CAM & ROI — overlays, JSON meta, mouth video"),
    ("fused_heatmap", "Fused heatmap chart (Grad-CAM + flow + freq)"),
    ("gradcam_intensity", "Grad-CAM intensity vs time (line chart)"),
    ("noma_permutation", "NOMA permutation sensitivity"),
    ("research_tab", "Models + web research tab (Serp / Lens / News)"),
    ("export_json", "Download final_combined_report.json"),
]
