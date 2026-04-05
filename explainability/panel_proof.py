"""
Slide- and demo-friendly "proof snippets" built from existing explainability outputs.

Use these strings/JSON blocks in talks or appendices: they restate *what was computed*
(reliability fusion, temporal corroboration, Grad-CAM status) without claiming legal proof.
"""

from __future__ import annotations

import json
import math
from typing import Any


def fusion_proof_chain(res: dict[str, Any] | None) -> dict[str, Any]:
    """
    Human-readable steps for reliability-weighted fusion (see explainability/reliability_fusion.py).

    Returns JSON-serializable dict for UI / download.
    """
    if not isinstance(res, dict):
        return {"status": "no_result"}

    p_a = res.get("p_audio_mean")
    p_v = res.get("p_avh_cal")
    tau = res.get("fusion_tau")
    tension = res.get("fusion_tension")
    w_a = res.get("fusion_w_audio")
    p_f = res.get("p_fused")
    verdict = res.get("fusion_verdict")
    late_mode = str(res.get("late_fusion_mode") or "full").strip().lower()

    if p_a is None or p_v is None or tau is None:
        return {
            "status": "incomplete",
            "note": "Run full Combined pipeline so p_audio_mean, p_avh_cal, fusion_tau are populated.",
        }

    if late_mode == "mean":
        p_fused = float(p_f) if p_f is not None else 0.5 * (float(p_a) + float(p_v))
        return {
            "status": "ok",
            "formula": "p_fused = 0.5 * (p_audio_mean + p_avh_cal)",
            "inputs": {
                "p_audio_mean": float(p_a),
                "p_avh_cal": float(p_v),
                "fusion_tau": float(tau),
                "late_fusion_mode": late_mode,
            },
            "derived": {
                "fusion_tension": float(tension) if tension is not None else abs(float(p_v) - float(p_a)),
                "p_fused": p_fused,
                "fusion_verdict": verdict,
            },
            "steps_plain": [
                "Late fusion mode: simple mean (not full reliability blend).",
                "Let p_audio = mean calibrated NOMA p(fake) over 1s blocks.",
                "Let p_avh = calibrated AVH p(fake).",
                f"p_fused = (p_audio + p_avh) / 2 = {p_fused:.4f}.",
                f"Verdict bands: compare p_fused to 0.5 ± τ with τ={float(tau):.4f} → `{verdict or 'n/a'}`.",
            ],
        }

    if late_mode == "audio_primary":
        p_fused = float(p_f) if p_f is not None else float(p_a)
        return {
            "status": "ok",
            "formula": "p_fused = p_audio_mean (NOMA-primary)",
            "inputs": {
                "p_audio_mean": float(p_a),
                "p_avh_cal": float(p_v),
                "fusion_tau": float(tau),
                "late_fusion_mode": late_mode,
            },
            "derived": {
                "fusion_tension": float(tension) if tension is not None else abs(float(p_v) - float(p_a)),
                "p_fused": p_fused,
                "fusion_verdict": verdict,
            },
            "steps_plain": [
                "Late fusion mode: audio_primary (verdict follows mean NOMA p(fake)).",
                f"p_fused = p_audio = {p_fused:.4f}.",
                f"Verdict bands: compare to 0.5 ± τ, τ={float(tau):.4f} → `{verdict or 'n/a'}`.",
            ],
        }

    if late_mode == "video_primary":
        p_fused = float(p_f) if p_f is not None else float(p_v)
        return {
            "status": "ok",
            "formula": "p_fused = p_avh_cal (AVH-primary)",
            "inputs": {
                "p_audio_mean": float(p_a),
                "p_avh_cal": float(p_v),
                "fusion_tau": float(tau),
                "late_fusion_mode": late_mode,
            },
            "derived": {
                "fusion_tension": float(tension) if tension is not None else abs(float(p_v) - float(p_a)),
                "p_fused": p_fused,
                "fusion_verdict": verdict,
            },
            "steps_plain": [
                "Late fusion mode: video_primary (verdict follows calibrated AVH p(fake)).",
                f"p_fused = p_avh = {p_fused:.4f}.",
                f"Verdict bands: compare to 0.5 ± τ, τ={float(tau):.4f} → `{verdict or 'n/a'}`.",
            ],
        }

    tau_b = max(float(tau), 1e-9)
    t = float(tension) if tension is not None else abs(float(p_v) - float(p_a))
    w_expect = math.exp(-t / tau_b)
    w_audio = float(w_a) if w_a is not None else w_expect
    p_fused = float(p_f) if p_f is not None else (w_audio * float(p_a) + float(p_v)) / (w_audio + 1.0)

    steps = [
        "Let p_audio = mean calibrated NOMA p(fake) over 1s blocks.",
        "Let p_avh = calibrated AVH p(fake) from lip–speech fusion score.",
        f"Tension = |p_avh − p_audio| = {t:.4f}.",
        f"τ = max(calibration margins) = {float(tau):.4f} (uncertainty band for verdicts).",
        f"Audio weight w_audio = exp(−tension/τ) = {w_audio:.4f} (down-weight audio when streams disagree).",
        f"p_fused = (w_audio·p_audio + p_avh) / (w_audio + 1) = {p_fused:.4f} (with softening + high-tension mean blend; see reliability_fusion.py).",
        f"Verdict bands: compare p_fused to 0.5 ± τ_eff → `{verdict or 'n/a'}`.",
    ]

    return {
        "status": "ok",
        "formula": "w_audio = exp(-tension/tau); p_fused = (w_audio*p_audio + p_avh) / (w_audio + 1)",
        "inputs": {
            "p_audio_mean": float(p_a),
            "p_avh_cal": float(p_v),
            "fusion_tau": float(tau),
        },
        "derived": {
            "fusion_tension": float(tension) if tension is not None else t,
            "fusion_w_audio": w_audio,
            "p_fused": p_fused,
            "fusion_verdict": verdict,
        },
        "steps_plain": steps,
        "sanity_check_w_audio": {"reported": w_a, "recomputed_exp_minus_t_over_tau": w_expect},
    }


def corroboration_proof_snippet(res: dict[str, Any] | None) -> dict[str, Any]:
    """
    Pick strongest corroboration and conflict seconds from temporal_corroboration bins.
    """
    if not isinstance(res, dict):
        return {"status": "no_result", "headline": "No Combined result dict."}
    tc = res.get("temporal_corroboration")
    if not isinstance(tc, dict) or tc.get("status") != "ok":
        return {
            "status": tc.get("status", "unavailable") if isinstance(tc, dict) else "unavailable",
            "headline": "No time-aligned audio vs video saliency table (enable Grad-CAM + Combined run).",
        }

    bins = tc.get("bins") or []
    if not bins:
        return {"status": "empty", "headline": "Corroboration bins empty."}

    corro = [b for b in bins if isinstance(b, dict) and b.get("corroboration")]
    conf = [b for b in bins if isinstance(b, dict) and b.get("conflict")]

    def _key_c(b: dict[str, Any]) -> float:
        return float(b.get("p_fake", 0.0)) * float(b.get("saliency", 0.0))

    def _key_f(b: dict[str, Any]) -> float:
        return float(b.get("p_fake", 0.0)) * (1.0 - float(b.get("saliency", 0.0)))

    best_c = max(corro, key=_key_c) if corro else None
    best_f = max(conf, key=_key_f) if conf else None

    lines = [
        f"Corroboration rate (fraction of blocks): {float(tc.get('corroboration_rate') or 0.0):.3f}",
        f"Conflict rate: {float(tc.get('conflict_rate') or 0.0):.3f}",
    ]
    if best_c:
        lines.append(
            f"Example corroboration at ~{best_c.get('second', '?')}s: "
            f"p_fake={float(best_c.get('p_fake', 0)):.3f}, saliency={float(best_c.get('saliency', 0)):.3f} "
            "(high audio suspicion + high video saliency)."
        )
    if best_f:
        lines.append(
            f"Example conflict at ~{best_f.get('second', '?')}s: "
            f"p_fake={float(best_f.get('p_fake', 0)):.3f}, saliency={float(best_f.get('saliency', 0)):.3f} "
            "(high audio suspicion + low video saliency)."
        )

    return {
        "status": "ok",
        "thresholds": {
            "p_threshold": tc.get("p_threshold"),
            "sal_threshold": tc.get("sal_threshold"),
        },
        "headline": " | ".join(lines),
        "best_corroboration_bin": best_c,
        "best_conflict_bin": best_f,
    }


def gradcam_status_proof(cam_idx: dict[str, Any] | None) -> dict[str, Any]:
    """One screen of facts about what XAI stages ran (from gradcam index + enrichments)."""
    if not isinstance(cam_idx, dict) or not cam_idx:
        return {"status": "missing", "headline": "No Grad-CAM index in session."}

    xs = cam_idx.get("xai_status") if isinstance(cam_idx.get("xai_status"), dict) else {}
    parts = [
        f"Temporal inconsistency: {xs.get('temporal_inconsistency', '?')}",
        f"Region tracks (IoU): {xs.get('region_tracks', '?')}",
        f"Multi-signal fusion: {xs.get('fusion', '?')}",
        f"Frequency stats: {xs.get('video_frequency_stats', '?')}",
    ]
    return {
        "status": "ok",
        "headline": " · ".join(parts),
        "artifacts": {
            "cam_volume": bool(cam_idx.get("cam_volume_path")),
            "fused_heatmap": bool(cam_idx.get("fused_heatmap_path")),
            "overlay_dir": bool(cam_idx.get("overlay_dir")),
            "T_use": cam_idx.get("T_use"),
            "roi_fps": cam_idx.get("roi_fps"),
        },
    }


def build_panel_proof_markdown(res: dict[str, Any] | None, cam_idx: dict[str, Any] | None) -> str:
    """Single markdown block for slides / appendix copy-paste."""
    fus = fusion_proof_chain(res)
    cor = corroboration_proof_snippet(res if isinstance(res, dict) else None)
    gc = gradcam_status_proof(cam_idx)

    lines = [
        "## Multimodal screening — proof chain (automated)",
        "",
        "### 1. Reliability-weighted fusion (audio vs video streams)",
    ]
    if fus.get("status") == "ok":
        lines.extend([f"- {s}" for s in fus.get("steps_plain", [])])
    else:
        lines.append(f"- _{fus.get('note', fus.get('status', 'n/a'))}_")

    lines.extend(["", "### 2. Time-aligned audio vs Grad-CAM saliency", f"- {cor.get('headline', '')}"])

    lines.extend(["", "### 3. Grad-CAM / XAI pipeline status", f"- {gc.get('headline', '')}"])

    lines.extend(
        [
            "",
            "_Disclaimer: screening aid; not legal or cryptographic proof._",
        ]
    )
    return "\n".join(lines)


def build_panel_proof_bundle(res: dict[str, Any] | None, cam_idx: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-serializable bundle for download alongside final_combined_report.json."""
    return {
        "fusion": fusion_proof_chain(res),
        "corroboration": corroboration_proof_snippet(res if isinstance(res, dict) else None),
        "gradcam": gradcam_status_proof(cam_idx),
        "markdown_appendix": build_panel_proof_markdown(res, cam_idx),
    }


def bundle_to_json_bytes(bundle: dict[str, Any]) -> bytes:
    return json.dumps(bundle, indent=2).encode("utf-8")
