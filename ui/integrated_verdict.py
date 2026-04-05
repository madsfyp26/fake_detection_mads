"""
Final Combined Report: user-facing summary, developer forensics (tables + Grad-CAM),
and one-shot research (Serp + Google Lens + News + Gemini) with model context.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config import get_late_fusion_mode
from ui.env_keys_help import render_missing_data_api_keys_hint
from ui.i18n import t
from ui.report_explain_payload import SECTION_LABELS, build_combined_report_guide_payload


def _overlay_paths(cam_idx: dict[str, Any] | None, res: dict[str, Any]) -> list[str]:
    d = None
    if isinstance(cam_idx, dict):
        d = cam_idx.get("overlay_dir")
    if not d or not os.path.isdir(d):
        d = res.get("cam_overlays_dir")
    if not d or not os.path.isdir(d):
        return []
    paths = sorted(glob.glob(os.path.join(d, "cam_frame_*.png")))
    return paths


def _fused_heatmap_slice_chart(fused: np.ndarray, frame_idx: int) -> alt.Chart | None:
    if fused.ndim != 3 or frame_idx < 0 or frame_idx >= fused.shape[0]:
        return None
    z = np.asarray(fused[frame_idx], dtype=float)
    h, w = z.shape
    rows: list[dict[str, Any]] = []
    for yy in range(h):
        for xx in range(w):
            rows.append({"x": xx, "y": h - 1 - yy, "v": float(z[yy, xx])})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("x:O", title=None, axis=None),
            y=alt.Y("y:O", title=None, axis=None),
            color=alt.Color("v:Q", title="intensity", scale=alt.Scale(scheme="inferno")),
            tooltip=["v"],
        )
        .properties(
            width=min(420, max(8 * w, 120)),
            height=min(420, max(8 * h, 120)),
            title="Fused heatmap (Grad-CAM + flow + freq)",
        )
    )


def render_integrated_final_report(
    *,
    res: dict[str, Any],
    cam_idx: dict[str, Any] | None,
    lang: str,
    use_unsup_avh: bool,
    render_limitations: Callable[..., None],
) -> None:
    from calibration_runtime import avh_score_to_calibrated_p_fake

    st.markdown("## Final Combined Report")
    st.caption(
        "Two views below: a short **user** summary and a **developer** breakdown. "
        "The **Models + web research** tab runs SerpAPI (web + Google Lens), NewsAPI, and Gemini — "
        "optionally with scores from this Combined run as extra context."
    )

    guide_payload = build_combined_report_guide_payload(
        res, cam_idx if isinstance(cam_idx, dict) else None, use_unsup_avh=use_unsup_avh
    )
    with st.expander("Gemini guide — simple English + technical “how it works”", expanded=False):
        st.caption(
            "Uses **GEMINI_API_KEY** only (no Serp/News required). "
            "The answer is written in **plain English**, with a separate **technical** section on how AVH, NOMA, fusion, and Grad-CAM actually work."
        )
        g1, g2 = st.columns([3, 1])
        with g1:
            sec_choice = st.selectbox(
                "Section",
                options=[x[0] for x in SECTION_LABELS],
                format_func=lambda k: next(t for t in SECTION_LABELS if t[0] == k)[1],
                key="gemini_guide_section_select",
            )
        with g2:
            st.markdown("")  # spacer
            run_guide = st.button("Generate explanation", type="primary", key="gemini_guide_run_btn")
        if run_guide:
            from integrations.research_chat.gemini_client import synthesize_ui_guide

            title = next(t for t in SECTION_LABELS if t[0] == sec_choice)[1]
            with st.spinner("Asking Gemini…"):
                text, err = synthesize_ui_guide(
                    section_id=sec_choice,
                    section_title=title,
                    guide_payload=guide_payload,
                )
            st.session_state["integrated_gemini_guide_text"] = text
            st.session_state["integrated_gemini_guide_error"] = err
        if st.session_state.get("integrated_gemini_guide_error"):
            st.error(st.session_state["integrated_gemini_guide_error"])
        gtxt = st.session_state.get("integrated_gemini_guide_text")
        if gtxt:
            st.markdown("---")
            st.markdown(gtxt)

    avh_score = res.get("avh_score")
    avh_p_fake = res.get("p_avh_cal")
    if avh_p_fake is None and isinstance(avh_score, (int, float)):
        avh_p_fake = float(
            avh_score_to_calibrated_p_fake(
                float(avh_score),
                use_unsup_avh=bool(res.get("use_unsup_avh", use_unsup_avh)),
            )
        )

    noma_df = res.get("noma_df")
    noma_mean_p_fake = None
    noma_blocks = 0
    if isinstance(noma_df, pd.DataFrame) and "p_fake" in noma_df.columns:
        noma_blocks = int(len(noma_df))
        if noma_blocks > 0:
            noma_mean_p_fake = float(noma_df["p_fake"].astype(float).mean())

    p_fused_res = res.get("p_fused")
    fusion_tension = res.get("fusion_tension")
    fusion_tau = res.get("fusion_tau")
    fusion_verdict = res.get("fusion_verdict")
    late_mode = res.get("late_fusion_mode") or get_late_fusion_mode()

    blended_p_fake = None
    verdict = "Insufficient evidence"
    if p_fused_res is not None:
        blended_p_fake = float(p_fused_res)
        verdict = str(fusion_verdict) if fusion_verdict else "Uncertain"
    else:
        weights = {"avh": 0.55, "noma": 0.45}
        blend_terms = []
        if avh_p_fake is not None:
            blend_terms.append((weights["avh"], avh_p_fake))
        if noma_mean_p_fake is not None:
            blend_terms.append((weights["noma"], noma_mean_p_fake))
        if blend_terms:
            w_sum = sum(w for w, _ in blend_terms)
            blended_p_fake = float(sum(w * v for w, v in blend_terms) / max(w_sum, 1e-9))
        if blended_p_fake is not None:
            if blended_p_fake >= 0.65:
                verdict = "Likely FAKE"
            elif blended_p_fake <= 0.35:
                verdict = "Likely REAL"
            else:
                verdict = "Uncertain"

    tab_user, tab_dev, tab_res = st.tabs(["User summary", "Developer forensics", "Models + web research"])

    with tab_user:
        st.markdown("### Plain-language result")
        if blended_p_fake is not None:
            try:
                st.progress(float(blended_p_fake), text=f"Fused risk (p fake): {blended_p_fake:.0%}")
            except TypeError:
                st.progress(float(blended_p_fake))
                st.caption(f"Fused risk (p fake): {blended_p_fake:.0%}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Screening verdict", verdict)
        with c2:
            st.metric("Fused p(fake)", f"{blended_p_fake:.3f}" if blended_p_fake is not None else "n/a")
        with c3:
            st.metric("AVH vs NOMA tension", f"{float(fusion_tension):.3f}" if fusion_tension is not None else "n/a")
        st.caption(
            f"Late fusion mode: `{late_mode}` (`LATE_FUSION_MODE` env: full | mean | audio_primary | video_primary)."
        )
        st.markdown(
            """
**What this means**
- **AVH** looks at whether **lip motion matches** the soundtrack (visual + audio together).
- **NOMA** looks at **speech alone** (timbre / artifacts typical of synthesis).
- **Late-fused p(fake)** combines AVH and NOMA per the mode above; **tension** is how much the two disagree.
            """
        )
        st.info(
            "**You decide what to do next** — these are screening signals, not proof. "
            "Use the **Models + web research** tab to cross-check claims about the clip or topic with external sources."
        )
        render_limitations(use_unsup_avh=use_unsup_avh)

    with tab_dev:
        with st.expander("Quick guide: what you are looking at (developer view)", expanded=False):
            st.markdown(
                """
**Plain English.** This tab shows the **raw numbers** behind the user summary: lip–video score (AVH), speech score (NOMA), how they were merged (fusion), and optional **explainability** (Grad-CAM, alignment tables, reviewer PNGs).

**Technical (how it fits together).**  
1. **AVH** runs AV-HuBERT on mouth ROI + audio, then a **Fusion MLP** outputs a single lip–speech consistency score; **calibration** maps it to a calibrated **p(fake)**.  
2. **NOMA** slices audio into **1-second blocks**, extracts **41 hand-crafted features** (spectral + MFCC-like), then an **SVM** outputs **p(fake)** per second.  
3. **Late fusion** blends calibrated AVH and mean NOMA **p(fake)** (reliability or learned mode).  
4. **Grad-CAM** backprops through the visual branch to show **which mouth pixels** drove the score (sensitivity map, not proof).  
5. **Reviewer figures** bundle ROI / CAM / mel / CMID into PNGs for slides.

Use **Gemini guide** at the top of this report for a longer, section-by-section walkthrough.
                """
            )
        st.markdown("### Score table (models)")
        _p_fused_notes = {
            "full": "Reliability blend (explainability/reliability_fusion.py); τ from margins",
            "mean": "Simple mean of calibrated AVH and mean NOMA p(fake)",
            "audio_primary": "p_fused = mean NOMA p(fake)",
            "video_primary": "p_fused = calibrated AVH p(fake)",
        }.get(late_mode, "Late fusion output")
        score_rows = [
            {"signal": "AVH raw score", "value": avh_score, "notes": "Fusion model output before calibration mapping"},
            {"signal": "AVH p(fake) cal.", "value": avh_p_fake, "notes": "Temperature scaling from calibration_artifacts.json"},
            {"signal": "NOMA mean p(fake)", "value": noma_mean_p_fake, "notes": f"Mean over {noma_blocks} 1s blocks"},
            {"signal": "late_fusion_mode", "value": late_mode, "notes": "From LATE_FUSION_MODE env or Combined run"},
            {"signal": "p_fused (late fusion)", "value": p_fused_res, "notes": _p_fused_notes},
            {"signal": "fusion_tension", "value": fusion_tension, "notes": "|p_avh − p_noma_mean|"},
            {"signal": "fusion τ", "value": fusion_tau, "notes": "Uncertainty band used in fusion"},
        ]
        sdf = pd.DataFrame(score_rows)
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.markdown("### Other diagnostics")
        st.markdown(f"- **cmid_status:** `{res.get('cmid_status', 'unknown')}`")
        inst0 = res.get("noma_confidence_instability")
        if isinstance(inst0, dict) and inst0.get("CII") is not None:
            st.markdown(f"- **CII** (NOMA confidence instability): `{float(inst0.get('CII', 0.0)):.4e}`")

        if isinstance(noma_df, pd.DataFrame) and len(noma_df) > 0:
            st.markdown("### NOMA: per-second Real / Fake (audio)")
            show = noma_df.copy()
            if "Prediction" in show.columns:
                st.dataframe(
                    show,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "p_fake": st.column_config.ProgressColumn("p_fake", format="%.3f", min_value=0.0, max_value=1.0),
                    },
                )
            else:
                st.dataframe(show, use_container_width=True, hide_index=True)

        tc = res.get("temporal_corroboration")
        if isinstance(tc, dict) and tc.get("status") == "ok" and isinstance(tc.get("bins"), list) and tc["bins"]:
            st.markdown("### Audio–video alignment (NOMA vs video saliency)")
            st.caption(
                "Each row is one 1s block: **p_fake** from NOMA; **saliency** from fused Grad-CAM / ROI timeline. "
                "**corroboration** = high audio suspicion + high saliency; **conflict** = high audio suspicion + low saliency."
            )
            bdf = pd.DataFrame(tc["bins"])
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        if not isinstance(cam_idx, dict):
            st.info("No Grad-CAM session index — run Combined with **forensics / Grad-CAM** enabled.")
        else:
            st.markdown("### Grad-CAM & ROI (mouth crop)")
            st.caption(
                "**Simple:** heatmaps and overlays show **where the model looked** on the mouth crop. "
                "**Technical:** Grad-CAM uses gradients of the AV-HuBERT visual path w.r.t. the fusion objective; "
                "high values mean that region influenced the lip–speech score strongly."
            )
            xai_status = cam_idx.get("xai_status") if isinstance(cam_idx.get("xai_status"), dict) else {}
            st.json(
                {
                    "cam_score": cam_idx.get("score"),
                    "T_cam_full": cam_idx.get("T_cam_full"),
                    "T_roi": cam_idx.get("T_roi"),
                    "T_use": cam_idx.get("T_use"),
                    "roi_fps": cam_idx.get("roi_fps"),
                    "xai_status": xai_status,
                }
            )
            roi_path = res.get("roi_path")
            if isinstance(roi_path, str) and os.path.isfile(roi_path):
                st.markdown("**Mouth ROI video** (what AVH / Grad-CAM see)")
                st.video(roi_path)

            paths = _overlay_paths(cam_idx, res)
            if paths:
                st.markdown("**Saved Grad-CAM overlay frames** (top-K temporal picks; existing AVH script output)")
                st.caption(
                    "**Simple:** a few key frames with heat overlaid on the mouth. "
                    "**Technical:** frames are chosen for diversity / salience (not every second of video)."
                )
                idx = st.slider("Overlay frame index", 0, len(paths) - 1, 0, key="dev_overlay_slider")
                st.image(paths[idx], caption=os.path.basename(paths[idx]), use_container_width=True)
            else:
                st.caption("No overlay PNGs found under `overlay_dir` / `cam_overlays_dir`.")

            tracks = (
                cam_idx.get("region_tracks", {}).get("tracks", [])
                if isinstance(cam_idx.get("region_tracks"), dict)
                else []
            )
            if tracks:
                st.markdown("### Region tracks (IoU on high-CAM pixels)")
                st.caption(
                    "**mean_cam / max_cam** = average max saliency inside the tracked box (proxy for “where attention stayed”). "
                    "This is not a separate fake/real classifier — it summarizes spatial Grad-CAM."
                )
                flat: list[dict[str, Any]] = []
                for i, tr in enumerate(tracks[:50]):
                    if not isinstance(tr, dict):
                        continue
                    flat.append(
                        {
                            "track": i,
                            "start_frame": tr.get("start_frame"),
                            "end_frame": tr.get("end_frame"),
                            "duration_frames": tr.get("duration_frames"),
                            "mean_cam": tr.get("mean_cam"),
                            "max_cam": tr.get("max_cam"),
                        }
                    )
                st.dataframe(pd.DataFrame(flat), use_container_width=True, hide_index=True)

            fused_path = cam_idx.get("fused_heatmap_path")
            if isinstance(fused_path, str) and os.path.isfile(fused_path):
                try:
                    fused = np.load(fused_path)
                    if fused.ndim == 3 and fused.shape[0] > 0:
                        st.markdown("### Fused heatmap volume (Grad-CAM + optical flow + frequency)")
                        st.caption(
                            "**Simple:** one 2D slice of a “fused” anomaly map per time step (inferno colors = stronger). "
                            "**Technical:** combines normalized Grad-CAM with optical-flow error and high-frequency noise maps, then window-fused; read left–right as space, slider as time."
                        )
                        fi = st.slider("Fused frame index", 0, int(fused.shape[0]) - 1, 0, key="dev_fused_slider")
                        ch = _fused_heatmap_slice_chart(fused, fi)
                        if ch is not None:
                            st.altair_chart(ch, use_container_width=True)
                except Exception as e:
                    st.caption(f"Could not load fused heatmap: {e}")

            if isinstance(cam_idx.get("cam_per_frame"), list) and len(cam_idx["cam_per_frame"]) > 0:
                roi_fps = cam_idx.get("roi_fps") or 1.0
                cp = [float(x) for x in cam_idx["cam_per_frame"]]
                cdf = pd.DataFrame(
                    {"t": [float(i) / float(roi_fps) for i in range(len(cp))], "gradcam_intensity": cp}
                )
                st.caption(
                    "**Simple:** line = how “strong” the Grad-CAM map is at each time (higher = more salient activation). "
                    "**Technical:** mean intensity of the CAM tensor over the mouth ROI per frame; x-axis is time in seconds (from ROI fps)."
                )
                st.altair_chart(
                    alt.Chart(cdf).mark_line().encode(x="t:Q", y="gradcam_intensity:Q", tooltip=["t", "gradcam_intensity"]),
                    use_container_width=True,
                )

            perm = res.get("noma_permutation_xai")
            if isinstance(perm, dict) and "error" not in perm and perm.get("feature_names"):
                st.markdown("### NOMA permutation sensitivity (developer)")
                st.caption("Top feature perturbations affecting p(fake) per block (when enabled in Combined).")
                st.json({k: perm[k] for k in ("block_times_seconds", "feature_names", "top_features") if k in perm})

        from explainability.panel_proof import build_panel_proof_bundle, bundle_to_json_bytes

        with st.expander("Panel proof snippets (slides / appendix)", expanded=False):
            st.caption(
                "Copy-paste chain: how fusion math, time-aligned saliency, and Grad-CAM stages relate — "
                "for reviewers, not legal evidence."
            )
            _bundle = build_panel_proof_bundle(res, cam_idx if isinstance(cam_idx, dict) else None)
            st.markdown(_bundle.get("markdown_appendix", ""))
            st.download_button(
                "Download panel_proof_bundle.json",
                data=bundle_to_json_bytes(_bundle),
                file_name="panel_proof_bundle.json",
                mime="application/json",
                key="download_panel_proof_bundle",
            )

        with st.expander("Reviewer figures (PNG)", expanded=False):
            st.markdown(
                """
**What this block is for.** One-click PNGs for slides: same data as above, formatted for a panel.

| Figure | Plain English | Technical (how it is built) |
|--------|----------------|---------------------------|
| **Triptych** | Three panels: raw mouth crop, Grad-CAM overlay, fused heatmap slice — same time index. | Aligns ROI frame `frame_idx` with CAM overlay PNG and fused `.npy` slice from `figure_triptych_png_bytes`. |
| **Mel + NOMA** | Speech “fingerprint” (frequency vs time) with NOMA’s fake probability curve. | Librosa mel spectrogram of extracted WAV + `p_fake` per second overlaid (`figure_mel_noma_png_bytes`). |
| **CMID** | How well audio and video embeddings agree over time. | Cosine similarity per step between AVH audio/visual tensors; CMID curve = deviation from running median (`figure_cmid_png_bytes`). Needs embedding dump. |
| **Attention vs Grad-CAM** | Compares transformer attention weights to CAM intensity (if captured). | Plots two time series from Grad-CAM run with `capture_attention`. |
| **Calibration** | Diagram of how raw scores map to probability. | Reads `calibration_artifacts.json` (temperature / bias) for AVH and NOMA. |

Built by `explainability/reviewer_figures.py` (same as `python tools/reviewer_figures.py`).
                """
            )
            try:
                from explainability.reviewer_figures import (
                    figure_attention_cam_png_bytes,
                    figure_calibration_png_bytes,
                    figure_cmid_png_bytes,
                    figure_mel_noma_png_bytes,
                    figure_triptych_png_bytes,
                )
            except ImportError as e:
                st.warning(f"Install optional deps for figures (e.g. matplotlib, opencv): {e}")
            else:
                roi_p = res.get("roi_path")
                fused_p = cam_idx.get("fused_heatmap_path") if isinstance(cam_idx, dict) else None
                cam_vol = cam_idx.get("cam_volume_path") if isinstance(cam_idx, dict) else None
                odir = None
                if isinstance(cam_idx, dict):
                    odir = cam_idx.get("overlay_dir")
                if not odir or not (isinstance(odir, str) and os.path.isdir(odir)):
                    odir = res.get("cam_overlays_dir")

                if isinstance(roi_p, str) and os.path.isfile(roi_p) and isinstance(cam_idx, dict):
                    tmax = int(cam_idx.get("T_use") or cam_idx.get("T_roi") or 1) - 1
                    tmax = max(0, tmax)
                    tri_t = st.slider("Triptych frame index", 0, tmax, 0, key="reviewer_tri_t")
                    if st.button("Render triptych PNG", key="btn_triptych"):
                        try:
                            tri_b = figure_triptych_png_bytes(
                                roi_path=roi_p,
                                frame_idx=int(tri_t),
                                fused_npy_path=fused_p if isinstance(fused_p, str) else None,
                                cam_npy_path=cam_vol if isinstance(cam_vol, str) else None,
                                overlay_dir=odir if isinstance(odir, str) else None,
                            )
                            st.session_state["reviewer_triptych"] = tri_b
                        except Exception as e:
                            st.error(str(e))
                    if st.session_state.get("reviewer_triptych"):
                        st.image(st.session_state["reviewer_triptych"], caption="ROI | Grad-CAM | fused")
                        st.download_button(
                            "Download triptych.png",
                            data=st.session_state["reviewer_triptych"],
                            file_name="triptych.png",
                            mime="image/png",
                            key="dl_triptych",
                        )
                else:
                    st.caption("Triptych needs `mouth_roi.mp4` and Grad-CAM index in session.")

                ap = res.get("audio_path")
                ndf = res.get("noma_df")
                if isinstance(ap, str) and os.path.isfile(ap) and isinstance(ndf, pd.DataFrame):
                    if {"Seconds", "p_fake"}.issubset(ndf.columns) and st.button("Render mel + NOMA PNG", key="btn_mel"):
                        try:
                            st.session_state["reviewer_mel"] = figure_mel_noma_png_bytes(
                                audio_path=ap,
                                seconds=ndf["Seconds"].values,
                                p_fake=ndf["p_fake"].values,
                            )
                        except Exception as e:
                            st.error(str(e))
                    if st.session_state.get("reviewer_mel"):
                        st.image(st.session_state["reviewer_mel"], caption="Mel spectrogram + NOMA p(fake)")
                        st.download_button(
                            "Download mel_noma.png",
                            data=st.session_state["reviewer_mel"],
                            file_name="mel_noma.png",
                            mime="image/png",
                            key="dl_mel",
                        )

                cmid_d = res.get("cmid")
                if isinstance(cmid_d, dict) and (cmid_d.get("similarity") or cmid_d.get("cmid")):
                    if st.button("Render CMID plot", key="btn_cmid"):
                        try:
                            st.session_state["reviewer_cmid"] = figure_cmid_png_bytes(cmid_d)
                        except Exception as e:
                            st.error(str(e))
                    if st.session_state.get("reviewer_cmid"):
                        st.image(st.session_state["reviewer_cmid"], caption="Cosine similarity & CMID")
                        st.download_button(
                            "Download cmid.png",
                            data=st.session_state["reviewer_cmid"],
                            file_name="cmid.png",
                            mime="image/png",
                            key="dl_cmid",
                        )
                else:
                    st.caption("CMID plot appears when AVH embeddings are dumped (Combined with embedding export).")

                if (
                    isinstance(cam_idx, dict)
                    and isinstance(cam_idx.get("attention_per_frame"), list)
                    and isinstance(cam_idx.get("cam_per_frame"), list)
                ):
                    if st.button("Render attention vs Grad-CAM", key="btn_attn"):
                        try:
                            st.session_state["reviewer_attn"] = figure_attention_cam_png_bytes(cam_idx)
                        except Exception as e:
                            st.error(str(e))
                    if st.session_state.get("reviewer_attn"):
                        st.image(st.session_state["reviewer_attn"], caption="Attention vs CAM")
                        st.download_button(
                            "Download attention_cam.png",
                            data=st.session_state["reviewer_attn"],
                            file_name="attention_cam.png",
                            mime="image/png",
                            key="dl_attn",
                        )
                else:
                    st.caption("Attention trace needs Grad-CAM with `capture_attention` enabled.")

                if st.button("Render calibration diagram", key="btn_cal"):
                    try:
                        st.session_state["reviewer_cal"] = figure_calibration_png_bytes()
                    except Exception as e:
                        st.error(str(e))
                if st.session_state.get("reviewer_cal"):
                    st.image(st.session_state["reviewer_cal"], caption="AVH & NOMA calibration maps")
                    st.download_button(
                        "Download calibration.png",
                        data=st.session_state["reviewer_cal"],
                        file_name="calibration.png",
                        mime="image/png",
                        key="dl_cal",
                    )

    with tab_res:
        from integrations.research_chat.chat_orchestrator import (
            format_detection_context_from_combined,
            run_research_turn,
        )

        st.markdown("### Combine external evidence with your models")
        st.caption(
            "Same pipeline as **Research chat**: SerpAPI organic web, **SerpAPI Google Lens**, NewsAPI headlines, then Gemini. "
            "Attach the latest **Combined** scores as optional context."
        )
        include_models = st.checkbox(
            "Include Combined model scores + XAI status in Gemini context",
            value=True,
            key="integrated_include_models",
        )
        det_ctx = None
        if include_models:
            det_ctx = format_detection_context_from_combined(res, cam_idx)

        q_default = (
            "Summarize what is known about deepfake detection limits for short social clips, "
            "and how lip-sync scores should be interpreted vs audio-only scores."
        )
        q_headline = (
            "Search the headline / topic: summarize **only** headlines and sources returned by Serp (web), "
            "Google Lens, and News API below. Quote titles; do not invent URLs."
        )
        if "integrated_research_prompt" not in st.session_state:
            st.session_state["integrated_research_prompt"] = q_default
        prompt = st.text_area(
            t("research_chat_placeholder", lang),
            height=120,
            key="integrated_research_prompt",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Use suggested: detection limits", key="integrated_btn_sug1"):
                st.session_state["integrated_research_prompt"] = q_default
                st.rerun()
        with c2:
            if st.button("Use suggested: verify headline", key="integrated_btn_sug2"):
                st.session_state["integrated_research_prompt"] = q_headline
                st.rerun()

        show_src = st.checkbox(t("research_show_sources", lang), value=True, key="integrated_show_src")

        if st.button("Run Serp + Lens + News + Gemini", type="primary", key="integrated_run_research"):
            with st.spinner("Fetching sources and synthesizing…"):
                turn = run_research_turn(prompt, detection_context=det_ctx, history=[])
            reply = (turn.text or turn.error or "No response.").strip()
            st.session_state["integrated_last_turn"] = {
                "reply": reply,
                "sources_used": turn.sources_used,
                "error": turn.error,
            }

        last = st.session_state.get("integrated_last_turn")
        if isinstance(last, dict) and last.get("reply"):
            st.markdown("#### Gemini answer")
            st.markdown(last["reply"])
            if last.get("error"):
                st.error(str(last["error"]))

        if isinstance(last, dict) and last.get("sources_used") and show_src:
            su = last["sources_used"]
            st.markdown("#### Sources and headlines")
            render_missing_data_api_keys_hint(su.get("errors"))

            def _bullets(rows: list[Any] | None, title: str) -> None:
                st.markdown(f"**{title}**")
                if not rows:
                    st.caption("(none)")
                    return
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    tt = (r.get("title") or "").strip() or "(no title)"
                    lk = (r.get("link") or "").strip()
                    sn = (r.get("snippet") or "").strip()
                    line = f"- [{tt}]({lk})" if lk else f"- {tt}"
                    st.markdown(line)
                    if sn:
                        st.caption(sn[:280] + ("…" if len(sn) > 280 else ""))

            _bullets(su.get("serp"), "Web (SerpAPI organic)")
            _bullets(su.get("google_lens"), "Google Lens (SerpAPI)")
            _bullets(su.get("news"), "News (NewsAPI)")

            st.markdown("#### Raw rows (developer)")
            s1, s2, s3 = st.columns(3)
            with s1:
                if su.get("serp"):
                    st.dataframe(pd.DataFrame(su["serp"]), use_container_width=True, hide_index=True)
            with s2:
                if su.get("google_lens"):
                    st.dataframe(pd.DataFrame(su["google_lens"]), use_container_width=True, hide_index=True)
            with s3:
                if su.get("news"):
                    st.dataframe(pd.DataFrame(su["news"]), use_container_width=True, hide_index=True)

        with st.expander(t("research_env_expander", lang), expanded=False):
            st.markdown(t("research_env_md", lang))

    # Export (same JSON as before, shown after tabs)
    inst = res.get("noma_confidence_instability")
    export_payload = {
        "verdict": verdict,
        "avh_score": avh_score,
        "avh_p_fake": avh_p_fake,
        "noma_mean_p_fake": noma_mean_p_fake,
        "blended_p_fake": blended_p_fake,
        "p_fused": float(p_fused_res) if p_fused_res is not None else None,
        "fusion_tension": float(fusion_tension) if fusion_tension is not None else None,
        "fusion_tau": float(fusion_tau) if fusion_tau is not None else None,
        "fusion_verdict": fusion_verdict,
        "cmid_status": res.get("cmid_status"),
        "cii": (float(inst.get("CII")) if isinstance(inst, dict) and inst.get("CII") is not None else None),
        "temporal_corroboration": res.get("temporal_corroboration"),
    }
    st.download_button(
        "Download final_combined_report.json",
        data=json.dumps(export_payload, indent=2).encode("utf-8"),
        file_name="final_combined_report.json",
        mime="application/json",
        key="download_final_combined_report",
    )
