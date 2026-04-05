"""
Unified Streamlit App: multimodal deepfake screening (audio, video + visual frames)
via NOMA (audio) and AVH-Align (audio-visual), plus optional multilingual UI and research chat.
"""

import io
import os
import re
import sys
import tempfile
from datetime import datetime, timezone

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

import streamlit as st
import pandas as pd
import altair as alt

from config import (
    PROJECT_ROOT,
    get_late_fusion_mode,
    NOMA_MODEL_CANDIDATES,
    AVH_DIR,
    AVH_TEST_SCRIPT,
    AVH_TEST_UNSUP_SCRIPT,
    AVH_FUSION_CKPT,
    AVH_AVHUBERT_DIR,
    AVH_FACE_PREDICTOR,
    AVH_MEAN_FACE,
    AVH_AVHUBERT_CKPT,
    AVH_GRADCAM_SCRIPT,
    STREAMLIT_GRADCAM_TOP_K,
    STREAMLIT_GRADCAM_MAX_FUSION_FRAMES,
    STREAMLIT_GRADCAM_SELECTION_MODE,
    STREAMLIT_GRADCAM_MIN_TEMPORAL_GAP,
    STREAMLIT_GRADCAM_REGION_TRACK_STRIDE,
)

from detectors.avh_align import (
    check_avh_setup,
    run_avh_from_npz,
    run_avh_on_video,
    run_avh_unsupervised_on_video,
)
from detectors.noma import (
    get_noma_model_path,
    get_noma_pipeline,
    noma_p_fake_raw_confidence_and_preds_from_probas,
    run_noma_prediction,
    run_noma_prediction_with_features,
)
from evidence.exporter import zip_evidence_bundle
from explainability.gradcam_avh import run_gradcam_mouth_roi
from orchestrator.combined_runner import run_combined_avh_to_noma
from ui.streamlit_css import STREAMLIT_CUSTOM_CSS
from ui.i18n import t
from ui.video_manual_crop import (
    get_saved_manual_rect,
    prepare_video_with_optional_manual_crop,
    render_manual_crop_ui,
)

# ─── Page config & CSS ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detection Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(STREAMLIT_CUSTOM_CSS, unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────
def get_noma_model_path():
    from detectors.noma import get_noma_model_path as _impl
    return _impl()

def run_noma_prediction(
    model_path: str,
    audio_path: str = None,
    audio_bytes: io.BytesIO = None,
    audio_filename: str | None = None,
):
    from detectors.noma import run_noma_prediction as _impl
    return _impl(
        model_path,
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
    )

def _sha256_file(path: str) -> str:
    from evidence.exporter import _sha256_file as _impl
    return _impl(path)

def _zip_evidence_bundle(
    *,
    input_video_path: str,
    input_video_name: str,
    avh_score: float | None,
    audio_path: str,
    roi_path: str,
    cam_idx: dict | None,
    overlays_dir: str | None,
    noma_df: pd.DataFrame | None,
) -> bytes:
    """
    Create an evidence zip bundle (bytes) suitable for download:
    - hashes (sha256)
    - scores
    - extracted audio + mouth ROI
    - Grad-CAM overlays + index.json
    - NOMA per-second predictions (csv)
    """
    return zip_evidence_bundle(
        input_video_path=input_video_path,
        input_video_name=input_video_name,
        avh_score=avh_score,
        audio_path=audio_path,
        roi_path=roi_path,
        cam_idx=cam_idx,
        overlays_dir=overlays_dir,
        noma_df=noma_df,
    )


def check_avh_setup():
    """Return list of (check_name, ok, detail) for AVH setup."""
    from detectors.avh_align import check_avh_setup as _impl
    return _impl()


def run_avh_on_video(
    video_path: str,
    timeout: int = 300,
    python_exe: str = None,
    keep_temp: bool = False,
    smart_crop: str = "auto",
):
    """Run AVH test_video.py on a video file.
    Returns (success, score_or_error_message) or (success, score, audio_path) if keep_temp=True.
    python_exe: use this Python (e.g. conda env where AVH works) instead of current one.
    """
    from detectors.avh_align import run_avh_on_video as _impl
    return _impl(
        video_path=video_path,
        timeout=timeout,
        python_exe=python_exe,
        keep_temp=keep_temp,
        smart_crop=smart_crop,
    )


def run_avh_unsupervised_on_video(
    video_path: str,
    timeout: int = 300,
    python_exe: str = None,
    keep_temp: bool = False,
    smart_crop: str = "auto",
):
    """
    Run unsupervised AVH scoring (no training) on a video.
    Returns (success, score_or_error) or (success, score, audio_path) if keep_temp=True.
    """
    from detectors.avh_align import run_avh_unsupervised_on_video as _impl
    return _impl(
        video_path=video_path,
        timeout=timeout,
        python_exe=python_exe,
        keep_temp=keep_temp,
        smart_crop=smart_crop,
    )


def run_gradcam_mouth_roi(
    video_path: str,
    python_exe: str,
    top_k: int = STREAMLIT_GRADCAM_TOP_K,
    adv_ckpt: str = None,
    roi_path: str = None,
    audio_path: str = None,
    timeout: int = 300,
    keep_temp: bool = False,
):
    """
    Runs AVH/gradcam_mouth_roi.py and returns:
    - (ok, overlays_dir_or_error, index_json_or_none)
    """
    from explainability.gradcam_avh import run_gradcam_mouth_roi as _impl
    return _impl(
        video_path=video_path,
        python_exe=python_exe,
        top_k=top_k,
        adv_ckpt=adv_ckpt,
        roi_path=roi_path,
        audio_path=audio_path,
        timeout=timeout,
        keep_temp=keep_temp,
    )


def run_avh_from_npz(npz_bytes: bytes, fusion_ckpt_path: str):
    """Score using only the Fusion checkpoint and pre-extracted .npz (no av_hubert/dlib).
    .npz must have keys 'visual' and 'audio' (arrays shape (T, 1024)). Returns (success, score_or_error).
    """
    from detectors.avh_align import run_avh_from_npz as _impl
    return _impl(npz_bytes=npz_bytes, fusion_ckpt_path=fusion_ckpt_path)


def _show_avh_score_or_error(ok, result, *, use_unsup_avh: bool = False):
    """Render AVH score or error in Streamlit (reused for video and .npz paths)."""
    if ok:
        score = result
        from calibration_runtime import avh_score_to_calibrated_p_fake, get_uncertainty_margins

        p_fake = avh_score_to_calibrated_p_fake(float(score), use_unsup_avh=use_unsup_avh)
        avh_margin, _ = get_uncertainty_margins()

        if p_fake >= 0.5 + avh_margin:
            score_bg, score_border, score_label = "#fee2e2", "#ef4444", "Likely FAKE"
        elif p_fake <= 0.5 - avh_margin:
            score_bg, score_border, score_label = "#dcfce7", "#22c55e", "Likely REAL"
        else:
            score_bg, score_border, score_label = "#fef9c3", "#eab308", "Uncertain"

        st.markdown("#### 🎯 Deepfake score")
        st.markdown(f"""
        <div style="padding:1.5rem; border-radius:12px; background:{score_bg}; border:3px solid {score_border}; text-align:center; margin:0.5rem 0;">
            <div style="font-size:2rem; font-weight:800; color:#0f172a;">{score:.4f}</div>
            <div style="font-size:0.95rem; font-weight:600; color:#475569; margin-top:0.25rem;">{score_label}</div>
            <div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">p(fake)={p_fake:.3f} ({'unsupervised' if use_unsup_avh else 'supervised'} calibration)</div>
        </div>
        """, unsafe_allow_html=True)
        if score_label == "Likely REAL":
            st.success("Calibration suggests **real** (lip–speech appears aligned).")
        elif score_label == "Likely FAKE":
            st.error("Calibration suggests **deepfake** (lip–speech mismatch).")
        else:
            st.info("Calibration is uncertain (near decision boundary); consider human review.")
    else:
        st.error("AVH pipeline failed. Check **🔧 AVH setup status** or use **Score from .npz** with a pre-extracted file.")
        with st.expander("Error details", expanded=True):
            st.text(result)


# ─── Sidebar: module navigation ────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ Deepfake Detection Lab")
lang = st.sidebar.selectbox(
    "Interface language / Idioma / भाषा",
    options=["en", "es", "hi"],
    format_func=lambda c: {"en": "English", "es": "Español", "hi": "हिन्दी"}[c],
    key="ui_lang",
)
st.sidebar.markdown(t("sidebar_blurb", lang))
page = st.sidebar.radio(
    "Select module",
    [
        "Home / Overview",
        "Dataset Explorer",
        "Preprocessing",
        "Feature Extraction",
        "Model & Training",
        "Inference Demo",
        "Video Explainability",
        "Audio Explainability",
        "Fusion & Custom Algorithms",
        "Evaluation & Metrics",
        "Final Combined Report",
        "Fact check (STT + Serp + News)",
        "Research chat",
    ],
    index=0,
)
use_unsup_avh = st.sidebar.checkbox(
    "Use unsupervised AVH (no training)",
    value=not os.path.isfile(AVH_FUSION_CKPT),
    help=(
        "Zero-shot lip–audio mismatch score (no Fusion checkpoint training). "
        "Uses a separate p(fake) mapping from supervised AVH — see calibration_runtime.avh_unsupervised_score_to_p_fake."
    ),
)
# Resolved checkpoint paths (what the app uses)
_noma_path = get_noma_model_path()
with st.sidebar.expander("📁 Model paths", expanded=False):
    st.markdown("**NOMA (audio)**")
    st.code(_noma_path or "— not found —", language=None)
    st.markdown("**AVH-Align (fusion)**")
    st.code(AVH_FUSION_CKPT, language=None)
    if _noma_path:
        st.caption(f"NOMA: {'✅ found' if os.path.isfile(_noma_path) else '❌ missing'}")
    st.caption(f"AVH: {'✅ found' if os.path.isfile(AVH_FUSION_CKPT) else '❌ missing'}")

# Python for AVH video pipeline — MUST use avh conda env (has correct omegaconf, fairseq, etc.)
def _find_avh_python():
    candidates = [
        "/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python",
        os.path.expanduser("~/miniforge3/envs/avh/bin/python"),
        os.path.expanduser("~/miniconda3/envs/avh/bin/python"),
        os.path.expanduser("~/anaconda3/envs/avh/bin/python"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

_avh_default = _find_avh_python()
avh_python_path = st.sidebar.text_input(
    "**Python for AVH video** (required for video upload)",
    value=_avh_default or "",
    placeholder="/path/to/conda/envs/avh/bin/python",
    help="Must point to the avh conda env Python. The venv Python has wrong omegaconf and will fail.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Runtime readiness**")
_noma_ok = bool(_noma_path and os.path.isfile(_noma_path))
st.sidebar.caption(f"{'✅' if _noma_ok else '❌'} NOMA model artifact")
_avh_script_ok = os.path.isfile(AVH_TEST_SCRIPT)
_avh_fusion_ok = os.path.isfile(AVH_FUSION_CKPT)
_avh_hub_ok = os.path.isfile(AVH_AVHUBERT_CKPT)
st.sidebar.caption(
    f"{'✅' if _avh_script_ok else '❌'} AVH `test_video.py` · "
    f"{'✅' if _avh_fusion_ok else '❌'} fusion ckpt · "
    f"{'✅' if _avh_hub_ok else '❌'} AV-HuBERT ckpt"
)
_py = (avh_python_path or "").strip()
_py_ok = bool(_py and os.path.isfile(_py))
st.sidebar.caption(f"{'✅' if _py_ok else '❌'} AVH Python path set and file exists")
_lf_opts = ["full", "mean", "audio_primary", "video_primary", "learned"]
_env_lf = get_late_fusion_mode()
_lf_default_idx = _lf_opts.index(_env_lf) if _env_lf in _lf_opts else 0
with st.sidebar.expander("Combined pipeline", expanded=True):
    combined_late_fusion_mode = st.selectbox(
        "Late fusion mode",
        options=_lf_opts,
        index=_lf_default_idx,
        key="sidebar_combined_late_fusion",
        help=(
            "**full** = reliability fusion (regime blend). Applies to **Inference Demo → Combined** runs. "
            "Overrides `LATE_FUSION_MODE` for this session only."
        ),
    )
with st.sidebar.expander("Readiness details", expanded=False):
    try:
        for name, ok, detail in check_avh_setup():
            st.markdown(f"- **{name}:** {'OK' if ok else 'Missing'} — `{detail}`")
    except Exception as e:
        st.caption(f"Could not run full AVH check: {e}")

# Session-state defaults for cross-page teaching/XAI flows.
if "last_combined_res" not in st.session_state:
    st.session_state["last_combined_res"] = None
if "last_cam_idx" not in st.session_state:
    st.session_state["last_cam_idx"] = None
if "research_chat_messages" not in st.session_state:
    st.session_state["research_chat_messages"] = []


def _streamlit_combined_persist_dir(video_name: str) -> str:
    """Stable folder under eval_runs/streamlit_combined/ for Combined artifacts."""
    safe = re.sub(r"[^\w.\-]+", "_", os.path.basename(video_name or "video").strip())[:80] or "video"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(PROJECT_ROOT, "eval_runs", "streamlit_combined")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{ts}_{safe}")


def _list_videos_in_dir(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    out: list[str] = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
            out.append(name)
    return out


def _safe_video_path_under_folder(folder: str, basename: str) -> str | None:
    """Resolve a file under folder only (no path traversal)."""
    try:
        root = os.path.abspath(os.path.realpath(folder))
        cand = os.path.abspath(os.path.join(root, basename))
        if os.path.commonpath([root, cand]) != root:
            return None
    except (OSError, ValueError):
        return None
    return cand if os.path.isfile(cand) else None


def _render_combined_limitations_expander(*, use_unsup_avh: bool) -> None:
    """Shared copy for Combined results and Final Combined Report."""
    with st.expander("Limitations and when to trust this", expanded=False):
        st.markdown(
            """
            - **Not legal or sole proof** — screening / research aid only; expert review may still be required.
            - **Face and audio quality** — poor lighting, heavy compression, small faces, or missing lips can skew AVH and Grad-CAM.
            - **Dubbing / mismatch** — dialogue replaced after filming can look like a “fake” to lip–speech models even when the video is authentic.
            - **Clip length** — shorter clips (e.g. under ~60s) are easier on CPU and RAM; long files take longer and may hit timeouts.
            """
        )
        if use_unsup_avh:
            st.caption(
                "Unsupervised AVH is enabled: scores reflect zero-shot lip–audio mismatch, not the same training as supervised AVH-Align."
            )
        else:
            st.caption(
                "Supervised AVH-Align uses the fusion checkpoint; ensure checkpoints match the intended evaluation setting."
            )


def _render_combined_demo_summary_from_res(res: dict) -> None:
    """Headline metrics after a successful Combined run (uses session result dict)."""
    from calibration_runtime import avh_score_to_calibrated_p_fake
    from config import get_late_fusion_mode

    avh_score = res.get("avh_score")
    avh_p_fake = res.get("p_avh_cal")
    if avh_p_fake is None and isinstance(avh_score, (int, float)):
        avh_p_fake = float(
            avh_score_to_calibrated_p_fake(
                float(avh_score),
                use_unsup_avh=bool(res.get("use_unsup_avh")),
            )
        )
    noma_df = res.get("noma_df")
    noma_mean_p_fake = None
    if isinstance(noma_df, pd.DataFrame) and "p_fake" in noma_df.columns and len(noma_df) > 0:
        noma_mean_p_fake = float(noma_df["p_fake"].astype(float).mean())
    p_fused = res.get("p_fused")
    fusion_tension = res.get("fusion_tension")
    fusion_verdict = res.get("fusion_verdict")
    fusion_tau = res.get("fusion_tau")
    late_mode = res.get("late_fusion_mode") or get_late_fusion_mode()

    with st.container(border=True):
        st.markdown(
            '<p style="margin:0 0 0.5rem 0;font-weight:600;">Combined demo summary</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Late fusion (NOMA + AVH) mode: **{late_mode}** — set `LATE_FUSION_MODE` env to "
            "`full` (default), `mean`, `audio_primary`, or `video_primary`."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("AVH p(fake)", f"{avh_p_fake:.3f}" if avh_p_fake is not None else "n/a")
        with c2:
            st.metric("NOMA mean p(fake)", f"{noma_mean_p_fake:.3f}" if noma_mean_p_fake is not None else "n/a")
        with c3:
            st.metric("Late-fused p(fake)", f"{float(p_fused):.3f}" if p_fused is not None else "n/a")
        with c4:
            st.metric(
                "Fusion tension",
                f"{float(fusion_tension):.3f}" if fusion_tension is not None else "n/a",
            )
        with c5:
            st.metric(
                "Fusion verdict",
                str(fusion_verdict) if fusion_verdict else "n/a",
            )
        if p_fused is not None and fusion_tau is not None:
            if late_mode == "full":
                st.caption(
                    "Reliability-weighted blend of calibrated AVH and mean NOMA p(fake); higher **tension** "
                    "(|AVH − NOMA|) down-weights NOMA in the default blend. "
                    f"Verdict bands use τ={float(fusion_tau):.3f} from calibration margins."
                )
            elif late_mode == "mean":
                st.caption(
                    f"**Simple mean:** p_fused = average of calibrated AVH and mean NOMA. "
                    f"Verdict bands use τ={float(fusion_tau):.3f}."
                )
            elif late_mode == "audio_primary":
                st.caption(
                    f"**NOMA-primary:** p_fused follows mean NOMA p(fake). "
                    f"Verdict bands use τ={float(fusion_tau):.3f}."
                )
            else:
                st.caption(
                    f"**AVH-primary:** p_fused follows calibrated AVH p(fake). "
                    f"Verdict bands use τ={float(fusion_tau):.3f}."
                )
        else:
            st.caption("Late fusion metrics populate after NOMA and calibration run in the same pipeline pass.")


def _render_home_overview(lang: str) -> None:
    st.markdown('<p class="main-header">🛡️ Deepfake Detection Lab</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{t("home_subheader", lang)}</p>', unsafe_allow_html=True)
    st.info(t("home_recommended", lang))
    st.caption(t("home_disclaimer", lang))
    st.markdown(
        f"""
{t("home_quickstart_title", lang)}
1. {t("home_q1", lang)}
2. {t("home_q2", lang)}
3. {t("home_q3", lang)}
        """
    )
    st.markdown(t("home_modality_note", lang))
    with st.expander(t("home_expander_compare", lang), expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
            <div style="padding:1rem; border-radius:10px; background:linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%); border:2px solid #3b82f6;">
                <div style="font-size:1.2rem; font-weight:700; color:#1e40af; margin-bottom:0.5rem;">{t("home_noma_title", lang)}</div>
                <div style="font-size:0.85rem; color:#1e3a8a;">{t("home_noma_sub", lang)}</div>
                <ul style="margin:0.5rem 0 0 1rem; font-size:0.9rem;">
                    <li>{t("home_noma_li1", lang)}</li>
                    <li>{t("home_noma_li2", lang)}</li>
                    <li>{t("home_noma_li3", lang)}</li>
                    <li>{t("home_noma_li4", lang)}</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
            <div style="padding:1rem; border-radius:10px; background:linear-gradient(180deg, #fef3c7 0%, #fde68a 100%); border:2px solid #f59e0b;">
                <div style="font-size:1.2rem; font-weight:700; color:#b45309; margin-bottom:0.5rem;">{t("home_avh_title", lang)}</div>
                <div style="font-size:0.85rem; color:#92400e;">{t("home_avh_sub", lang)}</div>
                <ul style="margin:0.5rem 0 0 1rem; font-size:0.9rem;">
                    <li>{t("home_avh_li1", lang)}</li>
                    <li>{t("home_avh_li2", lang)}</li>
                    <li>{t("home_avh_li3", lang)}</li>
                    <li>{t("home_avh_li4", lang)}</li>
                    <li>{t("home_avh_li5", lang)}</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
        {t("home_table_header", lang)}
        {t("home_table_sep", lang)}
        {t("home_table_r1", lang)}
        {t("home_table_r2", lang)}
        {t("home_table_r3", lang)}
        {t("home_table_r4", lang)}
        """
        )


def _render_teaching_block(problem: str, approach: str, algorithm: str, code_ref: str, output_notes: str, limitations: str):
    with st.expander("Build-from-scratch view", expanded=False):
        st.markdown(f"**Problem:** {problem}")
        st.markdown(f"**Approach:** {approach}")
        st.markdown(f"**Algorithm:** {algorithm}")
        st.markdown(f"**Code path:** `{code_ref}`")
        st.markdown(f"**Output:** {output_notes}")
        st.markdown(f"**Limitations:** {limitations}")


def _render_dataset_explorer() -> None:
    st.markdown("## Dataset Explorer")
    _render_teaching_block(
        "Understand class balance and metadata quality before model runs.",
        "Load AV1M metadata CSVs and inspect label distribution + samples.",
        "Count labels and show top rows for train/val/test.",
        "AVH/av1m_metadata/*.csv, AVH/avh_sup/csv_metadata/*",
        "Split-level sample tables and class histograms.",
        "Metadata quality depends on available CSV columns.",
    )
    import pandas as _pd
    import numpy as _np
    splits = ["train", "val", "test"]
    for split in splits:
        p = os.path.join(PROJECT_ROOT, "AVH", "av1m_metadata", f"{split}_metadata.csv")
        if os.path.isfile(p):
            df = _pd.read_csv(p)
            st.markdown(f"### {split.title()} metadata")
            st.caption(f"{len(df)} rows from `{p}`")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            if "label" in df.columns:
                dist = df["label"].value_counts().rename_axis("label").reset_index(name="count")
                chart = alt.Chart(dist).mark_bar().encode(x="label:N", y="count:Q", tooltip=["label", "count"])
                st.altair_chart(chart, use_container_width=True)
                probs = dist["count"].values / max(1, dist["count"].values.sum())
                entropy = float(-(probs * _np.log2(_np.clip(probs, 1e-12, 1.0))).sum())
                st.caption(f"Class entropy H(y) = -Σ p(y)log2 p(y) = {entropy:.4f} bits")
            missing = int(df.isna().sum().sum())
            st.caption(f"Missing cells in table: {missing}")
        else:
            st.info(f"Metadata not found: `{p}`")


def _render_preprocessing_page() -> None:
    st.markdown("## Preprocessing Pipeline")
    _render_teaching_block(
        "Inspect raw vs processed media transformations.",
        "AVH preprocess extracts mouth ROI and audio; NOMA normalizes audio blocks.",
        "Face detect -> landmarks -> ROI crop; ffmpeg extract audio.",
        "AVH/test_video.py preprocess_video(), detectors/noma.py decode+split",
        "Pipeline steps and expected artifacts (`mouth_roi.mp4`, `audio.wav`).",
        "Interactive raw-vs-processed visuals require a run artifact.",
    )
    st.markdown("""
    - AVH preprocessing outputs:
      - `mouth_roi.mp4` (cropped mouth sequence)
      - `audio.wav` (extracted mono audio)
    - NOMA preprocessing:
      - decode input
      - resample to 22.05kHz
      - split into 1-second blocks
    """)
    st.markdown("**Math snippets**")
    st.latex(r"x_{resampled}(n) = x\left(n \cdot \frac{f_{orig}}{f_{target}}\right)")
    st.latex(r"X(m,k)=\sum_{n=0}^{N-1} x(n+mH)w(n)e^{-j2\pi kn/N}")
    st.latex(r"\hat{x}=\frac{x-\mu}{\sigma+\epsilon}")


def _render_feature_extraction_page() -> None:
    st.markdown("## Feature Extraction")
    _render_teaching_block(
        "Show intermediate representations used by models.",
        "Expose NOMA 41-D handcrafted features and AVH 1024-D embeddings.",
        "NOMA: spectral+cepstral stats; AVH: audio/visual embedding extraction.",
        "detectors/noma.py _extract_features_from_array(), AVH/dump_avh_features.py",
        "Feature schemas and expected shapes for debugging.",
        "AVH embeddings require extra dump step per input clip.",
    )
    st.markdown("### NOMA feature schema (41 dims)")
    st.code(
        '["Chroma","RMS","Centroid","Bandwidth","Rolloff","ZCR","Tonnetz","Contrast"] + '
        '[f"MFCC{i}" for i in range(1,21)] + [f"IMFCC{i}" for i in range(1,14)]'
    )
    st.markdown("**Math snippets**")
    st.latex(r"\mathrm{MFCC}=\mathrm{DCT}(\log(\mathrm{MelFilterBank}(|\mathrm{STFT}(x)|^2)))")
    st.latex(r"\mathrm{IMFCC}\approx \mathrm{DCT}(\log(\mathrm{InvMelFilterBank}(|\mathrm{STFT}(x)|^2)))")
    st.latex(r"e_t^a, e_t^v \in \mathbb{R}^{1024}")

    noma_model_path = get_noma_model_path()
    feat_upload = st.file_uploader(
        "Optional: upload audio to inspect first feature vector",
        type=["wav", "mp3", "ogg"],
        key="feat_audio_upload",
    )
    if feat_upload is not None and noma_model_path:
        try:
            _, _, feature_matrix, feature_names = run_noma_prediction_with_features(
                noma_model_path,
                audio_bytes=io.BytesIO(feat_upload.getvalue()),
                audio_filename=feat_upload.name,
            )
            if feature_matrix is not None and len(feature_matrix) > 0:
                first = pd.DataFrame(
                    {"feature": list(feature_names), "value": feature_matrix[0].tolist()}
                )
                st.dataframe(first, use_container_width=True, hide_index=True)
                topk = first.reindex(first["value"].abs().sort_values(ascending=False).head(8).index)
                st.altair_chart(
                    alt.Chart(topk).mark_bar().encode(x="feature:N", y="value:Q", tooltip=["feature", "value"]),
                    use_container_width=True,
                )
        except Exception as e:
            st.warning(f"Feature preview failed: {e}")


def _render_model_training_page() -> None:
    st.markdown("## Model & Training")
    _render_teaching_block(
        "Explain model architecture and training paths.",
        "Summarize NOMA SVM and AVH FusionModel + AV-HuBERT flow.",
        "NOMA: L2 norm + RBF SVC; AVH: audio/visual proj + MLP fusion.",
        "detectors/noma.py, AVH/model.py, AVH/train.py, AVH/avh_sup/*",
        "Architecture and training components for demo/debug.",
        "Training logs/metrics may not exist in runtime artifacts.",
    )
    st.markdown("""
    - **NOMA:** sklearn `Pipeline(normalizer -> SVC(probability=True))`
    - **AVH-Align:** AV-HuBERT feature extraction + FusionModel MLP scoring
    """)
    st.markdown("### AV-HuBERT deep dive (research objective vs runtime in this code)")
    st.markdown("**Research objective (pretraining):** masked multimodal unit prediction with pseudo-labels")
    st.latex(r"\mathcal{L} = -\sum_{t\in \mathcal{M}} \log p\left(z_t \mid \mathbf{x}_{\setminus \mathcal{M}}^{a}, \mathbf{x}_{\setminus \mathcal{M}}^{v}\right)")
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")
    st.latex(r"s_t = \cos(e_t^{a}, e_t^{v}) = \frac{e_t^{a}\cdot e_t^{v}}{\|e_t^{a}\|\|e_t^{v}\|}")
    st.markdown(
        """
- **In this repository at inference time:** `AVH/test_video.py` runs preprocessing + feature dump; `AVH/model.py` fusion head maps aligned audio/visual embeddings to a fake score.
- **Not running online here:** AV-HuBERT pretraining loop and cluster target re-generation (those are part of upstream research training pipelines).
- **Calibration layer:** runtime uses `calibration_runtime.py` to map raw detector outputs into calibrated `p(fake)`.
        """
    )


def _render_xai_gemini_expander(*, kind: str, res: dict | None, cam_idx: dict | None) -> None:
    """Gemini walkthrough for Audio / Video Explainability pages."""
    from integrations.research_chat.gemini_client import synthesize_ui_guide
    from ui.report_explain_payload import XAI_SECTION_LABELS, build_xai_standalone_payload

    payload = build_xai_standalone_payload(kind, res, cam_idx)
    sid = "xai_audio" if kind == "audio" else "xai_video"
    title = next(t for t in XAI_SECTION_LABELS if t[0] == sid)[1]
    with st.expander("Gemini: explain this page (plots & metrics)", expanded=False):
        st.caption(
            "Uses **GEMINI_API_KEY** only. Answers use **simple English** plus a **technical** section on how CII / Grad-CAM are computed."
        )
        if st.button("Generate explanation", key=f"gemini_xai_btn_{kind}"):
            with st.spinner("Asking Gemini…"):
                text, err = synthesize_ui_guide(section_id=sid, section_title=title, guide_payload=payload)
            st.session_state[f"xai_gemini_{kind}_text"] = text
            st.session_state[f"xai_gemini_{kind}_err"] = err
        if st.session_state.get(f"xai_gemini_{kind}_err"):
            st.error(st.session_state[f"xai_gemini_{kind}_err"])
        gtxt = st.session_state.get(f"xai_gemini_{kind}_text")
        if gtxt:
            st.markdown(gtxt)


def _render_fact_check_page() -> None:
    """Serp + News + Google Fact Check + Gemini; optional Gemini STT on uploaded audio."""
    from integrations.research_chat.factcheck_turn import run_fact_check_with_optional_stt
    from ui.env_keys_help import render_missing_data_api_keys_hint

    st.markdown("## Fact check (sources + Gemini)")
    st.caption(
        "Type a **short claim** and/or upload **speech audio**. We transcribe with **Gemini** (STT), then query "
        "**SerpAPI** (web), **NewsAPI**, and **Google Fact Check Tools**, and ask **Gemini** to map sources to "
        "support / contradict / unclear. **Not legal proof** — APIs are incomplete and can miss context."
    )
    st.markdown(
        "**Env:** `GEMINI_API_KEY` (required) · `SERPAPI_API_KEY` · `NEWS_API_KEY` · "
        "`GOOGLE_FACT_CHECK_API_KEY` (optional but recommended for ClaimReview)."
    )

    claim = st.text_input("Claim to check (optional if you provide audio only)", key="fc_claim_text", placeholder="e.g. X won the 2024 election in region Y")
    audio_fc = st.file_uploader("Optional: audio clip (speech) — WAV/MP3/OGG", type=["wav", "mp3", "ogg", "m4a"], key="fc_audio")

    c1, c2 = st.columns(2)
    with c1:
        n_serp = st.slider("Serp results", 3, 10, 6, key="fc_n_serp")
    with c2:
        n_news = st.slider("News articles", 3, 20, 10, key="fc_n_news")

    if st.button("Run fact check", type="primary", key="fc_run"):
        ab = audio_fc.getvalue() if audio_fc is not None else None
        an = audio_fc.name if audio_fc is not None else None
        with st.spinner("Transcribing (if audio) + fetching sources + synthesizing…"):
            turn = run_fact_check_with_optional_stt(
                claim_text=claim or None,
                audio_bytes=ab,
                audio_name=an,
                num_serp=n_serp,
                num_news=n_news,
                num_factcheck=8,
            )
        st.session_state["fc_last_turn"] = turn

    turn = st.session_state.get("fc_last_turn")
    if turn is not None:
        if getattr(turn, "stt_error", None):
            st.warning(f"STT note: {turn.stt_error}")
        if getattr(turn, "transcript", None):
            st.markdown("#### Transcript (Gemini STT)")
            st.code(turn.transcript or "", language=None)
        if getattr(turn, "error", None) and not (turn.synthesis or "").strip():
            st.error(turn.error)
        if turn.synthesis:
            st.markdown("#### Gemini verdict (source-grounded)")
            st.markdown(turn.synthesis)
        su = getattr(turn, "sources_used", {}) or {}
        render_missing_data_api_keys_hint(su.get("errors"))
        with st.expander("Sources (raw)", expanded=False):
            if su.get("google_factcheck"):
                st.markdown("**Google Fact Check (ClaimReview)**")
                st.dataframe(pd.DataFrame(su["google_factcheck"]), use_container_width=True, hide_index=True)
            else:
                st.caption("No Fact Check rows — set `GOOGLE_FACT_CHECK_API_KEY` or try a different claim wording.")
            if su.get("serp"):
                st.markdown("**Web (SerpAPI)**")
                st.dataframe(pd.DataFrame(su["serp"]), use_container_width=True, hide_index=True)
            if su.get("news"):
                st.markdown("**News (NewsAPI)**")
                st.dataframe(pd.DataFrame(su["news"]), use_container_width=True, hide_index=True)


def _render_audio_xai_page() -> None:
    st.markdown("## Audio Explainability")
    _render_teaching_block(
        "Explain why a clip looks fake from audio-side evidence.",
        "Use NOMA feature sensitivity + confidence instability.",
        "Permutation sensitivity over 41 features and CII timeline variance.",
        "explainability/noma_feature_sensitivity.py, explainability/instability.py",
        "Feature heatmaps and instability indicators.",
        "Spectrogram attribution/reconstruction maps are best-effort placeholders.",
    )
    st.markdown("**Math snippets**")
    st.latex(r"S(\tau,f)=\left|\sum_n x[n]w[n-\tau]e^{-j2\pi fn}\right|^2")
    st.latex(r"\mathrm{CII}=\frac{1}{T}\sum_t \mathrm{Var}\left(p_{t-k:t+k}\right)")
    st.latex(r"E_{recon}(t)=\|x_t-\hat{x}_t\|_2")

    res = st.session_state.get("last_combined_res")
    if isinstance(res, dict):
        inst = res.get("noma_confidence_instability")
        if isinstance(inst, dict) and isinstance(inst.get("variance_per_time"), list):
            var = inst["variance_per_time"]
            df = pd.DataFrame({"idx": list(range(len(var))), "var": var})
            st.altair_chart(
                alt.Chart(df).mark_line().encode(x="idx:Q", y="var:Q", tooltip=["idx", "var"]),
                use_container_width=True,
            )
            st.caption(f"CII summary: {float(inst.get('CII', 0.0)):.4e}")
    else:
        st.info("Run Inference Demo → Combined first to populate CII and NOMA-side explanations.")
    st.caption("Spectrogram attribution and reconstruction-error overlays are scaffolded; model-specific hooks are pending.")

    _render_xai_gemini_expander(
        kind="audio",
        res=st.session_state.get("last_combined_res"),
        cam_idx=None,
    )


def _render_video_xai_page() -> None:
    st.markdown("## Video Explainability")
    _render_teaching_block(
        "Explain fake cues in visual stream over time and space.",
        "Use Grad-CAM + temporal inconsistency + region tracking + frequency and fused signals.",
        "CAM over ROI frames, IoU region tracks, Δt curve, high-frequency energy, fused anomaly intensity.",
        "AVH/gradcam_mouth_roi.py, explainability/gradcam_avh.py, explainability/video_*",
        "Interactive temporal plots and region summaries from latest run.",
        "Requires a completed Combined run with Grad-CAM enabled.",
    )
    cam_idx = st.session_state.get("last_cam_idx")
    if not isinstance(cam_idx, dict):
        st.info(
            "**No Grad-CAM in this browser session yet.**\n\n"
            "1. Sidebar → **Inference Demo** → **Combined (AVH → NOMA)**.\n"
            "2. Leave **Include mouth ROI Grad-CAM** checked (default).\n"
            "3. Set sidebar **Python for AVH video** to your `avh` conda Python.\n"
            "4. Run Combined and wait for it to finish (Grad-CAM runs after AVH).\n\n"
            "**Supervised vs unsupervised AVH:** both can produce Grad-CAM overlays — the vis path uses the "
            "Fusion + AV-HuBERT stack; your sidebar choice only changes how the **lip–speech score** is computed.\n\n"
            "If Combined shows a **Grad-CAM failed** warning, open it — usually wrong Python env, missing checkpoints, or timeout."
        )
        _render_xai_gemini_expander(
            kind="video",
            res=st.session_state.get("last_combined_res"),
            cam_idx=None,
        )
        return

    st.success("Loaded latest Grad-CAM evidence from session.")
    st.json({k: cam_idx.get(k) for k in ["score", "T_cam_full", "T_roi", "T_use", "roi_fps"]})
    st.markdown("**XAI compute status**")
    xai_status = cam_idx.get("xai_status") if isinstance(cam_idx.get("xai_status"), dict) else {}
    status_payload = {
        "temporal_inconsistency": xai_status.get("temporal_inconsistency", "unknown"),
        "region_tracks": xai_status.get("region_tracks", "unknown"),
        "fusion": xai_status.get("fusion", "unknown"),
        "video_frequency_stats": xai_status.get("video_frequency_stats", "unknown"),
    }
    st.json(status_payload)
    st.latex(r"\Delta_t=\|E_t-E_{t-1}\|_2")
    st.latex(r"H_t=w_1\hat{G}_t+w_2\hat{F}^{flow}_t+w_3\hat{F}^{freq}_t")

    roi_fps = cam_idx.get("roi_fps")
    cam_per = cam_idx.get("cam_per_frame")
    if isinstance(cam_per, list) and len(cam_per) > 0:
        cam_df = pd.DataFrame(
            {
                "t": [float(i) / float(roi_fps) if roi_fps else float(i) for i in range(len(cam_per))],
                "cam": [float(v) for v in cam_per],
            }
        )
        chart = alt.Chart(cam_df).mark_line().encode(x="t:Q", y="cam:Q", tooltip=["t", "cam"])
        st.altair_chart(chart, use_container_width=True)

    if isinstance(cam_idx.get("temporal_inconsistency"), list):
        d = cam_idx["temporal_inconsistency"]
        ddf = pd.DataFrame(
            {"t": [float(i) / float(roi_fps) if roi_fps else float(i) for i in range(len(d))], "delta_t": d}
        )
        chart = alt.Chart(ddf).mark_line().encode(x="t:Q", y="delta_t:Q", tooltip=["t", "delta_t"])
        st.altair_chart(chart, use_container_width=True)

    tracks = cam_idx.get("region_tracks", {}).get("tracks", []) if isinstance(cam_idx.get("region_tracks"), dict) else []
    if tracks:
        st.markdown("### Region tracks")
        st.dataframe(pd.DataFrame(tracks[:20]), use_container_width=True, hide_index=True)
    if isinstance(cam_idx.get("video_frequency_stats"), dict) and isinstance(
        cam_idx["video_frequency_stats"].get("high_freq_energy"), list
    ):
        hfe = cam_idx["video_frequency_stats"]["high_freq_energy"]
        hdf = pd.DataFrame({"idx": list(range(len(hfe))), "high_freq_energy": hfe})
        st.altair_chart(
            alt.Chart(hdf).mark_line().encode(x="idx:Q", y="high_freq_energy:Q", tooltip=["idx", "high_freq_energy"]),
            use_container_width=True,
        )
    fused_path = cam_idx.get("fused_heatmap_path")
    if isinstance(fused_path, str) and os.path.isfile(fused_path):
        try:
            import numpy as _np
            fused = _np.load(fused_path)
            if fused.ndim == 3 and fused.shape[0] > 0:
                intensity = fused.reshape(fused.shape[0], -1).mean(axis=1)
                fdf = pd.DataFrame({"idx": list(range(len(intensity))), "fused_intensity": intensity.tolist()})
                st.altair_chart(
                    alt.Chart(fdf).mark_line().encode(x="idx:Q", y="fused_intensity:Q", tooltip=["idx", "fused_intensity"]),
                    use_container_width=True,
                )
        except Exception as e:
            st.caption(f"Could not load fused heatmap tensor: {e}")

    _render_xai_gemini_expander(
        kind="video",
        res=st.session_state.get("last_combined_res"),
        cam_idx=st.session_state.get("last_cam_idx"),
    )


def _render_fusion_algorithms_page() -> None:
    st.markdown("## Fusion & Custom Algorithms")
    _render_teaching_block(
        "Combine signals across modalities and time.",
        "Use CII (implemented) and CMID (embedding-dependent).",
        "CII = mean local variance of p(fake); CMID = low cosine similarity between A_t and V_t.",
        "explainability/instability.py, explainability/cross_modal.py, orchestrator/combined_runner.py",
        "Temporal stability and cross-modal consistency diagnostics.",
        "CMID requires aligned AVH audio/visual embeddings at runtime.",
    )
    st.code("CII: var_t = Var(p[t-k:t+k]); CII = mean(var_t)")
    st.code("CMID: sim_t = cos(A_t, V_t); cmid_t = max(0, median(sim) - sim_t)")
    st.latex(r"\mathrm{CMID}_t=\max(0,\operatorname{median}(s)-s_t),\quad s_t=\cos(A_t,V_t)")
    res = st.session_state.get("last_combined_res")
    if isinstance(res, dict):
        inst = res.get("noma_confidence_instability")
        if isinstance(inst, dict) and isinstance(inst.get("variance_per_time"), list):
            v = inst["variance_per_time"]
            df = pd.DataFrame({"idx": list(range(len(v))), "var": v})
            st.altair_chart(alt.Chart(df).mark_line().encode(x="idx:Q", y="var:Q"), use_container_width=True)
            st.caption(f"CII: {float(inst.get('CII', 0.0)):.4e}")
        st.markdown(f"**CMID status:** `{res.get('cmid_status', 'unknown')}`")
        if isinstance(res.get("cmid"), dict):
            st.json(res["cmid"])
            if isinstance(res["cmid"].get("cmid"), list):
                c = res["cmid"]["cmid"]
                cdf = pd.DataFrame({"idx": list(range(len(c))), "cmid": c})
                st.altair_chart(alt.Chart(cdf).mark_line().encode(x="idx:Q", y="cmid:Q"), use_container_width=True)
    else:
        st.info("Run Combined inference first to populate fusion diagnostics.")


def _render_evaluation_page() -> None:
    st.markdown("## Evaluation & Metrics")
    _render_teaching_block(
        "Assess reliability and calibration quality.",
        "Show calibration, leakage audit, and available summary metrics.",
        "Offline evaluation scripts produce calibration and leakage artifacts.",
        "tools/calibration_fit.py, calibration_runtime.py, tools/leakage_audit.py",
        "Available metric artifacts and quick-read summaries.",
        "Full benchmark metrics require curated eval datasets.",
    )
    cal_path = os.path.join(PROJECT_ROOT, "calibration_artifacts.json")
    if os.path.isfile(cal_path):
        import json as _json
        with open(cal_path, "r", encoding="utf-8") as f:
            art = _json.load(f)
        st.json(art)
        if isinstance(art, dict):
            noma = art.get("noma", {})
            avh = art.get("avh", {})
            temp_vals = []
            if isinstance(noma, dict) and "temperature" in noma:
                temp_vals.append({"model": "NOMA", "temperature": float(noma["temperature"])})
            if isinstance(avh, dict) and "temperature" in avh:
                temp_vals.append({"model": "AVH", "temperature": float(avh["temperature"])})
            if temp_vals:
                st.altair_chart(
                    alt.Chart(pd.DataFrame(temp_vals)).mark_bar().encode(x="model:N", y="temperature:Q"),
                    use_container_width=True,
                )
    else:
        st.info("No calibration_artifacts.json found yet.")


def _render_research_chat_page(lang: str) -> None:
    """Serp + Google Lens + News + Gemini; optional Combined-run context."""
    from integrations.research_chat.chat_orchestrator import (
        format_detection_context_from_combined,
        run_research_turn,
    )

    st.markdown(t("research_title", lang))
    st.caption(t("research_caption", lang))

    has_combined = (
        isinstance(st.session_state.get("last_combined_res"), dict)
        and st.session_state["last_combined_res"].get("avh_ok")
    )
    include_combined = False
    if has_combined:
        include_combined = st.checkbox(
            t("research_include_combined", lang),
            value=True,
            key="research_include_combined",
            help=t("research_include_help", lang),
        )

    det_ctx = None
    if include_combined and has_combined:
        det_ctx = format_detection_context_from_combined(
            st.session_state["last_combined_res"],
            st.session_state.get("last_cam_idx"),
        )

    with st.expander(t("research_env_expander", lang), expanded=False):
        st.markdown(t("research_env_md", lang))

    show_src = st.checkbox(t("research_show_sources", lang), value=False, key="research_show_sources")

    msgs = st.session_state["research_chat_messages"]
    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input(t("research_chat_placeholder", lang)):
        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Fetching sources and synthesizing…"):
                turn = run_research_turn(
                    prompt,
                    detection_context=det_ctx if include_combined else None,
                    history=msgs[:-1],
                )
            reply = (turn.text or turn.error or "No response.").strip()
            st.markdown(reply)
            if show_src and turn.sources_used:
                with st.expander("Source fetch summary", expanded=False):
                    st.json(
                        {
                            "errors": turn.sources_used.get("errors"),
                            "serp_count": len(turn.sources_used.get("serp") or []),
                            "google_lens_count": len(turn.sources_used.get("google_lens") or []),
                            "news_count": len(turn.sources_used.get("news") or []),
                        }
                    )
        msgs.append({"role": "assistant", "content": reply})

    if msgs and st.button(t("research_clear", lang), key="research_clear_chat"):
        st.session_state["research_chat_messages"] = []
        st.rerun()


def _render_final_combined_report_page() -> None:
    _render_teaching_block(
        "Produce a single forensic summary after Combined inference.",
        "Aggregate AVH score, NOMA timeline, video/audio XAI, and fusion diagnostics.",
        "Combine calibrated AVH and mean NOMA p(fake) into a blended risk score for triage.",
        "orchestrator/combined_runner.py + calibration_runtime.py + explainability/*",
        "Tabs: user summary, developer tables + Grad-CAM/fused heatmaps, Serp/Lens/News + Gemini.",
        "CMID may be unavailable until AVH embedding export is enabled.",
    )
    res = st.session_state.get("last_combined_res")
    if not isinstance(res, dict):
        st.info("Run Inference Demo → Combined first to generate the final integrated report.")
        return

    from ui.integrated_verdict import render_integrated_final_report

    render_integrated_final_report(
        res=res,
        cam_idx=st.session_state.get("last_cam_idx"),
        lang=lang,
        use_unsup_avh=use_unsup_avh,
        render_limitations=_render_combined_limitations_expander,
    )

# ═══════════════════════════════════════════════════════════════════════════
#  NOMA (Audio-Only) interface  → lives under "Inference Demo" module
# ═══════════════════════════════════════════════════════════════════════════
if page == "Home / Overview":
    _render_home_overview(lang)
elif page == "Dataset Explorer":
    _render_dataset_explorer()
elif page == "Preprocessing":
    _render_preprocessing_page()
elif page == "Feature Extraction":
    _render_feature_extraction_page()
elif page == "Model & Training":
    _render_model_training_page()
elif page == "Video Explainability":
    _render_video_xai_page()
elif page == "Audio Explainability":
    _render_audio_xai_page()
elif page == "Fusion & Custom Algorithms":
    _render_fusion_algorithms_page()
elif page == "Evaluation & Metrics":
    _render_evaluation_page()
elif page == "Final Combined Report":
    _render_final_combined_report_page()
elif page == "Fact check (STT + Serp + News)":
    _render_fact_check_page()
elif page == "Research chat":
    _render_research_chat_page(lang)

# The existing NOMA / AVH / Combined controls are grouped under "Inference Demo".
method = None
if page == "Inference Demo":
    method = st.radio(
        "Detection method",
        ["NOMA (Audio-Only)", "AVH-Align (Audio-Visual)", "Combined (AVH → NOMA)"],
        index=2,
        help="Combined is the recommended demo path (AVH lip–speech + NOMA audio + fused score).",
    )

if page == "Inference Demo" and method == "NOMA (Audio-Only)":
    st.markdown("---")
    st.markdown("## 🎧 NOMA — Audio-Only Fake Detection")

    # Visual: pipeline flow diagram
    st.markdown("#### 🔀 Pipeline at a glance")
    st.markdown("""
    <div class="pipeline-flow">
        <div><div class="flow-step audio">🎵 Audio</div><div class="flow-label">WAV/MP3</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">✂️ Split</div><div class="flow-label">1s blocks</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">📊 Features</div><div class="flow-label">MFCC, spectral</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">📐 L2 norm</div><div class="flow-label">Normalize</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🤖 SVM</div><div class="flow-label">RBF kernel</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step output">✅ Verdict</div><div class="flow-label">Real / Fake</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Insight cards: Input | Model | Output
    st.markdown("""
    <div class="insight-row">
        <div class="insight-card"><div class="icon">🎤</div><div class="title">Input</div><div class="desc">Single audio file (speech)</div></div>
        <div class="insight-card"><div class="icon">🧮</div><div class="title">Model</div><div class="desc">Hand-crafted features + SVM</div></div>
        <div class="insight-card"><div class="icon">📋</div><div class="title">Output</div><div class="desc">Per-second Real/Fake + overall verdict</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 What does NOMA do?", expanded=True):
        st.markdown("""
        **NOMA** (from Mozilla AI’s fake-audio-detection) is a **lightweight, audio-only** system that decides whether 
        an audio clip is **real human speech** or **synthetic/fake** (e.g. TTS, voice cloning). It does **not** use video.
        - **Input:** A single audio file (WAV, MP3, OGG).
        - **Output:** Per-second predictions (Real/Fake) and an overall verdict, plus confidence.
        - **Use case:** Fast screening of speech recordings when you don’t have video (e.g. podcasts, calls).
        """)

    with st.expander("🔄 How does the process work?"):
        st.markdown("**Visual timeline:**")
        st.markdown("""
        <div class="timeline">
            <div class="timeline-step"><strong>1. Load audio</strong> — Resampled to 22.05 kHz mono.</div>
            <div class="timeline-step"><strong>2. Split into 1-second blocks</strong> — Each block is analyzed independently.</div>
            <div class="timeline-step"><strong>3. Feature extraction</strong> — Chroma STFT, spectral centroid/bandwidth/rolloff, MFCCs (20), IMFCCs (13), RMS, ZCR, tonnetz.</div>
            <div class="timeline-step"><strong>4. L2 normalization</strong> — Feature vectors normalized to unit length.</div>
            <div class="timeline-step"><strong>5. SVM classification</strong> — RBF kernel predicts Fake (0) or Real (1) per block.</div>
            <div class="timeline-step"><strong>6. Aggregation</strong> — Overall verdict from block-wise predictions.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="step-box">📊 <strong>Why 1-second blocks?</strong> Balance between temporal detail and enough signal for stable features.</div>', unsafe_allow_html=True)

    with st.expander("🧠 ML/DL behind it"):
        st.markdown("""
        - **Model:** Scikit-learn **Pipeline**: L2 normalizer → **SVC** (RBF kernel, C=1, gamma='scale', probability=True).
        - **Training:** Supervised on labeled Real/Fake audio; features from librosa + custom IMFCCs.
        - **No deep learning:** Hand-crafted features + SVM → interpretable, fast, and small footprint.
        - **Loss:** SVM uses hinge loss; we report accuracy, precision, recall, F1 on a hold-out set.
        """)

    st.markdown("---")
    st.markdown("### ▶️ Try NOMA")
    st.markdown("""
    <div style="padding:1rem 1.25rem; border-radius:10px; background:#f8fafc; border:1px solid #e2e8f0; margin-bottom:1rem;">
        Upload an audio file below and click <strong>Run NOMA prediction</strong> to see per-second Real/Fake analysis.
    </div>
    """, unsafe_allow_html=True)

    noma_model_path = get_noma_model_path()
    if not noma_model_path:
        st.warning("No NOMA model found. Add one of: `model/fake_audio_detection.joblib` or `model/noma-1`.")
    else:
        upload = st.file_uploader("Upload an audio file (WAV, MP3, OGG)", type=["wav", "mp3", "ogg"], key="noma_upload")
        if upload is not None:
            st.audio(upload, format=f"audio/{upload.name.split('.')[-1].lower()}")
        else:
            st.info("Upload an audio file above to run NOMA.")

        show_noma_feature_sensitivity = st.checkbox(
            "Explainability: NOMA feature sensitivity heatmap (per-block permutation)",
            value=False,
            key="noma_feature_sens_checkbox",
        )

        if st.button("Run NOMA prediction", key="noma_btn") and noma_model_path:
            if upload is None:
                st.error("Please upload an audio file first.")
            else:
                with st.spinner("Analyzing audio (feature extraction + SVM)…"):
                    try:
                        audio_bytes = io.BytesIO(upload.getvalue())
                        if show_noma_feature_sensitivity:
                            times, probas, feature_matrix, feature_names = run_noma_prediction_with_features(
                                noma_model_path,
                                audio_bytes=audio_bytes,
                                audio_filename=upload.name,
                            )
                        else:
                            times, probas = run_noma_prediction(
                                noma_model_path,
                                audio_bytes=audio_bytes,
                                audio_filename=upload.name,
                            )
                        noma_pipe = get_noma_pipeline(noma_model_path)
                        p_fake_raw, confidences, preds_str = noma_p_fake_raw_confidence_and_preds_from_probas(
                            noma_pipe,
                            probas,
                        )
                        from calibration_runtime import noma_p_fake_to_calibrated

                        p_fake = noma_p_fake_to_calibrated(p_fake_raw)
                        p_real = 1.0 - p_fake
                        df = pd.DataFrame({
                            "Seconds": times,
                            "Prediction": preds_str,
                            "Confidence": confidences,
                            "p_fake": p_fake,
                            "p_real": p_real,
                        })
                        from calibration_runtime import get_uncertainty_margins

                        _, noma_margin = get_uncertainty_margins()
                        df["Confidence Level"] = df.apply(
                            lambda r: "Uncertain" if abs(float(r["p_fake"]) - 0.5) < noma_margin else r["Prediction"],
                            axis=1,
                        )

                        st.markdown("#### 📈 Prediction by 1-second blocks")
                        chart = (
                            alt.Chart(df)
                            .mark_bar()
                            .encode(
                                x=alt.X("Seconds:O", title="Seconds"),
                                y=alt.value(30),
                                color=alt.Color(
                                    "Confidence Level:N",
                                    scale=alt.Scale(
                                        domain=["Fake", "Real", "Uncertain"],
                                        range=["#ef4444", "#22c55e", "#94a3b8"],
                                    ),
                                ),
                                tooltip=["Seconds", "Prediction", "p_fake", "Confidence"],
                            )
                            .properties(width=700, height=150)
                        )
                        st.altair_chart(chart, width="stretch")

                        if show_noma_feature_sensitivity:
                            from explainability.noma_feature_sensitivity import compute_noma_permutation_feature_sensitivity

                            pipeline = noma_pipe

                            max_blocks_for_heatmap = 120
                            sens = compute_noma_permutation_feature_sensitivity(
                                feature_matrix=feature_matrix,
                                pipeline=pipeline,
                                feature_names=feature_names,
                                block_times_seconds=times,
                                seed=42,
                                max_blocks=max_blocks_for_heatmap,
                                top_k=5,
                                use_calibrated_p_fake=True,
                            )

                            used_times = sens["block_times_seconds"] or times
                            used_times = list(used_times)

                            import numpy as np

                            sens_abs = np.asarray(sens["sensitivity_abs"], dtype=float)
                            used_F = sens_abs.shape[1]
                            if used_F != len(sens["feature_names"]):
                                raise RuntimeError("Sensitivity matrix shape mismatch.")

                            st.markdown("#### 🔥 NOMA feature sensitivity heatmap (per 1s block)")
                            if len(used_times) != len(times):
                                st.caption(f"Computed on {len(used_times)}/{len(times)} blocks (capped at {max_blocks_for_heatmap}).")

                            df_sens = pd.DataFrame(
                                {
                                    "Seconds": np.repeat(np.asarray(used_times, dtype=float), used_F),
                                    "Feature": np.tile(np.asarray(sens["feature_names"], dtype=str), len(used_times)),
                                    "Sensitivity": sens_abs.flatten(order="C"),
                                }
                            )

                            heatmap = (
                                alt.Chart(df_sens)
                                .mark_rect()
                                .encode(
                                    x=alt.X("Feature:N", title="Feature"),
                                    y=alt.Y("Seconds:Q", title="Seconds", axis=alt.Axis(labelAngle=0)),
                                    color=alt.Color(
                                        "Sensitivity:Q",
                                        title="|Delta p(fake)|",
                                        scale=alt.Scale(scheme="redyellowblue"),
                                    ),
                                )
                                .properties(width=900, height=220)
                            )
                            st.altair_chart(heatmap, use_container_width=True)

                            st.markdown("##### Top suspicious features (first blocks)")
                            st.json(sens["topk_per_block"][:3])

                            import json

                            sens_json_bytes = json.dumps(sens, indent=2).encode("utf-8")
                            st.download_button(
                                label="Download NOMA feature sensitivity JSON",
                                data=sens_json_bytes,
                                file_name="nona_feature_sensitivity.json",
                                mime="application/json",
                                key="nona_sens_download_json",
                            )

                            # Compact summary for quick human inspection.
                            feat_names = np.asarray(sens["feature_names"], dtype=str)
                            mean_sens_abs = np.mean(sens_abs, axis=0)
                            ranked = np.argsort(mean_sens_abs)[::-1][:10]
                            df_susp = pd.DataFrame(
                                {
                                    "feature": feat_names[ranked],
                                    "mean_sensitivity_abs": mean_sens_abs[ranked],
                                }
                            )
                            st.download_button(
                                label="Download suspicious_feature_names.csv",
                                data=df_susp.to_csv(index=False).encode("utf-8"),
                                file_name="suspicious_feature_names.csv",
                                mime="text/csv",
                                key="nona_sens_download_csv",
                            )

                        st.markdown("#### 🎯 Overall verdict")
                        if all(p == "Real" for p in preds_str):
                            st.success("**Verdict: REAL** — No synthetic segments detected.")
                        elif all(p == "Fake" for p in preds_str):
                            st.error("**Verdict: FAKE** — Audio appears synthetic across blocks.")
                        else:
                            st.warning("**Verdict: MIXED** — Some blocks detected as fake.")
                        n_blocks = len(preds_str)
                        mean_conf = float(confidences.mean())
                        st.markdown(f"""
                        <div class="step-box" style="margin-top:0.5rem;">
                            📊 <strong>Summary:</strong> {n_blocks} block(s) analyzed · Mean confidence: {mean_conf:.2%}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption("Analysis: NOMA (Mozilla-style audio-only pipeline).")
                    except Exception as e:
                        st.exception(e)

# ═══════════════════════════════════════════════════════════════════════════
#  AVH-Align (Audio-Visual) interface
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Inference Demo" and method == "AVH-Align (Audio-Visual)":
    st.markdown("---")
    st.markdown("## 🎬 AVH-Align — Audio-Visual Deepfake Detection")

    # Visual: pipeline flow diagram
    st.markdown("#### 🔀 Pipeline at a glance")
    st.markdown("""
    <div class="pipeline-flow">
        <div><div class="flow-step visual">🎬 Video</div><div class="flow-label">MP4 + face</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">👤 Face & mouth</div><div class="flow-label">dlib, 96×96 ROI</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🧠 AV-HuBERT</div><div class="flow-label">1024-d features</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🔀 Fusion MLP</div><div class="flow-label">audio+visual</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step output">📈 Score</div><div class="flow-label">higher = faker</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Insight cards
    st.markdown("""
    <div class="insight-row">
        <div class="insight-card"><div class="icon">🎥</div><div class="title">Input</div><div class="desc">Video with face + audio</div></div>
        <div class="insight-card"><div class="icon">🔗</div><div class="title">Idea</div><div class="desc">Lip–speech alignment</div></div>
        <div class="insight-card"><div class="icon">📊</div><div class="title">Output</div><div class="desc">Single deepfake score</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 What does AVH-Align do?", expanded=True):
        st.markdown("""
        **AVH-Align** (from [CVPR 2025](https://github.com/SOHAM240104/AVH)) detects deepfakes by checking whether 
        **mouth movements and speech are in sync**. It uses both **video** and **audio**.
        - **Input:** A video file (e.g. MP4) with a visible face and audio.
        - **Output:** A **deepfake score**. **Higher score ⇒ more likely fake** (mouth-speech mismatch).
        - **Use case:** Detecting face-swapped or lip-synced videos (e.g. talking-head deepfakes).
        """)

    with st.expander("🔄 How does the process work?"):
        st.markdown("**Visual timeline:**")
        st.markdown("""
        <div class="timeline">
            <div class="timeline-step"><strong>1. Preprocessing</strong> — Face detection (dlib), 68-point landmarks, mouth ROI crop (96×96). Audio extracted to WAV (16 kHz).</div>
            <div class="timeline-step"><strong>2. AV-HuBERT</strong> — Visual: mouth frames → 1024-d vectors. Audio: log-filterbank → 1024-d vectors (time-aligned).</div>
            <div class="timeline-step"><strong>3. Fusion & scoring</strong> — L2-normalized audio + visual features → MLP → single score (higher when lip–speech is out of sync).</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="step-box">🔬 <strong>Key idea:</strong> Real videos have natural lip–speech alignment; '
            'many deepfakes break this alignment, which AVH-Align learns to detect.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("🧠 ML/DL behind it"):
        st.markdown("""
        - **AV-HuBERT:** Transformer-based audio-visual representation (Facebook Research); pretrained on 433h of LRS3/VoxCeleb.
        - **FusionModel:** Lightweight MLP (visual_proj → 512, audio_proj → 512; concat → 1024 → 512 → 256 → 128 → 1).
        - **Training:** Unsupervised / contrastive-style on audio-visual alignment; fine-tuned on deepfake datasets (e.g. AV-Deepfake1M).
        - **Inference:** L2-normalize features → FusionModel → logsumexp over time → single deepfake score.
        """)

    st.markdown("---")
    st.markdown("### ▶️ Try AVH-Align")
    st.markdown("""
    <div style="padding:1rem 1.25rem; border-radius:10px; background:#fffbeb; border:1px solid #fde68a; margin-bottom:1rem;">
        Upload a video (face + audio) below and click <strong>Run AVH-Align</strong>. Processing may take 1–3 minutes on CPU.
    </div>
    """, unsafe_allow_html=True)

    avh_checks = check_avh_setup()
    avh_full_ready = all(ok for _, ok, _ in avh_checks)
    # Video path works if script + fusion ckpt exist; use sidebar "Python for AVH" if you use another env.
    avh_video_ready = os.path.exists(AVH_TEST_SCRIPT) and os.path.exists(AVH_AVHUBERT_CKPT) and (use_unsup_avh or os.path.exists(AVH_FUSION_CKPT))
    avh_fusion_only = os.path.exists(AVH_FUSION_CKPT)

    with st.expander("🔧 AVH setup status & how to fix", expanded=not avh_full_ready):
        for name, ok, path in avh_checks:
            st.markdown(f"{'✅' if ok else '❌'} **{name}** — `{path}`")
        auto_download = st.checkbox(
            "Attempt best-effort auto-download missing AVH artifacts",
            value=False,
            help="If enabled, the app will try to download missing AVH checkpoint files from known URLs. "
            "This may take time and requires network access.",
        )
        if st.button("Download missing artifacts", disabled=not auto_download, key="avh_download_btn"):
            from artifact_manager import ensure_artifacts, default_artifacts

            with st.spinner("Checking + downloading missing artifacts…"):
                res = ensure_artifacts(
                    artifacts=default_artifacts(),
                    download_missing=True,
                    strict_hash=False,
                    write_lock_if_missing=True,
                )
            st.success("Artifact check completed. If files were missing, they should now be present.")
            st.rerun()
        st.markdown("**Tip:** If you already made AVH work in another env (e.g. `conda activate avh`), set **Python for AVH video** in the sidebar to that Python path — then video upload will use it and no clone is needed in this app's env.")
        if not avh_full_ready:
            st.markdown("---")
            st.markdown("**Full setup (only if you want video → score in this app's env):**")
            st.markdown("""
            1. **Clone AV-HuBERT inside AVH** (required):
               ```bash
               cd AVH && git clone https://github.com/facebookresearch/av_hubert.git && cd av_hubert/avhubert
               git submodule init && git submodule update
               cd ../fairseq && pip install --editable . && cd ../avhubert
               ```
            2. **Install Python deps** (use a Python 3.10 env; fairseq needs it):
               ```bash
               pip install torch torchvision torchaudio
               pip install opencv-python dlib librosa python_speech_features scikit-video sentencepiece
               pip install "numpy<1.24" "omegaconf>=2.1" "hydra-core>=1.1"
               ```
            3. **Download face model & mean face** (from inside `AVH/av_hubert/avhubert`):
               ```bash
               mkdir -p content/data/misc
               curl -L -o content/data/misc/shape_predictor_68_face_landmarks.dat.bz2 \\
                 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
               bzip2 -d content/data/misc/shape_predictor_68_face_landmarks.dat.bz2
               curl -L -o content/data/misc/20words_mean_face.npy \\
                 https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/raw/master/preprocessing/20words_mean_face.npy
               ```
            4. **Download AV-HuBERT checkpoint** (~1 GB, in `AVH/av_hubert/avhubert`):
               ```bash
               curl -L -o self_large_vox_433h.pt \\
                 https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/self_large_vox_433h.pt
               ```
            5. **Install ffmpeg**: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux).

            Then restart this app and try again. Full details: [AVH README](https://github.com/SOHAM240104/AVH).
            """)

    if (not avh_fusion_only) and (not use_unsup_avh):
        st.warning("AVH-Align Fusion checkpoint not found. Add `AVH/checkpoints/AVH-Align_AV1M.pt` to use AVH.")
    else:
        # Video path: use sidebar Python if set (env where AVH works)
        if avh_video_ready:
            st.markdown("**From video:**")
            upload_v = st.file_uploader("Upload a video file (MP4 with face + audio)", type=["mp4", "avi", "mov"], key="avh_upload")
            if upload_v is not None:
                st.video(upload_v)
            avh_smart_crop = st.selectbox(
                "Spatial pre-crop (reels / on-screen text)",
                options=["auto", "off", "reel", "face"],
                index=3,
                key="avh_smart_crop",
                format_func=lambda m: {
                    "auto": "Auto — face or vertical UI bands",
                    "off": "Off — full frame before mouth ROI",
                    "reel": "Vertical band — strip top/bottom captions",
                    "face": "Face-focused (default) — dlib face box, then mouth ROI",
                }.get(m, m),
                help="Runs before AVH mouth detection: reduces caption/UI chrome on 9:16 clips and tightens on the speaker.",
            )
            render_manual_crop_ui(
                state_prefix="avh",
                file_bytes=upload_v.getvalue() if upload_v else None,
                filename=upload_v.name if upload_v else None,
            )
            if avh_python_path:
                st.caption(f"Using Python: `{avh_python_path}`")
            else:
                st.error("Set **Python for AVH video** in the sidebar to your avh conda Python (e.g. `.../envs/avh/bin/python`). The venv Python will fail with omegaconf errors.")
            avh_btn_label = "Run AVH Unsupervised on video" if use_unsup_avh else "Run AVH-Align on video"
            if st.button(avh_btn_label, key="avh_btn") and upload_v is not None:
                if not (avh_python_path and os.path.isfile(avh_python_path)):
                    st.error("Set **Python for AVH video** in the sidebar to your avh conda Python. Example: `/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python`")
                else:
                    suffix = os.path.splitext(upload_v.name)[-1] or ".mp4"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(upload_v.getvalue())
                        tmp_path = tmp.name
                    cropped_tmp = ""
                    manual_r = get_saved_manual_rect("avh", upload_v.getvalue())
                    path_avh, cropped_tmp, used_manual = prepare_video_with_optional_manual_crop(
                        tmp_path, upload_v.name, manual_r
                    )
                    sc_use = "off" if used_manual else avh_smart_crop
                    if manual_r and not used_manual:
                        st.warning("Manual crop failed to encode; running on the full frame with automatic pre-crop.")
                    try:
                        with st.spinner("Running AVH pipeline (preprocess → AV-HuBERT → FusionModel). This may take 1–3 min on CPU…"):
                            if use_unsup_avh:
                                ok, result = run_avh_unsupervised_on_video(
                                    path_avh,
                                    timeout=900,
                                    python_exe=avh_python_path,
                                    smart_crop=sc_use,
                                )
                            else:
                                ok, result = run_avh_on_video(
                                    path_avh,
                                    timeout=900,
                                    python_exe=avh_python_path,
                                    smart_crop=sc_use,
                                )
                    except Exception as e:
                        ok, result = False, str(e)
                    _show_avh_score_or_error(ok, result, use_unsup_avh=use_unsup_avh)
                    try:
                        os.unlink(tmp_path)
                        if cropped_tmp:
                            os.unlink(cropped_tmp)
                    except Exception:
                        pass
            elif upload_v is None and not avh_full_ready:
                st.info("Upload a video above, or use **Score from .npz** below (needs only the Fusion checkpoint).")
        else:
            st.caption("Video pipeline needs `AVH/test_video.py`. Use **Score from .npz** below with just the checkpoint.")

        # .npz path: only Fusion checkpoint (no av_hubert, no dlib)
        st.markdown("**Or score from pre-extracted features (no clone needed):**")
        upload_npz = st.file_uploader("Upload pre-extracted .npz (keys: `visual`, `audio`)", type=["npz"], key="avh_npz")
        if st.button("Score .npz with Fusion model", key="avh_npz_btn") and upload_npz is not None and avh_fusion_only:
            with st.spinner("Scoring with Fusion checkpoint…"):
                ok, result = run_avh_from_npz(upload_npz.getvalue(), AVH_FUSION_CKPT)
            _show_avh_score_or_error(ok, result)
        if avh_fusion_only and not upload_npz:
            st.caption("If you have a .npz from a previous AVH feature extraction, upload it — only the Fusion checkpoint is used.")

# ═══════════════════════════════════════════════════════════════════════════
#  Combined (AVH → NOMA) interface
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Inference Demo" and method == "Combined (AVH → NOMA)":
    st.markdown("---")
    st.markdown("## 🔀 Combined — Video → AVH → NOMA")

    st.markdown("#### 🔀 Pipeline at a glance")
    st.markdown("""
    <div class="pipeline-flow">
        <div><div class="flow-step visual">🎬 Video</div><div class="flow-label">MP4</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step process">🎬 AVH</div><div class="flow-label">Lip–speech</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step audio">🎵 Extract</div><div class="flow-label">Audio WAV</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step ml">🤖 NOMA</div><div class="flow-label">TTS detection</div></div>
        <span class="flow-arrow">→</span>
        <div><div class="flow-step output">📊 Both scores</div><div class="flow-label">AVH + NOMA</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-row">
        <div class="insight-card"><div class="icon">🎬</div><div class="title">Input</div><div class="desc">Single MP4 video</div></div>
        <div class="insight-card"><div class="icon">🔗</div><div class="title">Flow</div><div class="desc">AVH extracts audio → NOMA analyzes it</div></div>
        <div class="insight-card"><div class="icon">📊</div><div class="title">Output</div><div class="desc">AVH score + NOMA per-second Real/Fake</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📌 What does Combined do?", expanded=True):
        st.markdown("""
        **Combined** runs both detectors on a single video:
        1. **AVH-Align** processes the video (face + mouth crop, AV-HuBERT, Fusion) → **AVH score** (lip–speech alignment).
        2. The **extracted audio** from AVH preprocessing is passed to **NOMA** → **per-second Real/Fake** (TTS/synthetic detection).
        - **Use case:** Get both audio-visual and audio-only signals in one run.
        """)

    st.markdown("---")
    st.markdown("### ▶️ Try Combined")
    st.markdown("""
    <div style="padding:1rem 1.25rem; border-radius:10px; background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%); border:1px solid #f59e0b; margin-bottom:1rem;">
        Upload a video <em>or</em> choose <strong>Local folder</strong> (defaults to <code>untitled folder 2</code>) and click <strong>Run Combined</strong>.
        <strong>Visual proof:</strong> mouth ROI Grad-CAM — <em>~6 diverse key frames</em> by default (not hundreds).
        <strong>Audio proof:</strong> NOMA per-second table + charts below. <strong>Fusion:</strong> sidebar <code>full</code> reliability blend.
    </div>
    """, unsafe_allow_html=True)

    noma_model_path = get_noma_model_path()
    avh_video_ready = os.path.exists(AVH_TEST_SCRIPT) and os.path.exists(AVH_AVHUBERT_CKPT) and (use_unsup_avh or os.path.exists(AVH_FUSION_CKPT))

    if not (noma_model_path and avh_video_ready):
        st.warning("Combined needs both NOMA model and AVH setup. Check sidebar **Model paths** and **Python for AVH video**.")
    elif not (avh_python_path and os.path.isfile(avh_python_path)):
        st.error("Set **Python for AVH video** in the sidebar to your avh conda Python.")
    else:
        default_eval_folder = os.path.join(PROJECT_ROOT, "untitled folder 2")
        video_source = st.radio(
            "Video source",
            ("Upload file", "Local folder on this machine"),
            horizontal=True,
            key="combined_video_source",
            help="Pick **Local folder** to run on files under this repo (e.g. `untitled folder 2`) without re-uploading.",
        )
        upload_combined = None
        local_folder_for_combined = default_eval_folder
        local_pick_name: str | None = None
        if video_source == "Upload file":
            upload_combined = st.file_uploader(
                "Upload a video (MP4 with face + audio)",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="combined_upload",
            )
            if upload_combined is not None:
                st.video(upload_combined)
        else:
            local_folder_for_combined = st.text_input(
                "Folder containing videos",
                value=default_eval_folder,
                key="combined_local_folder",
            )
            _vids = _list_videos_in_dir(local_folder_for_combined)
            if not _vids:
                st.caption("No video files in that folder (expects .mp4, .mov, .avi, …).")
            else:
                local_pick_name = st.selectbox("Select a video", _vids, key="combined_local_pick")
                _safe_prev = _safe_video_path_under_folder(
                    local_folder_for_combined, local_pick_name or ""
                )
                if _safe_prev:
                    st.video(_safe_prev)

        file_bytes_for_crop: bytes | None = None
        fname_for_crop: str | None = None
        if upload_combined is not None:
            file_bytes_for_crop = upload_combined.getvalue()
            fname_for_crop = upload_combined.name
        elif (
            video_source == "Local folder on this machine"
            and local_pick_name
            and local_folder_for_combined
        ):
            _p_crop = _safe_video_path_under_folder(local_folder_for_combined, local_pick_name)
            if _p_crop:
                try:
                    with open(_p_crop, "rb") as _cf:
                        file_bytes_for_crop = _cf.read()
                    fname_for_crop = local_pick_name
                except OSError:
                    file_bytes_for_crop = None

        include_gradcam = st.checkbox(
            "Include mouth ROI Grad-CAM (visual proof for the panel)",
            value=True,
            key="combined_include_gradcam",
            help=(
                f"Default saves **{STREAMLIT_GRADCAM_TOP_K}** diverse high-saliency frames (cap **{STREAMLIT_GRADCAM_MAX_FUSION_FRAMES}** fusion frames) — enough to review without hundreds of images. "
                "Uncheck for AVH + NOMA scores only (faster). Uses Fusion + AV-HuBERT; needs sidebar **Python for AVH video**."
            ),
        )
        run_forensics_cam = bool(include_gradcam)

        with st.expander("Optional manual crop (overrides automatic face crop)", expanded=True):
            render_manual_crop_ui(
                state_prefix="combined",
                file_bytes=file_bytes_for_crop,
                filename=fname_for_crop,
            )

        combined_smart_crop = st.selectbox(
            "Spatial pre-crop before AVH (automatic face detect or bands)",
            options=["auto", "off", "reel", "face"],
            index=3,
            key="combined_smart_crop",
            format_func=lambda m: {
                "auto": "Auto — try face crop, else vertical band on tall video",
                "off": "Off — full frame",
                "reel": "Vertical band only (strip captions)",
                "face": "Face-focused (default) — largest face across sampled frames",
            }.get(m, m),
            help="Uses dlib frontal face detector in AVH smart_spatial_crop; manual crop above wins over this.",
        )

        with st.expander("Advanced options (optional)", expanded=False):
            st.caption(
                f"Panel defaults are set in `config.py`: **{STREAMLIT_GRADCAM_TOP_K}** Grad-CAM overlays, "
                f"**{STREAMLIT_GRADCAM_MAX_FUSION_FRAMES}** max fusion frames, selection **{STREAMLIT_GRADCAM_SELECTION_MODE}**, "
                f"min gap **{STREAMLIT_GRADCAM_MIN_TEMPORAL_GAP}**."
            )
            export_bundle = st.checkbox(
                "Download evidence .zip (extracted audio, mouth ROI, Grad-CAM PNGs if enabled)",
                value=False,
                key="combined_export_bundle",
            )
            dump_embeddings_for_cmid = st.checkbox(
                "Cross-modal proof: dump AVH embeddings for **CMID** (slower; supervised AVH only)",
                value=False,
                disabled=use_unsup_avh,
                key="combined_dump_emb",
            )
            noma_perm_max_blocks = st.slider(
                "NOMA permutation sensitivity (extra audio proof; 0 = off)",
                min_value=0,
                max_value=60,
                value=0,
                key="combined_noma_perm",
            )
            persist_combined_artifacts = st.checkbox(
                "Persist run to **eval_runs/streamlit_combined/** (stable audio, ROI, Grad-CAM; survives refresh)",
                value=True,
                key="combined_persist_disk",
                help="Copies artifacts out of temp dirs and rewrites paths in session so Final Combined Report / Video Explainability keep working.",
            )
            cleanup_after_persist = st.checkbox(
                "Delete temp folders after persist (recommended; frees disk)",
                value=True,
                key="combined_cleanup_temp",
                disabled=not persist_combined_artifacts,
                help="Only deletes paths under the system temp directory; requires persist enabled.",
            )

        # Grad-CAM / XAI: Streamlit panel defaults (compact overlays + audio NOMA table from same run).
        forensics_top_k = int(STREAMLIT_GRADCAM_TOP_K)
        forensics_selection_mode = str(STREAMLIT_GRADCAM_SELECTION_MODE)
        forensics_min_temporal_gap = int(STREAMLIT_GRADCAM_MIN_TEMPORAL_GAP)
        forensics_max_fusion_frames = int(STREAMLIT_GRADCAM_MAX_FUSION_FRAMES)
        region_track_stride = int(STREAMLIT_GRADCAM_REGION_TRACK_STRIDE)
        run_robustness_delta = False
        adv_ckpt_path = ""
        capture_attention = False
        _can_run_combined = False
        if upload_combined is not None:
            _can_run_combined = True
        elif video_source == "Local folder on this machine" and local_pick_name:
            _can_run_combined = bool(
                _safe_video_path_under_folder(local_folder_for_combined, local_pick_name)
            )

        if st.button("Run Combined (AVH → NOMA)", key="combined_btn") and _can_run_combined:
            tmp_path: str | None = None
            video_name_run = ""
            source_for_prepare = ""

            if upload_combined is not None:
                suffix = os.path.splitext(upload_combined.name)[-1] or ".mp4"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(upload_combined.getvalue())
                    tmp_path = tmp.name
                video_name_run = upload_combined.name
                source_for_prepare = tmp_path
            else:
                _local_abs = _safe_video_path_under_folder(
                    local_folder_for_combined, local_pick_name or ""
                )
                if not _local_abs:
                    st.error("Could not resolve a safe path under the selected folder.")
                    st.stop()
                source_for_prepare = _local_abs
                video_name_run = local_pick_name or "video.mp4"

            cropped_tmp = ""
            manual_r = get_saved_manual_rect("combined", file_bytes_for_crop)
            path_avh, cropped_tmp, used_manual = prepare_video_with_optional_manual_crop(
                source_for_prepare, video_name_run, manual_r
            )
            sc_combined = "off" if used_manual else combined_smart_crop
            if manual_r and not used_manual:
                st.warning("Manual crop failed to encode; running on the full frame with automatic pre-crop.")

            cam_parent_dir = None
            try:
                _persist_dir = None
                _cleanup_volatile = False
                if st.session_state.get("combined_persist_disk", True):
                    _persist_dir = _streamlit_combined_persist_dir(video_name_run)
                    _cleanup_volatile = bool(st.session_state.get("combined_cleanup_temp", True))
                with st.status("Running Combined pipeline (AVH → NOMA)…", expanded=True) as _combined_status:
                    _combined_status.write(
                        "AVH → optional Grad-CAM (~6 diverse ROI frames) → NOMA on extracted audio → "
                        "calibrated reliability fusion. A few minutes on CPU is normal."
                    )
                    if _persist_dir:
                        _combined_status.write(f"Persist dir: `{_persist_dir}`")
                    res = run_combined_avh_to_noma(
                        video_path=path_avh,
                        video_name=video_name_run,
                        use_unsup_avh=use_unsup_avh,
                        python_exe=avh_python_path,
                        smart_crop=sc_combined,
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
                        timeout=900,
                        persist_run_dir=_persist_dir,
                        cleanup_volatile_after_persist=_cleanup_volatile,
                        dump_embeddings_for_cmid=dump_embeddings_for_cmid,
                        noma_permutation_max_blocks=(
                            int(noma_perm_max_blocks) if noma_perm_max_blocks > 0 else None
                        ),
                        late_fusion_mode=combined_late_fusion_mode,
                    )
                    st.session_state["last_combined_res"] = res
                    # Always sync Grad-CAM index: clear stale session when CAM is off or failed.
                    _ci = res.get("cam_idx") if isinstance(res, dict) else None
                    st.session_state["last_cam_idx"] = _ci if isinstance(_ci, dict) else None
                    _combined_status.update(label="Combined pipeline finished", state="complete")

                if isinstance(res, dict) and res.get("persist_run_dir"):
                    st.caption(f"Artifacts persisted under: `{res['persist_run_dir']}`")

                if res["avh_ok"]:
                    avh_score = res["avh_score"]
                    if use_unsup_avh:
                        st.success(
                            f"**AVH unsupervised score:** {avh_score:.4f} (higher = more likely mismatch/fake)"
                        )
                    else:
                        st.success(f"**AVH score:** {avh_score:.4f} (higher = more likely fake)")

                    _render_combined_demo_summary_from_res(res)
                    st.info(
                        "For a one-page summary and **Download final_combined_report.json**, open **Final Combined Report** in the sidebar."
                    )
                    _render_combined_limitations_expander(use_unsup_avh=use_unsup_avh)

                    if run_forensics_cam and res["cam_ok"] and res["cam_overlays_dir"]:
                        cam_parent_dir = res["cam_parent_dir"]
                        cam_idx = res["cam_idx"]
                        overlays_dir = res["cam_overlays_dir"]

                        st.markdown("#### 📌 Forensics evidence: mouth ROI Grad-CAM")
                        st.caption(f"Grad-CAM target score: {float(cam_idx.get('score', float('nan'))):.4f}")

                        # Temporal CAM evidence (best-effort heatmap/trace).
                        cam_per_frame = cam_idx.get("cam_per_frame")
                        cam_to_roi = cam_idx.get("cam_to_roi_index")
                        roi_fps = cam_idx.get("roi_fps")
                        if isinstance(cam_per_frame, list) and len(cam_per_frame) > 0:
                            points = []
                            for cam_i, cam_val in enumerate(cam_per_frame):
                                roi_i = cam_to_roi[cam_i] if isinstance(cam_to_roi, list) and cam_i < len(cam_to_roi) else cam_i
                                if roi_i is None:
                                    continue
                                t_sec = float(roi_i) / float(roi_fps) if roi_fps else float(roi_i)
                                points.append({"t": t_sec, "cam": float(cam_val)})
                            if points:
                                cam_df = pd.DataFrame(points)
                                cam_chart = (
                                    alt.Chart(cam_df)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("t:Q", title="Time (s)" if roi_fps else "Time index"),
                                        y=alt.Y("cam:Q", title="CAM intensity (normalized)"),
                                        tooltip=["t", "cam"],
                                    )
                                )
                                st.altair_chart(cam_chart, use_container_width=True)

                        # XAI additions: temporal inconsistency, region tracks, fused heatmaps, and frequency stats.
                        temporal_inconsistency = cam_idx.get("temporal_inconsistency")
                        region_tracks = cam_idx.get("region_tracks")
                        fused_heatmap_path = cam_idx.get("fused_heatmap_path")
                        freq_stats = cam_idx.get("video_frequency_stats")

                        roi_fps = cam_idx.get("roi_fps")
                        cam_to_roi = cam_idx.get("cam_to_roi_index")

                        if isinstance(temporal_inconsistency, list) and len(temporal_inconsistency) > 1:
                            points = []
                            for cam_i, d in enumerate(temporal_inconsistency):
                                if cam_to_roi and isinstance(cam_to_roi, list) and cam_i < len(cam_to_roi):
                                    roi_i = cam_to_roi[cam_i]
                                else:
                                    roi_i = cam_i
                                if roi_i is None:
                                    continue
                                t_sec = float(roi_i) / float(roi_fps) if roi_fps else float(roi_i)
                                points.append({"t": t_sec, "delta_t": float(d)})
                            if points:
                                inc_df = pd.DataFrame(points)
                                inc_chart = (
                                    alt.Chart(inc_df)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("t:Q", title="Time (s)"),
                                        y=alt.Y("delta_t:Q", title="Temporal inconsistency (Δt)"),
                                        tooltip=["t", "delta_t"],
                                    )
                                    .properties(height=160)
                                )
                                st.altair_chart(inc_chart, use_container_width=True)

                        if isinstance(region_tracks, dict):
                            tracks = region_tracks.get("tracks", [])
                            if tracks:
                                st.markdown("#### 🧩 Persistent suspicious regions (CAM tracks)")
                                # Sort by duration descending and show top 8 tracks.
                                tracks_sorted = sorted(tracks, key=lambda x: x.get("duration_frames", 0), reverse=True)
                                top_tracks = tracks_sorted[:8]
                                st.write(
                                    {
                                        "num_tracks": len(tracks_sorted),
                                        "top_tracks": top_tracks,
                                    }
                                )

                        if isinstance(freq_stats, dict) and isinstance(freq_stats.get("high_freq_energy"), list):
                            st.markdown("#### 📈 High-frequency artifact energy (per frame)")
                            hfe = freq_stats["high_freq_energy"]
                            if len(hfe) > 1:
                                # Map frame indices -> seconds if roi_fps is known.
                                x_vals = []
                                for i, v in enumerate(hfe):
                                    t_sec = float(i) / float(roi_fps) if roi_fps else float(i)
                                    x_vals.append({"t": t_sec, "hfe": float(v)})
                                hfe_df = pd.DataFrame(x_vals)
                                chart = (
                                    alt.Chart(hfe_df)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("t:Q", title="Time (s)"),
                                        y=alt.Y("hfe:Q", title="High-frequency energy"),
                                        tooltip=["t", "hfe"],
                                    )
                                    .properties(height=160)
                                )
                                st.altair_chart(chart, use_container_width=True)

                        if fused_heatmap_path and os.path.isfile(fused_heatmap_path):
                            st.markdown("#### 🔥 Multi-signal fused heatmap (summary)")
                            try:
                                import numpy as _np
                                fused = _np.load(fused_heatmap_path)
                                if fused.ndim == 3:
                                    # Plot mean fused intensity across space per frame.
                                    fused_t = fused.mean(axis=(1, 2))
                                    fused_df = pd.DataFrame(
                                        {
                                            "t": [float(i) / float(roi_fps) if roi_fps else float(i) for i in range(len(fused_t))],
                                            "fused": fused_t.astype(float),
                                        }
                                    )
                                    chart = (
                                        alt.Chart(fused_df)
                                        .mark_line()
                                        .encode(
                                            x=alt.X("t:Q", title="Time (s)"),
                                            y=alt.Y("fused:Q", title="Fused anomaly intensity"),
                                            tooltip=["t", "fused"],
                                        )
                                        .properties(height=160)
                                    )
                                    st.altair_chart(chart, use_container_width=True)
                            except Exception as _e:
                                st.caption(f"Could not load/plot fused heatmap: {_e}")

                        baseline_score = cam_idx.get("baseline_score", None)
                        adv_score = cam_idx.get("adv_score", None)
                        delta_score = cam_idx.get("delta_score", None)

                        if run_robustness_delta and baseline_score is not None and adv_score is not None:
                            st.markdown("#### 🧪 Robustness delta (feature-adversary)")
                            st.write(
                                {
                                    "baseline_score": float(baseline_score),
                                    "adversarial_score": float(adv_score),
                                    "delta_score": float(delta_score) if delta_score is not None else None,
                                    "epsilon": cam_idx.get("adv_epsilon", None),
                                }
                            )

                        overlay_files = [f for f in os.listdir(overlays_dir) if f.endswith(".png")]
                        for f in sorted(overlay_files):
                            st.image(os.path.join(overlays_dir, f), caption=f)
                    elif run_forensics_cam:
                        _cerr = res.get("cam_error") or "unknown error"
                        st.warning(f"Grad-CAM failed — **Video Explainability** will stay empty until this succeeds. Details:\n\n{_cerr}")

                    noma_df = res.get("noma_df")
                    if noma_df is not None and not noma_df.empty:
                        st.markdown("#### 📈 NOMA per-second predictions")
                        st.dataframe(noma_df, width="stretch", hide_index=True)
                        fake_pct = 100 * (noma_df["Prediction"] == "Fake").mean()
                        st.caption(f"Overall: {fake_pct:.1f}% blocks predicted Fake, {100-fake_pct:.1f}% Real")

                        if res.get("cmid_status") == "computed" and isinstance(res.get("cmid"), dict):
                            cmid_d = res["cmid"]
                            cmid_arr = cmid_d.get("cmid") or []
                            if isinstance(cmid_arr, list) and len(cmid_arr) > 0:
                                import statistics as _stats

                                st.markdown("#### CMID (cross-modal embedding inconsistency)")
                                st.caption(
                                    f"Mean CMID={_stats.mean(float(x) for x in cmid_arr):.4f}, "
                                    f"max={max(float(x) for x in cmid_arr):.4f} (requires embedding dump)."
                                )

                        tc = res.get("temporal_corroboration")
                        if isinstance(tc, dict) and tc.get("status") == "ok":
                            st.markdown("#### Multimodal corroboration (NOMA p(fake) vs video saliency, time-aligned)")
                            c1, c2, c3 = st.columns(3)
                            if res.get("tension_index") is not None:
                                c1.metric(
                                    "Tension |cal AVH − mean NOMA|",
                                    f"{float(res['tension_index']):.3f}",
                                )
                            if tc.get("corroboration_rate") is not None:
                                c2.metric("Corroboration rate", f"{float(tc['corroboration_rate']):.2f}")
                            if tc.get("conflict_rate") is not None:
                                c3.metric("Conflict rate", f"{float(tc['conflict_rate']):.2f}")
                            st.caption(
                                "Corroboration: high p(fake) and high saliency in the same second bin. "
                                "Conflict: high p(fake) but low saliency (review suggested)."
                            )
                            bins = tc.get("bins") or []
                            if bins:
                                corr_df = pd.DataFrame(bins)
                                base = alt.Chart(corr_df).encode(
                                    alt.X("second:Q", title="Time (s)"),
                                    tooltip=["second", "p_fake", "saliency", "corroboration", "conflict"],
                                )
                                line_p = base.mark_line(color="#2563eb").encode(
                                    alt.Y("p_fake:Q", title="NOMA p(fake) (calibrated)")
                                )
                                line_s = base.mark_line(color="#ea580c").encode(
                                    y=alt.Y(
                                        "saliency:Q",
                                        title="Video saliency (norm)",
                                        axis=alt.Axis(orient="right"),
                                    )
                                )
                                st.altair_chart(
                                    (line_p + line_s).resolve_scale(y="independent"),
                                    use_container_width=True,
                                )
                                corr_pts = corr_df[corr_df["corroboration"]]
                                if not corr_pts.empty:
                                    st.altair_chart(
                                        alt.Chart(corr_pts)
                                        .mark_circle(size=80, color="green")
                                        .encode(x="second:Q", y="p_fake:Q", tooltip=["second", "p_fake"]),
                                        use_container_width=True,
                                    )

                        perm = res.get("noma_permutation_xai")
                        if isinstance(perm, dict):
                            if perm.get("error"):
                                st.warning(f"NOMA permutation XAI: {perm['error']}")
                            elif perm.get("topk_per_block") is not None:
                                st.markdown("#### NOMA permutation feature sensitivity (top drivers)")
                                st.caption("Per-block top features by |Δ calibrated p(fake)| when shuffled across blocks.")
                                st.json(
                                    {
                                        "topk_per_block": perm.get("topk_per_block", [])[:12],
                                        "note": "Truncated display; full output in API result JSON.",
                                    }
                                )

                        # XAI: Confidence Instability Index over NOMA p(fake).
                        inst = res.get("noma_confidence_instability")
                        if isinstance(inst, dict) and inst.get("variance_per_time") is not None:
                            st.markdown("#### 🧮 NOMA Confidence Instability (CII)")
                            var = inst.get("variance_per_time", [])
                            cii = inst.get("CII", 0.0)
                            try:
                                inst_df = pd.DataFrame(
                                    {
                                        "Seconds": noma_df["Seconds"],
                                        "Variance": var,
                                    }
                                )
                                chart = (
                                    alt.Chart(inst_df)
                                    .mark_line()
                                    .encode(
                                        x=alt.X("Seconds:Q", title="Seconds"),
                                        y=alt.Y("Variance:Q", title="Local variance of p(fake)"),
                                        tooltip=["Seconds", "Variance"],
                                    )
                                    .properties(height=150)
                                )
                                st.altair_chart(chart, use_container_width=True)
                                st.caption(f"Global CII (mean variance): {float(cii):.4e}")
                            except Exception:
                                st.write(inst)

                    if export_bundle:
                        if res.get("bundle_bytes") is not None:
                            st.download_button(
                                "⬇️ Download evidence bundle (.zip)",
                                data=res["bundle_bytes"],
                                file_name="evidence_bundle.zip",
                                mime="application/zip",
                            )
                        elif res.get("bundle_error"):
                            st.error(f"Failed to build evidence bundle: {res['bundle_error']}")
                else:
                    st.error("AVH pipeline failed.")
                    with st.expander("Error details", expanded=True):
                        st.text(res.get("avh_error") or "Unknown error")
            except Exception as e:
                st.error(str(e))
            finally:
                # Clean Grad-CAM temp outputs (kept_temp for rendering)
                if run_forensics_cam and cam_parent_dir and os.path.isdir(cam_parent_dir):
                    try:
                        import shutil

                        shutil.rmtree(cam_parent_dir, ignore_errors=True)
                    except Exception:
                        pass
                try:
                    if tmp_path:
                        os.unlink(tmp_path)
                    if cropped_tmp:
                        os.unlink(cropped_tmp)
                except Exception:
                    pass
        elif not _can_run_combined:
            st.info("Upload a video, or choose **Local folder** and pick a file, then run the combined pipeline.")

        st.markdown("---")
        with st.expander("Test chat — Serp + Google Lens + News + Gemini", expanded=False):
            st.caption("No video upload required. Uses the same API stack as **Research chat** (Serp + Lens + News + Gemini).")
            from integrations.research_chat.chat_orchestrator import run_research_turn

            q_test = st.text_area(
                "Type a query to test external search + synthesis",
                height=80,
                key="combined_test_query",
                placeholder="e.g. Verify headlines about synthetic media regulation in 2025",
            )
            if st.button("Run test query", key="combined_test_run"):
                with st.spinner("Fetching Serp, Lens, News, Gemini…"):
                    turn = run_research_turn((q_test or "").strip(), detection_context=None, history=[])
                st.session_state["combined_test_turn"] = {
                    "text": turn.text,
                    "error": turn.error,
                    "sources_used": turn.sources_used,
                }
            ct = st.session_state.get("combined_test_turn")
            if isinstance(ct, dict) and (ct.get("text") or ct.get("error")):
                if ct.get("error"):
                    st.error(ct["error"])
                else:
                    st.markdown("**Gemini**")
                    st.markdown(ct.get("text") or "")
                su = ct.get("sources_used") or {}
                if su.get("errors"):
                    st.warning("Tool errors: " + "; ".join(su["errors"]))
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Serp rows", len(su.get("serp") or []))
                with c2:
                    st.metric("Lens rows", len(su.get("google_lens") or []))
                with c3:
                    st.metric("News rows", len(su.get("news") or []))
                if st.button("Clear test results", key="combined_test_clear"):
                    st.session_state.pop("combined_test_turn", None)
                    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**How to run (terminal)**")
st.sidebar.code("streamlit run unified_deepfake_app.py", language="bash")
st.sidebar.markdown("---")
st.sidebar.caption("NOMA: Mozilla-style audio-only · AVH-Align: CVPR 2025 audio-visual")
