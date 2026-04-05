import os

# Centralized path/config constants for the Streamlit app.
#
# NOTE: AVH runtime dependencies (fairseq/omegaconf/etc.) are handled by
# invoking AVH scripts via a user-selected `python_exe` (see UI sidebar).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── NOMA (audio-only) paths ────────────────────────────────────────────────────
# NOTE: the app currently supports the repo's bundled NOMA model directory
# `model/noma-1`. The original joblib artifact is not present in this repo.
NOMA_MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "model", "noma-1"),
]

# ─── AVH paths ────────────────────────────────────────────────────────────────
AVH_DIR = os.path.join(PROJECT_ROOT, "AVH")
AVH_TEST_SCRIPT = os.path.join(AVH_DIR, "test_video.py")
AVH_TEST_UNSUP_SCRIPT = os.path.join(AVH_DIR, "test_video_unsupervised.py")

# AVH-Align fusion checkpoint (FusionModel MLP head)
AVH_FUSION_CKPT = os.path.join(AVH_DIR, "checkpoints", "AVH-Align_AV1M.pt")

# AV-HuBERT checkpoint + required AV-HuBERT code/data inside this repo
AVH_AVHUBERT_DIR = os.path.join(AVH_DIR, "av_hubert", "avhubert")
AVH_FACE_PREDICTOR = os.path.join(
    AVH_AVHUBERT_DIR,
    "content",
    "data",
    "misc",
    "shape_predictor_68_face_landmarks.dat",
)
AVH_MEAN_FACE = os.path.join(
    AVH_AVHUBERT_DIR,
    "content",
    "data",
    "misc",
    "20words_mean_face.npy",
)
AVH_AVHUBERT_CKPT = os.path.join(AVH_AVHUBERT_DIR, "self_large_vox_433h.pt")

# Grad-CAM evidence script
AVH_GRADCAM_SCRIPT = os.path.join(AVH_DIR, "gradcam_mouth_roi.py")

# Grad-CAM / XAI defaults (batch tools, gradcam_avh.py fallbacks)
GRADCAM_DEFAULT_MAX_FUSION_FRAMES = 200
GRADCAM_DEFAULT_REGION_TRACK_STRIDE = 1
GRADCAM_DEFAULT_SELECTION_MODE = "top_k"
GRADCAM_DEFAULT_MIN_TEMPORAL_GAP = 24

# Streamlit Combined panel: compact, review-friendly (few strong overlays + spread in time)
STREAMLIT_GRADCAM_TOP_K = 6
STREAMLIT_GRADCAM_MAX_FUSION_FRAMES = 8
STREAMLIT_GRADCAM_SELECTION_MODE = "diverse_topk"
STREAMLIT_GRADCAM_MIN_TEMPORAL_GAP = 8
STREAMLIT_GRADCAM_REGION_TRACK_STRIDE = 1

# Late fusion: how to combine calibrated mean NOMA p(fake) with calibrated AVH p(fake).
# full = reliability_fusion.py (default); mean / audio_primary / video_primary = simple rules.
# learned = explainability/learned_reliability_fusion.py (params from calibration JSON or LEARNED_FUSION_PARAMS_PATH).
LATE_FUSION_MODES = frozenset({"full", "mean", "audio_primary", "video_primary", "learned"})


def get_noma_fake_class_label() -> int:
    """
    Sklearn class label used for Fake during NOMA training.

    Mozilla / project notebook convention: 0 = Fake, 1 = Real. If your joblib was
    trained with the opposite encoding, set env `NOMA_FAKE_CLASS_LABEL=1`.
    """
    raw = os.environ.get("NOMA_FAKE_CLASS_LABEL", "0").strip()
    try:
        v = int(raw)
    except ValueError:
        return 0
    return 1 if v == 1 else 0


def get_late_fusion_mode() -> str:
    # Default `full` = explainability/reliability_fusion.py (regime-based blend).
    # Override with LATE_FUSION_MODE=mean|audio_primary|video_primary for simpler rules.
    raw = os.environ.get("LATE_FUSION_MODE", "full").strip().lower()
    if raw in LATE_FUSION_MODES:
        return raw
    return "full"


# Allowed python executables for running AVH subprocesses.
# This prevents arbitrary code execution via user-supplied `python_exe`.
AVH_PYTHON_ALLOWLIST = [
    "/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python",
    os.path.expanduser("~/miniforge3/envs/avh/bin/python"),
    os.path.expanduser("~/miniconda3/envs/avh/bin/python"),
    os.path.expanduser("~/anaconda3/envs/avh/bin/python"),
]

_extra_py = os.environ.get("AVH_PYTHON_ALLOWLIST_EXTRA", "")
if _extra_py.strip():
    for _chunk in _extra_py.replace(";", ",").split(","):
        _p = _chunk.strip()
        if _p and os.path.isfile(_p):
            _ap = os.path.abspath(_p)
            allow_abs = [os.path.abspath(x) for x in AVH_PYTHON_ALLOWLIST]
            if _ap not in allow_abs:
                AVH_PYTHON_ALLOWLIST.append(_ap)

