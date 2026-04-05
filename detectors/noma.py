import io
import math
import os
import hashlib
import tempfile
from functools import lru_cache
from typing import Any

import numpy as np

from config import NOMA_MODEL_CANDIDATES, PROJECT_ROOT, get_noma_fake_class_label
from logging_utils import get_logger, log_timed
from metrics import inc_counter, observe_latency_ms


TARGET_SR = 22050
BLOCK_SECONDS = 1.0
EPS = 1e-9
NOMA_FEATURE_IMPL_VERSION = 2
NOMA_CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache", "noma")


FEATURE_NAMES: list[str] = (
    ["Chroma", "RMS", "Centroid", "Bandwidth", "Rolloff", "ZCR", "Tonnetz", "Contrast"]
    + [f"MFCC{i}" for i in range(1, 21)]
    + [f"IMFCC{i}" for i in range(1, 14)]
)


def get_noma_model_path() -> str | None:
    """
    Return the first existing NOMA model candidate path.

    The Streamlit UI expects either:
    - a model artifact path (e.g. `model/noma-1`).
    """

    for p in NOMA_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


@lru_cache(maxsize=2)
def _load_noma_pipeline(model_path: str) -> Any:
    """
    Load the joblib artifact (stored as a dict with key `pipeline`).
    Cached to avoid re-loading in the same process.
    """

    import joblib

    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    # Backward/forward compatibility if artifact is directly the pipeline.
    return obj


def get_noma_pipeline(model_path: str) -> Any:
    """
    Public accessor for the cached NOMA sklearn pipeline.
    """

    return _load_noma_pipeline(model_path)


def _final_estimator_classes(pipeline: Any) -> np.ndarray:
    """Binary sklearn `classes_` in column order for `predict_proba`."""
    if hasattr(pipeline, "steps"):
        est = pipeline.steps[-1][1]
    else:
        est = pipeline
    c = getattr(est, "classes_", None)
    if c is None:
        return np.array([0, 1], dtype=int)
    return np.asarray(c)


def noma_fake_proba_column_index(pipeline: Any) -> int:
    """
    Column index of P(fake) in `predict_proba`, using `classes_` and `get_noma_fake_class_label()`.
    """
    classes = _final_estimator_classes(pipeline)
    fake_label = int(get_noma_fake_class_label())
    idx = np.where(classes == fake_label)[0]
    if idx.size != 1:
        raise ValueError(
            f"NOMA classes_={classes.tolist()} must contain fake label {fake_label} exactly once; "
            "set NOMA_FAKE_CLASS_LABEL to 0 or 1 to match training."
        )
    return int(idx[0])


def noma_p_fake_raw_confidence_and_preds_from_probas(
    pipeline: Any,
    probas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Map raw sklearn `predict_proba` (n, 2) to p(fake) raw, max-proba confidence, and Fake/Real labels.

    Uses `classes_` so argmax labels stay consistent when column 0 is Real and column 1 is Fake.
    """
    probas = np.asarray(probas, dtype=float)
    if probas.ndim != 2 or probas.shape[1] != 2:
        raise ValueError(f"Expected probas shape (n, 2), got {probas.shape}")
    classes = _final_estimator_classes(pipeline)
    fake_label = int(get_noma_fake_class_label())
    fake_col = noma_fake_proba_column_index(pipeline)
    p_fake_raw = probas[:, fake_col]
    confidences = probas.max(axis=1)
    win = probas.argmax(axis=1)
    pred_labels = classes[win]
    preds_str: list[str] = []
    for pl in np.asarray(pred_labels).ravel():
        preds_str.append("Fake" if int(pl) == fake_label else "Real")
    return p_fake_raw, confidences, preds_str


@lru_cache(maxsize=8)
def _sha256_file_cached(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(bs: bytes) -> str:
    h = hashlib.sha256()
    h.update(bs)
    return h.hexdigest()


def _decode_audio_to_waveform(
    *,
    audio_path: str | None = None,
    audio_bytes: io.BytesIO | None = None,
    audio_filename: str | None = None,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """Decode to mono float32 @ `target_sr` (single resample; see `detectors/audio_decode.py`)."""
    from detectors.audio_decode import decode_audio_to_mono_float32

    y = decode_audio_to_mono_float32(
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        target_sr=int(target_sr),
        timeout_s=600,
    )
    if y is None or len(y) == 0:
        raise RuntimeError("Decoded waveform is empty.")
    return np.asarray(y, dtype=np.float32)


def _compute_imfcc(
    y: np.ndarray,
    *,
    sr: int = TARGET_SR,
    n_imfcc: int = 13,
    n_fft: int = 2048,
) -> np.ndarray:
    import librosa
    from scipy import fft

    S = np.abs(librosa.stft(y, n_fft=n_fft))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128)
    mel_basis_inverted = mel_basis[::-1, :]
    mel_spec = np.dot(mel_basis_inverted, S)
    log_mel = np.log(mel_spec + EPS)
    imfccs = fft.dct(log_mel, axis=0, norm="ortho")[:n_imfcc]
    return imfccs


def _extract_features_from_array(y: np.ndarray, *, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract the 41-feature vector used by the NOMA SVC.
    Order must match `FEATURE_NAMES`.
    """

    import librosa

    y = np.asarray(y, dtype=np.float32)
    # Librosa STFT defaults use n_fft=2048; very short segments (edge clips / decode quirks)
    # trigger warnings and unstable stats — pad to at least one full STFT window.
    _min_win = 2048
    if len(y) < _min_win:
        y = np.pad(y, (0, _min_win - len(y)), mode="constant")

    # These come straight from `fake_audio_detection_pipeline.ipynb` reference code.
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    imfccs = _compute_imfcc(y, sr=sr, n_imfcc=13)

    features = np.hstack(
        [
            np.mean(chroma_stft),
            np.mean(rms),
            np.mean(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(rolloff),
            np.mean(zcr),
            np.mean(tonnetz),
            np.mean(spectral_contrast),
            np.mean(mfccs, axis=1),
            np.mean(imfccs, axis=1),
        ]
    ).astype(np.float64)

    if features.shape[0] != 41:
        raise RuntimeError(f"Expected 41 features, got {features.shape[0]}")
    return features


def _split_waveform_into_block_times(
    *,
    n_samples: int,
    sr: int = TARGET_SR,
    block_seconds: float = BLOCK_SECONDS,
) -> np.ndarray:
    block_len = int(round(sr * block_seconds))
    n_blocks = int(math.ceil(n_samples / block_len)) if block_len > 0 else 0
    if n_blocks <= 0:
        n_blocks = 1
    return np.arange(n_blocks, dtype=float) * float(block_seconds)


def _extract_feature_matrix_from_waveform(
    *,
    y: np.ndarray,
    sr: int = TARGET_SR,
    block_seconds: float = BLOCK_SECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (times_seconds, feature_matrix) with one 41-D vector per 1-second block.
    """

    block_len = int(round(sr * block_seconds))
    n_blocks = int(math.ceil(len(y) / block_len)) if block_len > 0 else 1
    n_blocks = max(n_blocks, 1)

    times = np.arange(n_blocks, dtype=float) * float(block_seconds)
    X = np.zeros((n_blocks, 41), dtype=np.float64)

    for bi in range(n_blocks):
        start = bi * block_len
        end = start + block_len
        block = y[start:end]
        if len(block) < block_len:
            block = np.pad(block, (0, block_len - len(block)), mode="constant")
        X[bi] = _extract_features_from_array(block, sr=sr)

    return times, X


def run_noma_prediction_with_features(
    model_path: str,
    audio_path: str | None = None,
    audio_bytes: io.BytesIO | None = None,
    audio_filename: str | None = None,
    *,
    block_seconds: float = BLOCK_SECONDS,
    target_sr: int = TARGET_SR,
    return_cache_hit: bool = False,
) -> Any:
    """
    Local NOMA inference that also returns the per-block feature matrix.

    Returns:
      times_seconds: (n_blocks,)
      probas: (n_blocks, 2) sklearn `predict_proba` columns aligned with `pipeline.classes_`
      feature_matrix: (n_blocks, 41)
      feature_names: (41,)
    """

    logger = get_logger("detectors.noma")
    pipeline = _load_noma_pipeline(model_path)

    # Cache (decode -> features -> SVM proba) so repeated runs are fast.
    os.makedirs(NOMA_CACHE_DIR, exist_ok=True)
    model_hash = _sha256_file_cached(model_path)
    if audio_path:
        input_hash = _sha256_file_cached(audio_path)
    elif audio_bytes is not None:
        input_hash = _sha256_bytes(bytes(audio_bytes.getbuffer()))
    else:
        input_hash = "missing_input"

    cache_key = f"{input_hash}|{model_hash}|sr={target_sr}|block_s={block_seconds}|v={NOMA_FEATURE_IMPL_VERSION}"
    cache_key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    cache_path = os.path.join(NOMA_CACHE_DIR, f"pred_{cache_key_hash}.npz")

    if os.path.isfile(cache_path):
        cached = np.load(cache_path)
        times = cached["times"]
        probas = cached["probas"]
        X = cached["X"]
        if probas.shape[0] == times.shape[0] and X.shape == (times.shape[0], 41):
            inc_counter("noma_cache_hit", stage="noma")
            if return_cache_hit:
                return times, probas, X, FEATURE_NAMES, True
            return times, probas, X, FEATURE_NAMES

    with log_timed(logger, "noma_infer", cache_hit=False):
        y = _decode_audio_to_waveform(
            audio_path=audio_path,
            audio_bytes=audio_bytes,
            audio_filename=audio_filename,
            target_sr=target_sr,
        )
        times, X = _extract_feature_matrix_from_waveform(
            y=y,
            sr=target_sr,
            block_seconds=block_seconds,
        )

        X = np.asarray(X, dtype=np.float64)
        probas = pipeline.predict_proba(X)
        if probas.shape != (len(times), 2):
            raise RuntimeError(f"Unexpected probas shape: {probas.shape}")

    # Best-effort cache write (avoid failing inference if disk cache fails).
    tmp_path = cache_path + ".tmp"
    try:
        np.savez_compressed(tmp_path, times=times, probas=probas, X=X)
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            if os.path.isfile(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass

    if return_cache_hit:
        return times, probas, X, FEATURE_NAMES, False
    return times, probas, X, FEATURE_NAMES


def run_noma_prediction(
    model_path: str,
    audio_path: str | None = None,
    audio_bytes: io.BytesIO | None = None,
    audio_filename: str | None = None,
):
    """
    Returns:
      times_seconds, probas (per-block probabilities) for 1-second blocks.
    """

    times, probas, _X, _names = run_noma_prediction_with_features(
        model_path,
        audio_path=audio_path,
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
    )
    return times, probas

