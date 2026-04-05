"""
High-fidelity mono audio decode for NOMA and related pipelines.

Conversion trace (historical → current)
--------------------------------------
**Previous NOMA path (removed):**
  container/WAV → ffmpeg PCM WAV @ target_sr → librosa.load(@ target_sr)
  That forced **two** passes over the samples: ffmpeg resampled, then librosa could
  resample/normalize again.

**Current path:**
  1) **WAV/FLAC/AIFF** readable by libsndfile: `soundfile.read` → float mono in *file* SR.
     At most **one** resample to `target_sr` via `scipy.signal.resample_poly` when the
     ratio is rational, else `librosa.resample(..., res_type="kaiser_best")`.
  2) **Everything else** (and fallback): **single** `ffmpeg` decode to raw **f32le**
     mono @ `target_sr` into memory (no intermediate WAV file).

Loss / quality
---------------
- Demux + decode from video/audio containers is **always** a decode step (not “lossless”
  vs the compressed bitstream; you recover PCM).
- Resampling is **lossy** in the DSP sense; we avoid **double** resampling.
- No gratuitous peak normalization (preserves level; avoids artificial clipping).

Video
------
Video frame decoding stays in AV-HuBERT / preprocessing (skvideo + ROI encode). This
module only handles **audio** for NOMA. PyAV/decord would replace *video* readers in AVH,
not this path.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from typing import BinaryIO

import numpy as np


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def resample_mono_once(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float waveform once; prefers polyphase for rational ratios."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")
    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    g = _gcd(int(orig_sr), int(target_sr))
    up = int(target_sr // g)
    down = int(orig_sr // g)
    if up > 0 and down > 0 and orig_sr * up // down == target_sr:
        from scipy import signal

        return signal.resample_poly(y, up, down).astype(np.float32)
    import librosa

    return librosa.resample(
        y.astype(np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="kaiser_best",
    ).astype(np.float32)


def _to_mono_float32(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        return y.mean(axis=1).astype(np.float32)
    raise ValueError("Expected 1D or 2D audio array.")


def _try_soundfile_read(path: str) -> tuple[np.ndarray, int] | tuple[None, None]:
    try:
        import soundfile as sf

        y, sr = sf.read(path, always_2d=False)
    except (OSError, RuntimeError, ValueError):
        return None, None
    if y is None or len(y) == 0:
        return None, None
    y = _to_mono_float32(np.asarray(y, dtype=np.float32))
    return y, int(sr)


def _ffmpeg_f32le_mono(path: str, target_sr: int, timeout_s: int) -> np.ndarray:
    """Decode any ffmpeg-supported file to mono f32le @ target_sr via one ffmpeg pass."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-vn",
        "-map",
        "0:a:0?",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg decode timed out: {path}") from e
    if p.returncode != 0:
        err = (p.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg decode failed (rc={p.returncode}): {err[:2000]}")
    raw = p.stdout
    if not raw or len(raw) % 4 != 0:
        raise RuntimeError("ffmpeg returned empty or misaligned f32le audio.")
    y = np.frombuffer(raw, dtype=np.float32).copy()
    if y.size == 0:
        raise RuntimeError("Decoded waveform is empty.")
    return y


def decode_audio_to_mono_float32(
    *,
    audio_path: str | None = None,
    audio_bytes: BinaryIO | None = None,
    audio_filename: str | None = None,
    target_sr: int,
    timeout_s: int = 600,
) -> np.ndarray:
    """
    Return mono float32 PCM at exactly `target_sr`.

    Prefer soundfile + one resample for WAV-like inputs; otherwise one ffmpeg decode.
    """
    if not audio_path and audio_bytes is None:
        raise ValueError("Provide either `audio_path` or `audio_bytes`.")

    input_tmp_path: str | None = None
    try:
        if audio_path:
            path = os.path.abspath(audio_path)
        else:
            suffix = ".wav"
            if audio_filename:
                _, ext = os.path.splitext(audio_filename)
                if ext:
                    suffix = ext.lower()
            t = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            input_tmp_path = t.name
            t.write(audio_bytes.getbuffer())  # type: ignore[union-attr]
            t.flush()
            t.close()
            path = input_tmp_path

        y, sr = _try_soundfile_read(path)
        if y is not None and sr is not None:
            return resample_mono_once(y, sr, target_sr)

        return _ffmpeg_f32le_mono(path, target_sr, timeout_s=timeout_s)
    finally:
        if input_tmp_path and os.path.isfile(input_tmp_path):
            try:
                os.remove(input_tmp_path)
            except OSError:
                pass


def decode_bytes_to_mono_float32(
    data: bytes,
    *,
    suffix: str,
    target_sr: int,
    timeout_s: int = 600,
) -> np.ndarray:
    """Convenience wrapper for raw bytes (e.g. uploads)."""
    return decode_audio_to_mono_float32(
        audio_bytes=io.BytesIO(data),
        audio_filename=f"upload{suffix}",
        target_sr=target_sr,
        timeout_s=timeout_s,
    )
