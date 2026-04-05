import os
import tempfile

import numpy as np
import soundfile as sf

from detectors.audio_decode import decode_audio_to_mono_float32, resample_mono_once


def test_resample_mono_identity():
    y = np.linspace(-1, 1, 5000, dtype=np.float32)
    z = resample_mono_once(y, 22050, 22050)
    assert np.allclose(z, y)


def test_resample_rational_16k_to_22k():
    n = 16000
    y = np.sin(2 * np.pi * 440 * np.arange(n) / 16000).astype(np.float32)
    z = resample_mono_once(y, 16000, 22050)
    assert z.shape == (22050,)


def test_decode_soundfile_path_no_double_ffmpeg():
    sr = 22050
    n = int(sr * 0.5)
    y = 0.1 * np.sin(2 * np.pi * 300 * np.arange(n) / sr).astype(np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, y, sr)
        out = decode_audio_to_mono_float32(audio_path=path, target_sr=sr)
        assert out.shape == y.shape
        assert np.all(np.isfinite(out))
    finally:
        os.remove(path)
