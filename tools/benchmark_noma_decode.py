#!/usr/bin/env python3
"""
Benchmark NOMA decode + inference after audio_decode refactor.

Run from repo root:
  PYTHONPATH=. python tools/benchmark_noma_decode.py

Prints wall time for `decode_audio_to_mono_float32` and mean NOMA p(fake) on a
synthetic WAV. Use this to compare decode latency across machines; prediction
values change when `NOMA_FEATURE_IMPL_VERSION` or the model changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import soundfile as sf

from detectors.audio_decode import decode_audio_to_mono_float32
from detectors.noma import (
    NOMA_FEATURE_IMPL_VERSION,
    get_noma_model_path,
    get_noma_pipeline,
    noma_fake_proba_column_index,
    run_noma_prediction_with_features,
)


def _tone_wav(path: str, sr: int = 22050, sec: float = 2.0) -> None:
    n = int(sr * sec)
    t = np.arange(n) / sr
    rng = np.random.default_rng(0)
    y = (
        0.25 * np.sin(2 * np.pi * 440 * t)
        + 0.05 * np.sin(2 * np.pi * 880 * t)
        + 0.01 * rng.standard_normal(n)
    )
    y = np.clip(y, -1, 1).astype(np.float32)
    sf.write(path, y, sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    model_path = get_noma_model_path()
    if not model_path:
        raise SystemExit("NOMA model not found.")

    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _tone_wav(wav, sr=22050, sec=2.0)

        dt_decode = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            decode_audio_to_mono_float32(audio_path=wav, target_sr=22050)
            dt_decode.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        times, probas, X, names, hit = run_noma_prediction_with_features(
            model_path,
            audio_path=wav,
            return_cache_hit=True,
        )
        t_infer = time.perf_counter() - t0

        pipe = get_noma_pipeline(model_path)
        fc = noma_fake_proba_column_index(pipe)
        pf = probas[:, fc].astype(float)
        print("benchmark_noma_decode")
        print(f"  NOMA_FEATURE_IMPL_VERSION={NOMA_FEATURE_IMPL_VERSION}")
        print(f"  decode_audio_to_mono_float32: mean_ms={1000 * float(np.mean(dt_decode)):.2f}")
        print(f"  run_noma_prediction_with_features (cold cache): infer_s={t_infer:.3f} cache_hit={hit}")
        print(f"  n_blocks={len(times)} mean_p_fake={float(np.mean(pf)):.4f}")
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


if __name__ == "__main__":
    main()
