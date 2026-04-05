"""
Optional wav2vec2 embedding stats from mono audio (for fusion / analysis).

Requires: pip install transformers torch (heavy). If unavailable, returns ok=False.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def wav2vec_embedding_proxy(
    audio_path: str | None,
    *,
    max_seconds: float = 120.0,
    model_name: str = "facebook/wav2vec2-base-960h",
) -> dict[str, Any]:
    """
    Mean L2 norm of frame embeddings as a simple scalar feature (not a trained classifier).
    """
    if not audio_path or not os.path.isfile(audio_path):
        return {"ok": False, "error": "no_audio_path", "embedding_norm_mean": None}

    if os.environ.get("WAV2VEC_DISABLED", "").strip() == "1":
        return {"ok": False, "error": "WAV2VEC_DISABLED=1", "embedding_norm_mean": None}

    try:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
    except ImportError as e:
        return {"ok": False, "error": f"transformers/torch: {e}", "embedding_norm_mean": None}

    try:
        import soundfile as sf
    except ImportError as e:
        return {"ok": False, "error": f"soundfile: {e}", "embedding_norm_mean": None}

    try:
        y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    except Exception as e:
        return {"ok": False, "error": str(e), "embedding_norm_mean": None}

    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32).ravel()
    max_samples = int(min(len(y), float(max_seconds) * float(sr)))
    if max_samples < 400:
        return {"ok": False, "error": "too_short", "embedding_norm_mean": None}
    y = y[:max_samples]

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()
    # Resample to 16 kHz expected by many checkpoints
    target_sr = 16000
    if sr != target_sr:
        try:
            import librosa

            y = librosa.resample(y, orig_sr=int(sr), target_sr=target_sr).astype(np.float32)
        except Exception as e:
            return {"ok": False, "error": f"resample: {e}", "embedding_norm_mean": None}
    else:
        y = y.astype(np.float32)

    with torch.no_grad():
        inputs = processor(y, sampling_rate=target_sr, return_tensors="pt", padding=True)
        out = model(**inputs)
        hidden = out.last_hidden_state.squeeze(0).numpy()  # (T, D)
        norms = np.linalg.norm(hidden, axis=1)
        mean_norm = float(np.mean(norms))

    return {"ok": True, "error": None, "embedding_norm_mean": mean_norm, "model_name": model_name}
