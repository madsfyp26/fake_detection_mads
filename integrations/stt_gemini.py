"""
Speech-to-text via Gemini multimodal (uses GEMINI_API_KEY).

Writes uploaded bytes to a temp file and uses google.generativeai.upload_file.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, BinaryIO

_DEFAULT_MODEL = "gemini-2.5-flash"


def transcribe_audio_bytes(
    data: bytes,
    *,
    suffix: str = ".wav",
    mime_hint: str = "audio/wav",
) -> tuple[str, str | None]:
    """
    Transcribe speech in audio bytes. Returns (transcript, error).

    suffix: temp file extension (.wav, .mp3, .ogg, .m4a) — should match content.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "", "GEMINI_API_KEY not set"

    try:
        import google.generativeai as genai
    except ImportError as e:
        return "", f"google-generativeai not installed: {e}"

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        return _transcribe_path(genai, model_name, path, mime_hint=mime_hint)
    finally:
        try:
            if fd >= 0:
                os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass


def transcribe_audio_stream(stream: BinaryIO, *, filename: str) -> tuple[str, str | None]:
    """Infer suffix from filename and transcribe."""
    low = filename.lower()
    if low.endswith(".mp3"):
        return transcribe_audio_bytes(stream.read(), suffix=".mp3", mime_hint="audio/mpeg")
    if low.endswith(".ogg") or low.endswith(".oga"):
        return transcribe_audio_bytes(stream.read(), suffix=".ogg", mime_hint="audio/ogg")
    if low.endswith(".m4a") or low.endswith(".aac"):
        return transcribe_audio_bytes(stream.read(), suffix=".m4a", mime_hint="audio/mp4")
    return transcribe_audio_bytes(stream.read(), suffix=".wav", mime_hint="audio/wav")


def _transcribe_path(genai: Any, model_name: str, path: str, *, mime_hint: str) -> tuple[str, str | None]:
    try:
        try:
            audio = genai.upload_file(path=path, mime_type=mime_hint)
        except TypeError:
            audio = genai.upload_file(path=path)
        # Wait until file is active (required for some accounts)
        for _ in range(30):
            if getattr(audio, "state", None) and getattr(audio.state, "name", "") == "ACTIVE":
                break
            if getattr(audio, "state", None) and getattr(audio.state, "name", "") == "FAILED":
                return "", "Gemini file upload failed (processing)"
            time.sleep(0.5)
            try:
                audio = genai.get_file(audio.name)
            except Exception:
                break

        model = genai.GenerativeModel(model_name)
        prompt = (
            "Transcribe all spoken words in this audio. "
            "Output only the transcript in plain text, preserving the spoken language. "
            "If there is no speech, reply with: [no speech detected]."
        )
        resp = model.generate_content([prompt, audio])
        text = (resp.text or "").strip()
        if not text:
            return "", "Gemini returned empty transcript"
        try:
            genai.delete_file(audio.name)
        except Exception:
            pass
        return text, None
    except Exception as e:
        return "", f"Transcription failed: {e}"
