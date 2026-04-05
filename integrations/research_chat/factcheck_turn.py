"""
Fact-check orchestration: Serp + News + Google Fact Check → Gemini verdict.

Optional: transcribe audio first via integrations.stt_gemini (Gemini multimodal).
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from integrations.research_chat.factcheck_client import fetch_factcheck_results
from integrations.research_chat.gemini_client import synthesize_fact_check_verdict
from integrations.research_chat.news_client import fetch_news_results
from integrations.research_chat.serp_client import fetch_serp_results


@dataclass
class FactCheckTurn:
    """One fact-check pass."""

    claim: str
    synthesis: str
    error: str | None = None
    transcript: str | None = None
    stt_error: str | None = None
    sources_used: dict[str, Any] = field(default_factory=dict)


def run_fact_check_turn(
    claim: str,
    *,
    num_serp: int = 6,
    num_news: int = 10,
    num_factcheck: int = 8,
) -> FactCheckTurn:
    """
    Fetch web + news + ClaimReview, then ask Gemini for a structured verdict vs the claim.

    `claim` should be a short declarative sentence when possible.
    """
    q = (claim or "").strip()
    if not q:
        return FactCheckTurn(claim="", synthesis="", error="Empty claim.")

    sources_used: dict[str, Any] = {
        "serp": [],
        "news": [],
        "google_factcheck": [],
        "errors": [],
    }

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_serp = ex.submit(fetch_serp_results, q, num=num_serp)
        f_news = ex.submit(fetch_news_results, q, page_size=num_news)
        f_fc = ex.submit(fetch_factcheck_results, q, page_size=num_factcheck)
        for name, fut in (("serp", f_serp), ("news", f_news), ("google_factcheck", f_fc)):
            try:
                rows, err = fut.result()
                sources_used[name] = rows
                if err:
                    sources_used["errors"].append(f"{name}: {err}")
            except Exception as e:  # noqa: BLE001
                sources_used["errors"].append(f"{name}: {e}")

    tool_bundle = {
        "claim_query": q,
        "serp_organic_results": sources_used["serp"],
        "news_results": sources_used["news"],
        "google_factcheck_claimreviews": sources_used["google_factcheck"],
    }

    text, err = synthesize_fact_check_verdict(q, tool_bundle=tool_bundle)
    return FactCheckTurn(
        claim=q,
        synthesis=text or "",
        error=err,
        sources_used=sources_used,
    )


def run_fact_check_with_optional_stt(
    *,
    claim_text: str | None,
    audio_bytes: bytes | None,
    audio_name: str | None,
    num_serp: int = 6,
    num_news: int = 10,
    num_factcheck: int = 8,
) -> FactCheckTurn:
    """
    If `audio_bytes` is set, transcribe with Gemini STT first, then merge with optional `claim_text`.
    """
    transcript: str | None = None
    stt_err: str | None = None
    parts: list[str] = []

    if audio_bytes and len(audio_bytes) > 0:
        from integrations.stt_gemini import transcribe_audio_stream

        name = (audio_name or "upload.wav").strip()
        transcript, stt_err = transcribe_audio_stream(io.BytesIO(audio_bytes), filename=name)
        if transcript and "[no speech detected]" not in transcript.lower():
            parts.append(f"[From audio] {transcript}")

    ct = (claim_text or "").strip()
    if ct:
        parts.append(f"[From text] {ct}")

    merged = " ".join(parts).strip()
    if not merged:
        return FactCheckTurn(
            claim="",
            synthesis="",
            error="Provide either typed claim text or an audio clip.",
            transcript=transcript,
            stt_error=stt_err,
            sources_used={"errors": ["no claim"]},
        )

    # Use a focused search query: prefer short user text, else first ~200 chars of transcript
    search_q = ct if ct else (transcript or merged)[:280]

    turn = run_fact_check_turn(search_q, num_serp=num_serp, num_news=num_news, num_factcheck=num_factcheck)
    turn.transcript = transcript
    turn.stt_error = stt_err
    if ct and transcript:
        turn.claim = f"{ct}\n\n---\nSTT excerpt: {transcript[:800]}"
    elif transcript and not ct:
        turn.claim = transcript[:2000]
    return turn
