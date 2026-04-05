"""Orchestrate Serp + Google Lens (SerpAPI) + News + Gemini for one chat turn."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from integrations.research_chat.gemini_client import synthesize_research_answer
from integrations.research_chat.news_client import fetch_news_results
from integrations.research_chat.serp_client import fetch_google_lens_results, fetch_serp_results


@dataclass
class AssistantTurn:
    text: str
    error: str | None = None
    sources_used: dict[str, Any] = field(default_factory=dict)


def _format_history_for_prompt(history: list[dict[str, str]], max_turns: int = 6) -> str:
    """history items: {\"role\": \"user\"|\"assistant\", \"content\": str}"""
    if not history:
        return ""
    tail = history[-max_turns:]
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def run_research_turn(
    user_text: str,
    *,
    detection_context: str | None = None,
    history: list[dict[str, str]] | None = None,
    num_serp: int = 5,
    num_news: int = 8,
    num_lens: int = 8,
) -> AssistantTurn:
    """
    Fetch external evidence in parallel, then ask Gemini to synthesize.

    Uses: SerpAPI organic web, SerpAPI Google Lens, NewsAPI.org. Google Fact Check removed.
    """
    q = (user_text or "").strip()
    if not q:
        return AssistantTurn(text="", error="Empty message.")

    hist = history or []
    history_prompt = _format_history_for_prompt(hist)

    sources_used: dict[str, Any] = {
        "serp": [],
        "google_lens": [],
        "news": [],
        "errors": [],
    }

    with ThreadPoolExecutor(max_workers=3) as ex:
        f_serp = ex.submit(fetch_serp_results, q, num=num_serp)
        f_lens = ex.submit(fetch_google_lens_results, q, num=num_lens)
        f_news = ex.submit(fetch_news_results, q, page_size=num_news)
        for name, fut in (("serp", f_serp), ("google_lens", f_lens), ("news", f_news)):
            try:
                rows, err = fut.result()
                sources_used[name] = rows
                if err:
                    sources_used["errors"].append(f"{name}: {err}")
            except Exception as e:
                sources_used["errors"].append(f"{name}: {e}")

    tool_bundle = {
        "query": q,
        "serp_organic_results": sources_used["serp"],
        "google_lens_results": sources_used["google_lens"],
        "news_results": sources_used["news"],
    }

    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return AssistantTurn(
            text="",
            error="GEMINI_API_KEY is not set. Add it to your environment to enable synthesis.",
            sources_used=sources_used,
        )

    text, err = synthesize_research_answer(
        q,
        tool_bundle=tool_bundle,
        detection_context=detection_context,
        conversation_history=history_prompt,
    )
    if err:
        return AssistantTurn(text="", error=err, sources_used=sources_used)

    return AssistantTurn(text=text, error=None, sources_used=sources_used)


def format_detection_context_from_combined(
    res: dict[str, Any] | None,
    cam_idx: dict[str, Any] | None = None,
) -> str:
    """Build a short string from Combined pipeline result for Gemini context."""
    if not isinstance(res, dict) or not res.get("avh_ok"):
        return ""
    parts: list[str] = []
    if res.get("avh_score") is not None:
        parts.append(f"AVH score (raw): {float(res['avh_score']):.4f}")
    if res.get("p_fused") is not None:
        parts.append(f"Late-fused p(fake): {float(res['p_fused']):.3f}")
    if res.get("fusion_verdict"):
        parts.append(f"Fusion verdict: {res['fusion_verdict']}")
    if res.get("fusion_tension") is not None:
        parts.append(f"AVH vs NOMA tension: {float(res['fusion_tension']):.3f}")
    if isinstance(cam_idx, dict):
        if cam_idx.get("T_use") is not None:
            parts.append(f"Grad-CAM T_use={int(cam_idx['T_use'])} frames")
        xs = cam_idx.get("xai_status") or {}
        if isinstance(xs, dict) and xs.get("temporal_inconsistency"):
            parts.append(f"Grad-CAM temporal_inconsistency: {xs['temporal_inconsistency']}")
    return " | ".join(parts)
