"""NewsAPI.org everything search."""

from __future__ import annotations

import os
from typing import Any

import httpx

NEWS_EVERYTHING_URL = "https://newsapi.org/v2/everything"


def fetch_news_results(
    query: str,
    *,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Returns (normalized_results, error_message).

    Each item: title, link, snippet (description).
    """
    key = api_key or os.environ.get("NEWS_API_KEY", "").strip()
    if not key:
        return [], "NEWS_API_KEY not set"

    params = {
        "q": query,
        "pageSize": min(max(page_size, 1), 20),
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": key,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(NEWS_EVERYTHING_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return [], f"News API request failed: {e}"

    if data.get("status") != "ok":
        return [], data.get("message") or "News API returned non-ok status"

    articles = data.get("articles") or []
    out: list[dict[str, Any]] = []
    for a in articles[:page_size]:
        out.append(
            {
                "title": a.get("title") or "",
                "link": a.get("url") or "",
                "snippet": (a.get("description") or "")[:500],
                "source": "newsapi",
            }
        )
    return out, None
