"""SerpAPI Google search (organic results)."""

from __future__ import annotations

import os
from typing import Any

import httpx

SERPAPI_URL = "https://serpapi.com/search.json"


def fetch_serp_results(
    query: str,
    *,
    num: int = 5,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Returns (normalized_results, error_message).

    Each item: title, link, snippet (best-effort).
    """
    key = api_key or os.environ.get("SERPAPI_API_KEY", "").strip()
    if not key:
        return [], "SERPAPI_API_KEY not set"

    params = {
        "engine": "google",
        "q": query,
        "api_key": key,
        "num": min(max(num, 1), 10),
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(SERPAPI_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return [], f"SerpAPI request failed: {e}"

    organic = data.get("organic_results") or []
    out: list[dict[str, Any]] = []
    for row in organic[:num]:
        out.append(
            {
                "title": row.get("title") or "",
                "link": row.get("link") or "",
                "snippet": row.get("snippet") or "",
                "source": "serpapi",
            }
        )
    return out, None


def fetch_google_lens_results(
    query: str,
    *,
    num: int = 8,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    SerpAPI Google Lens (visual matches / related). Uses the same SERPAPI_API_KEY as organic search.

    Each item: title, link, snippet, source.
    """
    key = api_key or os.environ.get("SERPAPI_API_KEY", "").strip()
    if not key:
        return [], "SERPAPI_API_KEY not set"

    params = {
        "engine": "google_lens",
        "q": query,
        "api_key": key,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(SERPAPI_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return [], f"SerpAPI Google Lens request failed: {e}"

    out: list[dict[str, Any]] = []
    n = min(max(num, 1), 20)
    # Typical shapes: visual_matches, products, related_content
    for key_name in ("visual_matches", "products", "related_content"):
        block = data.get(key_name)
        if isinstance(block, list):
            for row in block[:n]:
                if not isinstance(row, dict):
                    continue
                title = row.get("title") or row.get("source") or row.get("name") or ""
                link = row.get("link") or row.get("url") or ""
                snippet = (row.get("snippet") or row.get("price") or row.get("subtitle") or "")[:500]
                if title or link:
                    out.append(
                        {
                            "title": str(title),
                            "link": str(link),
                            "snippet": str(snippet),
                            "source": "serpapi_google_lens",
                        }
                    )
            if out:
                break

    if not out and data.get("error"):
        return [], str(data.get("error"))
    return out, None
