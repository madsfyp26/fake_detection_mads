"""Google Fact Check Tools API (ClaimReview search)."""

from __future__ import annotations

import os
from typing import Any

import httpx

FACTCHECK_SEARCH_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def fetch_factcheck_results(
    query: str,
    *,
    page_size: int = 5,
    api_key: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Returns (normalized_results, error_message).

    Each item: claim, claimant, rating, link, snippet.
    """
    key = api_key or os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "").strip()
    if not key:
        return [], "GOOGLE_FACT_CHECK_API_KEY not set"

    params = {
        "query": query,
        "pageSize": min(max(page_size, 1), 20),
        "key": key,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(FACTCHECK_SEARCH_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return [], f"Google Fact Check API request failed: {e}"

    claims = data.get("claims") or []
    out: list[dict[str, Any]] = []
    max_items = page_size * 3
    for c in claims:
        text = c.get("text") or ""
        claimant = c.get("claimant") or ""
        reviews = c.get("claimReview") or []
        if reviews:
            for rev in reviews:
                if len(out) >= max_items:
                    break
                rating = rev.get("textualRating") or rev.get("title") or ""
                url = rev.get("url") or ""
                publisher = (rev.get("publisher") or {}).get("name") or ""
                out.append(
                    {
                        "claim": text,
                        "claimant": claimant,
                        "rating": rating,
                        "link": url,
                        "snippet": f"{publisher} — {rating}"[:500] if publisher or rating else text[:500],
                        "source": "google_factcheck",
                    }
                )
        else:
            if len(out) >= max_items:
                break
            out.append(
                {
                    "claim": text,
                    "claimant": claimant,
                    "rating": "",
                    "link": "",
                    "snippet": "Claim listed; no ClaimReview in response.",
                    "source": "google_factcheck",
                }
            )

    return out[:max_items], None
