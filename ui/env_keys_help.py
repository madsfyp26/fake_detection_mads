"""User-facing help when Serp / News / Fact Check API keys are missing."""

from __future__ import annotations

import streamlit as st


def render_missing_data_api_keys_hint(errors: list[str] | None) -> None:
    """
    Call when `sources_used['errors']` is non-empty from Serp/News/Fact Check fetches.
    """
    if not errors:
        return
    err_text = "; ".join(str(e) for e in errors if e)
    st.info(
        "**No third-party headlines or web snippets were loaded.** "
        "That is normal until you add API keys to the `.env` file in the project root and **restart Streamlit**.\n\n"
        "| Environment variable | What it unlocks |\n"
        "|---------------------|----------------|\n"
        "| `SERPAPI_API_KEY` | Google **web** results and **Google Lens** (one key for both, via SerpAPI) |\n"
        "| `NEWS_API_KEY` | **NewsAPI.org** article search |\n"
        "| `GOOGLE_FACT_CHECK_API_KEY` | **Google Fact Check Tools** (ClaimReview) on the Fact check page |\n\n"
        "**`GEMINI_API_KEY`** is separate: it only powers the written answer. It does not fetch news or web results.\n\n"
        f"*Server messages:* {err_text}"
    )
