"""Gemini synthesis grounded on tool JSON (Serp organic, Google Lens via SerpAPI, NewsAPI)."""

from __future__ import annotations

import json
import os
from typing import Any

# Default: stable Flash for generate_content (text synthesis). Not a Live API model.
_DEFAULT_MODEL = "gemini-2.5-flash"


def _truncate_payload(obj: Any, max_chars: int = 12000) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + "\n... [truncated]"


def synthesize_research_answer(
    user_message: str,
    *,
    tool_bundle: dict[str, Any],
    detection_context: str | None,
    conversation_history: str,
    model: str | None = None,
) -> tuple[str, str | None]:
    """
    Returns (assistant_text, error_message).

    Grounds the reply in tool_bundle only; instructs model not to invent URLs.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "", "GEMINI_API_KEY not set"

    try:
        import google.generativeai as genai
    except ImportError as e:
        return "", f"google-generativeai not installed: {e}"

    genai.configure(api_key=api_key)
    name = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    payload_text = _truncate_payload(tool_bundle)
    ctx = (detection_context or "").strip()
    hist = (conversation_history or "").strip()

    system_prefix = """You are a careful research assistant.
You MUST base your answer ONLY on the JSON data provided:
- serp_organic_results: Google web results (title, link, snippet),
- google_lens_results: SerpAPI Google Lens visual/related matches (title, link, snippet),
- news_results: NewsAPI headlines (title, link, snippet).
List headlines and sources explicitly when present. If the JSON is empty or insufficient, say so and avoid inventing URLs.
Include a short summary, bullet points when useful, and a limitations line if evidence is weak.
Do not provide legal advice.

---

"""

    user_block = f"""{system_prefix}User question:
{user_message.strip()}

Optional prior context from a separate deepfake-detection run (may be empty):
{ctx if ctx else "[none]"}

Recent conversation (may be empty):
{hist if hist else "[none]"}

Retrieved evidence (JSON):
{payload_text}
"""

    try:
        model_obj = genai.GenerativeModel(name)
        resp = model_obj.generate_content(user_block)
        text = (resp.text or "").strip()
        if not text:
            return "", "Gemini returned empty text"
        return text, None
    except Exception as e:
        err = str(e)
        hint = ""
        low = err.lower()
        if "quota" in low or "resourceexhausted" in low or "429" in err:
            hint = (
                " For free tier, try `GEMINI_MODEL=gemini-2.5-flash-lite` or "
                "`gemini-3.1-flash-lite-preview`, wait and retry, or enable billing. "
                "https://ai.google.dev/gemini-api/docs/rate-limits"
            )
        return "", f"Gemini error ({name}): {err}{hint}"


def synthesize_ui_guide(
    *,
    section_id: str,
    section_title: str,
    guide_payload: dict[str, Any],
    model: str | None = None,
) -> tuple[str, str | None]:
    """
    Explain Combined-report UI: what each output is, why it exists, how to read tables/charts.

    Returns (markdown_text, error_message).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "", "GEMINI_API_KEY not set"

    try:
        import google.generativeai as genai
    except ImportError as e:
        return "", f"google-generativeai not installed: {e}"

    genai.configure(api_key=api_key)
    name = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    payload_text = _truncate_payload(guide_payload, max_chars=16000)

    if section_id == "full":
        focus = """Cover EVERY area in the glossary that has run_flags / run_values relevance:
1) User summary (verdict, fused p(fake), tension)
2) Score table signals
3) CMID status and CII
4) NOMA per-second table (how to read p_fake, Confidence)
5) Temporal corroboration table if present
6) Grad-CAM (purpose, overlays, JSON fields like T_use)
7) Fused heatmap and Grad-CAM intensity line (axes, how to read)
8) NOMA permutation if present
9) Research tab (external APIs vs model scores)
10) Export JSON

Use markdown headings (##). For each chart type, state the x-axis, y-axis or color meaning, and one sentence on misinterpretation."""
    elif section_id.startswith("xai_"):
        focus = f"""Standalone **Explainability** page (id={section_id}).
Explain the full page using `glossary` + `run_values` / `run_flags` in the JSON:
- For **audio**: CII variance plot (axes: index/time vs variance), what CII scalar means, and that Combined run is required.
- For **video**: Grad-CAM intensity vs time (t on x-axis), temporal inconsistency Δ_t, high-frequency energy, fused intensity, region tracks — for each, state axes and how to read spikes.
If `run_flags` shows missing Grad-CAM, tell the user to run Combined with Grad-CAM enabled.
Use markdown ## headings."""
    else:
        focus = f"""Focus ONLY on this section: "{section_title}" (id={section_id}).
Cover what it shows, why it exists, and how to read tables/charts. If data is missing in `run_flags`, say what to enable.
Use ### subheadings inside this section."""

    style_and_structure = """
Writing style (important):
- Use **plain, simple English**: short sentences, everyday words. A motivated high-school reader should follow it.
- Still include **real technical depth** in one dedicated part (see structure below). Name real components (e.g. AV-HuBERT, Fusion MLP, 1s NOMA blocks, reliability fusion) when the JSON/glossary supports it.

Required markdown structure:
## Plain summary
3–6 short bullets: what this output is for, in simple words.

## How to read each number, table, or chart
For each relevant item in the payload: what the rows/columns or axes mean, and one line on how to spot something suspicious vs normal.

## How it works (technical)
Explain the underlying mechanism: how signals are computed and combined (embeddings, calibration, fusion, Grad-CAM gradients on the mouth ROI, etc.). Use the glossary and `run_values`; do not invent modules not in the JSON.

## What this is not
2–4 sentences: screening / evidence of model focus — not a court verdict or guaranteed ground truth.

Tone: friendly, clear, precise. Define acronyms once (e.g. ROI = region of interest).
"""

    user_block = f"""You are explaining a deepfake **screening** lab UI (AVH lip–audio + NOMA speech + late fusion + optional Grad-CAM).

Hard rules:
- Ground numbers and labels in the JSON (`run_values`, `run_flags`, `ui_glossary`) — do not invent scores or file paths.
- Grad-CAM = **where the neural net looked** (sensitivity), not automatic proof of manipulation.
- Research / Serp / News APIs are separate from detector scores; if the payload mentions missing APIs, say keys live in `.env`.

{style_and_structure}

Task focus:
{focus}

Section label: {section_title}

Structured payload (glossary + this run):
{payload_text}
"""

    try:
        model_obj = genai.GenerativeModel(name)
        resp = model_obj.generate_content(user_block)
        text = (resp.text or "").strip()
        if not text:
            return "", "Gemini returned empty text"
        return text, None
    except Exception as e:
        err = str(e)
        hint = ""
        low = err.lower()
        if "quota" in low or "resourceexhausted" in low or "429" in err:
            hint = (
                " For free tier, try `GEMINI_MODEL=gemini-2.5-flash-lite` or wait and retry."
            )
        return "", f"Gemini error ({name}): {err}{hint}"


def synthesize_fact_check_verdict(
    claim: str,
    *,
    tool_bundle: dict[str, Any],
    model: str | None = None,
) -> tuple[str, str | None]:
    """
    Grounded fact-check narrative: map Serp / News / Google Fact Check rows to support vs refute vs unclear.

    Returns (markdown_text, error_message).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "", "GEMINI_API_KEY not set"

    try:
        import google.generativeai as genai
    except ImportError as e:
        return "", f"google-generativeai not installed: {e}"

    genai.configure(api_key=api_key)
    name = model or os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    payload_text = _truncate_payload(tool_bundle, max_chars=18000)
    c = (claim or "").strip()

    user_block = f"""You are a careful fact-checking assistant (not a court). Write in **plain English** (short sentences) but stay accurate.

Claim to assess (may come from speech-to-text; treat transcript as imperfect):
{c if c else "[empty]"}

Retrieved evidence (JSON only — do not invent URLs or publishers not listed):
{payload_text}

Use this markdown structure:

## Verdict (simple)
One line label: **Supported**, **Contradicted**, **Mixed / unclear**, or **Insufficient evidence** — from the JSON only.

## What each source type means (technical, short)
Briefly explain: **Serp organic** = ranked web snippets for the query; **News** = NewsAPI articles; **Google Fact Check** = ClaimReview entries when the API returns them. Say if a channel is empty in the JSON.

## Source-by-source
For each useful row: does it support, contradict, or stay neutral vs the claim? Quote titles; paste links exactly as given.

## Limitations
API gaps, English bias in news, recency, STT errors — in simple words.

Rules: no legal advice; if a channel has no rows, say so clearly.
"""

    try:
        model_obj = genai.GenerativeModel(name)
        resp = model_obj.generate_content(user_block)
        text = (resp.text or "").strip()
        if not text:
            return "", "Gemini returned empty text"
        return text, None
    except Exception as e:
        err = str(e)
        low = err.lower()
        hint = ""
        if "quota" in low or "resourceexhausted" in low or "429" in err:
            hint = " Try a smaller GEMINI_MODEL or wait."
        return "", f"Gemini error ({name}): {err}{hint}"
