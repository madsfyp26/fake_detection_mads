"""Mocked tests for research chat orchestrator (no real API calls)."""

from __future__ import annotations

def test_run_research_turn_empty_message():
    from integrations.research_chat.chat_orchestrator import run_research_turn

    t = run_research_turn("   ")
    assert t.error
    assert "Empty" in t.error


def test_run_research_turn_missing_gemini_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.fetch_serp_results",
        lambda q, **kw: ([{"title": "a", "link": "http://x", "snippet": "s", "source": "serpapi"}], None),
    )
    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.fetch_news_results",
        lambda q, **kw: ([], "NEWS_API_KEY not set"),
    )
    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.fetch_google_lens_results",
        lambda q, **kw: ([{"title": "l", "link": "http://l", "snippet": "", "source": "lens"}], None),
    )

    from integrations.research_chat.chat_orchestrator import run_research_turn

    t = run_research_turn("test query")
    assert t.error
    assert "GEMINI_API_KEY" in t.error
    assert t.sources_used is not None
    assert "errors" in t.sources_used


def test_run_research_turn_with_mocked_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.fetch_serp_results",
        lambda q, **kw: ([{"title": "t", "link": "http://a", "snippet": "sn", "source": "serpapi"}], None),
    )
    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.fetch_news_results",
        lambda q, **kw: ([{"title": "n", "link": "http://b", "snippet": "d", "source": "newsapi"}], None),
    )
    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.fetch_google_lens_results",
        lambda q, **kw: ([{"title": "lens", "link": "http://l", "snippet": "", "source": "lens"}], None),
    )
    def _fake_synth(user_message: str, **kw):
        return ("Summary from mocked Gemini.", None)

    monkeypatch.setattr(
        "integrations.research_chat.chat_orchestrator.synthesize_research_answer",
        _fake_synth,
    )

    from integrations.research_chat.chat_orchestrator import run_research_turn

    t = run_research_turn("hello world", detection_context="ctx", history=[])
    assert t.error is None
    assert "mocked" in t.text.lower() or "Summary" in t.text
    assert t.sources_used["serp"]


def test_format_detection_context_from_combined():
    from integrations.research_chat.chat_orchestrator import format_detection_context_from_combined

    assert format_detection_context_from_combined(None) == ""
    s = format_detection_context_from_combined(
        {"avh_ok": True, "avh_score": 0.5, "p_fused": 0.6, "fusion_tension": 0.1, "fusion_verdict": "Uncertain"}
    )
    assert "AVH" in s and "0.6" in s
    s2 = format_detection_context_from_combined(
        {"avh_ok": True, "avh_score": 0.5, "p_fused": 0.6},
        cam_idx={"xai_status": {"temporal_inconsistency": "computed"}, "T_use": 120},
    )
    assert "Grad-CAM" in s2 and "120" in s2
