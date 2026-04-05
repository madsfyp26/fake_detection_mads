"""Tests for fact-check orchestration (mocked APIs)."""

from unittest.mock import patch

import pytest


@patch("integrations.research_chat.factcheck_turn.synthesize_fact_check_verdict")
@patch("integrations.research_chat.factcheck_turn.fetch_factcheck_results")
@patch("integrations.research_chat.factcheck_turn.fetch_news_results")
@patch("integrations.research_chat.factcheck_turn.fetch_serp_results")
def test_run_fact_check_turn(mock_serp, mock_news, mock_fc, mock_gem):
    mock_serp.return_value = ([{"title": "t", "link": "http://x", "snippet": "s", "source": "serpapi"}], None)
    mock_news.return_value = ([], None)
    mock_fc.return_value = ([], None)
    mock_gem.return_value = ("## Verdict\nok", None)

    from integrations.research_chat.factcheck_turn import run_fact_check_turn

    t = run_fact_check_turn("test claim")
    assert "ok" in t.synthesis
    assert t.claim == "test claim"


def test_run_fact_check_empty_claim():
    from integrations.research_chat.factcheck_turn import run_fact_check_turn

    t = run_fact_check_turn("   ")
    assert t.error == "Empty claim."
