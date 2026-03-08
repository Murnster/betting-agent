"""Tests for Discord webhook notification module."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch


from betting_agent.intelligence.picks import BetCandidate
from betting_agent.notifications.discord import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_GREY,
    COLOR_RED,
    MAX_EMBEDS_PER_MESSAGE,
    _build_alltime_sport_embed,
    _build_breakdown_embed,
    _build_pick_embed,
    _build_results_embed,
    _build_summary_embed,
    _get_webhook_url,
    _send_webhook,
    is_discord_configured,
    send_alltime_to_discord,
    send_picks_to_discord,
    send_results_to_discord,
)


def _make_candidate(**overrides) -> BetCandidate:
    defaults = dict(
        game_id=1,
        home_team="KC",
        away_team="BUF",
        game_date=date(2026, 1, 15),
        sport="NFL",
        bet_type="moneyline",
        pick_side="KC",
        model_prob=0.62,
        implied_prob=0.50,
        edge=0.08,
        odds=-110,
        kelly_fraction=0.03,
        recommended_bet=30.0,
        bankroll_at_pick=1000.0,
    )
    defaults.update(overrides)
    return BetCandidate(**defaults)


# ---------------------------------------------------------------------------
# _get_webhook_url / is_discord_configured
# ---------------------------------------------------------------------------


def test_get_webhook_url_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_NFL_PICKS", raising=False)
    assert _get_webhook_url("NFL", "PICKS") is None


def test_get_webhook_url_returns_url_when_set(monkeypatch):
    url = "https://discord.com/api/webhooks/123/abc"
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_PICKS", url)
    assert _get_webhook_url("NFL", "PICKS") == url


def test_get_webhook_url_empty_string_returns_none(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_PICKS", "")
    assert _get_webhook_url("NFL", "PICKS") is None


def test_is_discord_configured_true(monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_NBA_RESULTS", "https://example.com/hook")
    assert is_discord_configured("NBA", "RESULTS") is True


def test_is_discord_configured_false(monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_NBA_RESULTS", raising=False)
    assert is_discord_configured("NBA", "RESULTS") is False


# ---------------------------------------------------------------------------
# Embed builders
# ---------------------------------------------------------------------------


def test_build_pick_embed_structure():
    pick = _make_candidate(edge=0.08, odds=-110)
    embed = _build_pick_embed(pick, rank=1)

    assert embed["title"].startswith("#1")
    assert "BUF @ KC" in embed["description"]
    assert embed["color"] == COLOR_GREEN
    # Compact inline format: all info in description, no fields
    assert "fields" not in embed
    assert "Odds:" in embed["description"]
    assert "Edge:" in embed["description"]
    assert "Bet:" in embed["description"]


def test_build_pick_embed_with_key_factors():
    pick = _make_candidate(edge=0.08, odds=-110)
    analysis = {
        "key_factors": [
            "Elite pass rush vs weak O-line",
            "Home field advantage",
        ]
    }
    embed = _build_pick_embed(pick, rank=1, analysis=analysis)

    assert "Key Factors:" in embed["description"]
    assert "- Elite pass rush vs weak O-line" in embed["description"]
    assert "- Home field advantage" in embed["description"]


def test_build_pick_embed_no_analysis():
    pick = _make_candidate(edge=0.08, odds=-110)
    embed = _build_pick_embed(pick, rank=1, analysis=None)
    assert "Key Factors" not in embed["description"]


def test_build_pick_embed_empty_key_factors():
    pick = _make_candidate(edge=0.08, odds=-110)
    embed = _build_pick_embed(pick, rank=1, analysis={"key_factors": []})
    assert "Key Factors" not in embed["description"]


def test_build_summary_embed():
    candidates = [_make_candidate(recommended_bet=30.0), _make_candidate(recommended_bet=20.0)]
    embed = _build_summary_embed(candidates, bankroll=1000.0, sport="NFL")

    assert "Picks of the Day" in embed["title"]
    assert "NFL" in embed["title"]
    assert embed["color"] == COLOR_BLUE
    # Compact inline format: all info in description, no fields
    assert "fields" not in embed
    assert "$50.00" in embed["description"]
    assert "2 Picks" in embed["description"]
    assert "$1,000.00" in embed["description"]


def test_build_results_embed_positive_pnl():
    summary = {
        "total_bets": 5,
        "wins": 3,
        "losses": 2,
        "pushes": 0,
        "win_rate_pct": 60.0,
        "total_pnl": 45.50,
        "roi_pct": 9.1,
        "avg_edge_pct": 3.2,
    }
    embed = _build_results_embed(summary, "NFL", graded_date=date(2026, 1, 14))

    assert embed["color"] == COLOR_GREEN
    assert "3-2-0" in embed["description"]
    assert "2026-01-14" in embed["description"]


def test_build_results_embed_negative_pnl():
    summary = {
        "total_bets": 4,
        "wins": 1,
        "losses": 3,
        "pushes": 0,
        "win_rate_pct": 25.0,
        "total_pnl": -75.0,
        "roi_pct": -15.0,
        "avg_edge_pct": 2.0,
    }
    embed = _build_results_embed(summary, "NBA")
    assert embed["color"] == COLOR_RED


def test_build_results_embed_no_data():
    summary = {"message": "No graded picks found"}
    embed = _build_results_embed(summary, "NFL")
    assert embed["color"] == COLOR_GREY
    assert "No graded picks found" in embed["description"]


def test_build_results_embed_with_clv():
    summary = {
        "total_bets": 3,
        "wins": 2,
        "losses": 1,
        "pushes": 0,
        "win_rate_pct": 66.7,
        "total_pnl": 20.0,
        "roi_pct": 4.0,
        "avg_edge_pct": 3.0,
        "avg_clv_pct": 1.5,
    }
    embed = _build_results_embed(summary, "NFL")
    assert "Avg CLV" in embed["description"]
    assert "+1.50%" in embed["description"]


def test_build_results_embed_with_pick_details():
    summary = {
        "total_bets": 2,
        "wins": 1,
        "losses": 1,
        "pushes": 0,
        "win_rate_pct": 50.0,
        "total_pnl": -2.73,
        "roi_pct": -4.55,
        "avg_edge_pct": 3.0,
    }
    pick_details = [
        {
            "pick_side": "KC",
            "bet_type": "moneyline",
            "odds": -110,
            "result": "win",
            "pnl": 27.27,
            "home_team": "KC",
            "away_team": "BUF",
        },
        {
            "pick_side": "DEN",
            "bet_type": "spread",
            "odds": -110,
            "result": "loss",
            "pnl": -30.00,
            "home_team": "LV",
            "away_team": "DEN",
        },
    ]
    embed = _build_results_embed(summary, "NFL", pick_details=pick_details)

    assert "Picks:" in embed["description"]
    assert "WIN" in embed["description"]
    assert "LOSS" in embed["description"]
    assert "KC Moneyline (-110)" in embed["description"]
    assert "BUF @ KC" in embed["description"]
    assert "+$27.27" in embed["description"]
    assert "DEN Spread (-110)" in embed["description"]
    assert "DEN @ LV" in embed["description"]
    assert "-$30.00" in embed["description"]


def test_build_results_embed_no_pick_details():
    summary = {
        "total_bets": 2,
        "wins": 1,
        "losses": 1,
        "pushes": 0,
        "win_rate_pct": 50.0,
        "total_pnl": 10.0,
        "roi_pct": 5.0,
        "avg_edge_pct": 3.0,
    }
    embed = _build_results_embed(summary, "NFL", pick_details=None)
    assert "Picks:" not in embed["description"]


def test_build_breakdown_embed():
    breakdown = [
        {"bet_type": "moneyline", "wins": 2, "losses": 1, "win_rate_pct": 66.7, "roi_pct": 8.5},
        {"bet_type": "spread", "wins": 1, "losses": 2, "win_rate_pct": 33.3, "roi_pct": -12.0},
    ]
    embed = _build_breakdown_embed(breakdown, "NFL")
    assert embed["color"] == COLOR_BLUE
    assert "MONEYLINE" in embed["description"]
    assert "SPREAD" in embed["description"]


# ---------------------------------------------------------------------------
# _send_webhook
# ---------------------------------------------------------------------------


@patch("betting_agent.notifications.discord.requests.post")
def test_send_webhook_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 204
    mock_post.return_value = mock_resp

    assert _send_webhook("https://example.com/hook", {"embeds": []}) is True
    mock_post.assert_called_once()


@patch("betting_agent.notifications.discord.requests.post")
def test_send_webhook_retries_on_429(mock_post):
    rate_limit_resp = MagicMock()
    rate_limit_resp.status_code = 429
    rate_limit_resp.json.return_value = {"retry_after": 0.1}

    ok_resp = MagicMock()
    ok_resp.status_code = 204

    mock_post.side_effect = [rate_limit_resp, ok_resp]

    assert _send_webhook("https://example.com/hook", {"embeds": []}) is True
    assert mock_post.call_count == 2


@patch("betting_agent.notifications.discord.requests.post")
def test_send_webhook_returns_false_on_error(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_post.return_value = mock_resp

    assert _send_webhook("https://example.com/hook", {"embeds": []}) is False


@patch("betting_agent.notifications.discord.requests.post")
def test_send_webhook_returns_false_on_exception(mock_post):
    import requests as req
    mock_post.side_effect = req.ConnectionError("connection refused")

    assert _send_webhook("https://example.com/hook", {"embeds": []}) is False


# ---------------------------------------------------------------------------
# send_picks_to_discord
# ---------------------------------------------------------------------------


@patch("betting_agent.notifications.discord._send_webhook")
def test_send_picks_skips_when_unconfigured(mock_send, monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_NFL_PICKS", raising=False)
    result = send_picks_to_discord([_make_candidate()], 1000.0, "NFL")
    assert result is False
    mock_send.assert_not_called()


@patch("betting_agent.notifications.discord._send_webhook", return_value=True)
def test_send_picks_splits_at_max_embeds(mock_send, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_PICKS", "https://example.com/hook")
    # 12 candidates + 1 summary = 13 embeds → should split into 2 messages
    candidates = [_make_candidate(game_id=i, edge=0.05 + i * 0.001) for i in range(12)]

    result = send_picks_to_discord(candidates, 1000.0, "NFL")
    assert result is True
    assert mock_send.call_count == 2

    # First message should have MAX_EMBEDS_PER_MESSAGE embeds
    first_call_embeds = mock_send.call_args_list[0][0][1]["embeds"]
    assert len(first_call_embeds) == MAX_EMBEDS_PER_MESSAGE

    # Second message has the remainder
    second_call_embeds = mock_send.call_args_list[1][0][1]["embeds"]
    assert len(second_call_embeds) == 3  # 13 - 10


@patch("betting_agent.notifications.discord._send_webhook", return_value=True)
def test_send_picks_single_message(mock_send, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_PICKS", "https://example.com/hook")
    candidates = [_make_candidate()]

    result = send_picks_to_discord(candidates, 1000.0, "NFL")
    assert result is True
    assert mock_send.call_count == 1

    # 1 summary + 1 pick = 2 embeds
    embeds = mock_send.call_args_list[0][0][1]["embeds"]
    assert len(embeds) == 2


# ---------------------------------------------------------------------------
# send_results_to_discord
# ---------------------------------------------------------------------------


@patch("betting_agent.notifications.discord._send_webhook", return_value=True)
def test_send_results_with_breakdown(mock_send, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_RESULTS", "https://example.com/hook")
    summary = {
        "total_bets": 5,
        "wins": 3,
        "losses": 2,
        "pushes": 0,
        "win_rate_pct": 60.0,
        "total_pnl": 45.50,
        "roi_pct": 9.1,
        "avg_edge_pct": 3.2,
    }
    breakdown = [
        {"bet_type": "moneyline", "wins": 2, "losses": 1, "win_rate_pct": 66.7, "roi_pct": 8.5},
    ]

    result = send_results_to_discord(summary, "NFL", breakdown, date(2026, 1, 14))
    assert result is True
    assert mock_send.call_count == 1

    # Should have 2 embeds: results + breakdown
    embeds = mock_send.call_args_list[0][0][1]["embeds"]
    assert len(embeds) == 2


@patch("betting_agent.notifications.discord._send_webhook")
def test_send_results_skips_when_unconfigured(mock_send, monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_NFL_RESULTS", raising=False)
    result = send_results_to_discord({"total_bets": 1}, "NFL")
    assert result is False
    mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# All-time results
# ---------------------------------------------------------------------------


def test_build_alltime_sport_embed_positive_pnl():
    summary = {
        "total_bets": 50,
        "wins": 30,
        "losses": 18,
        "pushes": 2,
        "win_rate_pct": 62.5,
        "total_pnl": 245.50,
        "roi_pct": 12.3,
        "avg_edge_pct": 4.1,
    }
    embed = _build_alltime_sport_embed(summary, "NFL", starting_bankroll=1000.0)

    assert embed["title"] == "NFL"
    assert embed["color"] == COLOR_GREEN

    desc = embed["description"]
    assert "30-18-2" in desc
    assert "$1,000.00" in desc
    assert "$1,245.50" in desc
    assert "+$245.50" in desc
    assert "Record" in desc


def test_build_alltime_sport_embed_negative_pnl():
    summary = {
        "total_bets": 20,
        "wins": 7,
        "losses": 13,
        "pushes": 0,
        "win_rate_pct": 35.0,
        "total_pnl": -150.0,
        "roi_pct": -15.0,
        "avg_edge_pct": 2.0,
    }
    embed = _build_alltime_sport_embed(summary, "NBA", starting_bankroll=1000.0)

    assert embed["color"] == COLOR_RED
    assert "$850.00" in embed["description"]


def test_build_alltime_sport_embed_no_data():
    summary = {"message": "No graded picks found"}
    embed = _build_alltime_sport_embed(summary, "NHL", starting_bankroll=500.0)

    assert embed["color"] == COLOR_GREY
    assert "No graded picks yet" in embed["description"]


def test_build_alltime_sport_embed_with_clv():
    summary = {
        "total_bets": 10,
        "wins": 6,
        "losses": 4,
        "pushes": 0,
        "win_rate_pct": 60.0,
        "total_pnl": 80.0,
        "roi_pct": 8.0,
        "avg_edge_pct": 3.5,
        "avg_clv_pct": 1.2,
    }
    embed = _build_alltime_sport_embed(summary, "NFL", starting_bankroll=1000.0)
    assert "Avg CLV" in embed["description"]
    assert "+1.20%" in embed["description"]


@patch("betting_agent.notifications.discord._send_webhook")
def test_send_alltime_skips_when_unconfigured(mock_send, monkeypatch):
    monkeypatch.delenv("DISCORD_WEBHOOK_ALLTIME_RESULTS", raising=False)
    result = send_alltime_to_discord({"NFL": {"total_bets": 10}}, 1000.0)
    assert result is False
    mock_send.assert_not_called()


@patch("betting_agent.notifications.discord._send_webhook", return_value=True)
def test_send_alltime_sends_header_plus_sport_embeds(mock_send, monkeypatch):
    monkeypatch.setenv("DISCORD_WEBHOOK_ALLTIME_RESULTS", "https://example.com/hook")
    summaries = {
        "NFL": {
            "total_bets": 50, "wins": 30, "losses": 18, "pushes": 2,
            "win_rate_pct": 62.5, "total_pnl": 200.0, "roi_pct": 10.0, "avg_edge_pct": 4.0,
        },
        "NBA": {
            "total_bets": 30, "wins": 15, "losses": 14, "pushes": 1,
            "win_rate_pct": 51.7, "total_pnl": -20.0, "roi_pct": -2.0, "avg_edge_pct": 2.5,
        },
    }

    result = send_alltime_to_discord(summaries, 1000.0)
    assert result is True
    assert mock_send.call_count == 1

    # 1 header + 2 sport embeds = 3
    embeds = mock_send.call_args_list[0][0][1]["embeds"]
    assert len(embeds) == 3
    assert embeds[0]["title"] == "All-Time Results"
    assert embeds[0]["color"] == COLOR_BLUE


@patch("betting_agent.notifications.discord._send_webhook")
def test_send_results_skips_no_graded_picks(mock_send, monkeypatch):
    """send_results_to_discord should not post when summary has no actual data."""
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_RESULTS", "https://example.com/hook")
    summary = {"message": "No graded picks found"}
    result = send_results_to_discord(summary, "NFL")
    assert result is True
    mock_send.assert_not_called()


@patch("betting_agent.notifications.discord._send_webhook")
def test_send_alltime_skips_all_empty_summaries(mock_send, monkeypatch):
    """send_alltime_to_discord should not post when all sports have no data."""
    monkeypatch.setenv("DISCORD_WEBHOOK_ALLTIME_RESULTS", "https://example.com/hook")
    summaries = {
        "NFL": {"message": "No graded picks found"},
        "NBA": {"message": "No graded picks found"},
    }
    result = send_alltime_to_discord(summaries, 1000.0)
    assert result is True
    mock_send.assert_not_called()


@patch("betting_agent.notifications.discord._send_webhook", return_value=True)
def test_send_alltime_filters_out_empty_sports(mock_send, monkeypatch):
    """send_alltime_to_discord should only include sports with actual data."""
    monkeypatch.setenv("DISCORD_WEBHOOK_ALLTIME_RESULTS", "https://example.com/hook")
    summaries = {
        "NFL": {"message": "No graded picks found"},
        "NBA": {
            "total_bets": 30, "wins": 15, "losses": 14, "pushes": 1,
            "win_rate_pct": 51.7, "total_pnl": -20.0, "roi_pct": -2.0, "avg_edge_pct": 2.5,
        },
    }
    result = send_alltime_to_discord(summaries, 1000.0)
    assert result is True
    # 1 header + 1 sport (NBA only, NFL filtered out) = 2
    embeds = mock_send.call_args_list[0][0][1]["embeds"]
    assert len(embeds) == 2
    assert embeds[1]["title"] == "NBA"
