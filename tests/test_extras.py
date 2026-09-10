"""
The extras channel: props that cleared the floors but missed the card.

They are the same model, markets and book as the main card — only the slate
cap separated them — so they stay on_card=False rather than becoming a
Pick.strategy, and they are reported in their own channel against their own
paper bankroll. The channel is the live A/B for "should the card be wider".
"""

from __future__ import annotations

from datetime import date

import pytest

from betting_agent.intelligence.picks import BetCandidate
from betting_agent.notifications import discord as d


def _cand(player: str, edge: float, bet: float = 1.89, market: str = "player_receptions"):
    return BetCandidate(
        game_id=0, external_id="e1", home_team="Seattle Seahawks",
        away_team="New England Patriots", game_date=date(2026, 9, 9),
        scheduled_game_date=date(2026, 9, 9), sport="NFL", bet_type="prop",
        pick_side="under", player=player, market=market, line=3.5,
        model_prob=0.7, implied_prob=0.7 - edge, edge=edge, odds=-115,
        kelly_fraction=0.019, recommended_bet=bet, bankroll_at_pick=100.0,
        extra={"bookmaker": "draftkings"},
    )


@pytest.fixture
def sent(monkeypatch):
    posts: list[tuple[str, dict]] = []
    monkeypatch.setattr(d, "_send_webhook", lambda url, payload: posts.append((url, payload)) or True)
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_EXTRAS", "https://discord.test/extras")
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_EXTRAS_RESULTS", "https://discord.test/extras-results")
    return posts


class TestExtrasCard:
    def test_posts_every_off_card_pick_ranked_by_edge(self, sent):
        assert d.send_extras_to_discord(
            "Wednesday — NE @ SEA", [_cand("Low", 0.05), _cand("High", 0.30)], 100.0)
        assert all(url.endswith("/extras") for url, _ in sent)
        body = "\n".join(e.get("description", "") for _, p in sent for e in p["embeds"])
        assert "2 off-card" in body
        assert body.index("High") < body.index("Low")
        assert "never counted in the main record" in body

    def test_long_boards_are_chunked_under_the_embed_limit(self, sent):
        cands = [_cand(f"P{i}", 0.10 + i / 1000) for i in range(45)]
        assert d.send_extras_to_discord("Sunday Early — 8 games", cands, 100.0)
        embeds = [e for _, p in sent for e in p["embeds"]]
        assert all(len(e.get("description", "")) < 4096 for e in embeds)
        body = "\n".join(e.get("description", "") for e in embeds)
        assert all(f"P{i}" in body for i in range(45))

    def test_silent_without_a_webhook(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_NFL_EXTRAS", raising=False)
        assert d.send_extras_to_discord("x", [_cand("A", 0.1)], 100.0) is False

    def test_nothing_is_posted_when_the_card_took_everything(self, sent):
        assert d.send_extras_to_discord("x", [], 100.0) is True
        assert sent == []


class TestExtrasResults:
    def test_posts_to_its_own_results_channel(self, sent, monkeypatch):
        """Picks and results split the same way for the extras as for the card."""
        monkeypatch.delenv("DISCORD_WEBHOOK_NFL_EXTRAS_RESULTS", raising=False)
        summary = {"total_bets": 1, "wins": 1, "losses": 0, "pushes": 0, "win_rate_pct": 100.0,
                   "total_pnl": 1.0, "roi_pct": 10.0, "avg_edge_pct": 5.0}
        # The extras PICKS webhook must not be used as a fallback.
        assert d.send_extras_results_to_discord(summary, "NFL") is False
        assert sent == []

    def test_reports_its_own_bankroll_not_the_card_s(self, sent):
        summary = {"total_bets": 9, "wins": 6, "losses": 3, "pushes": 0,
                   "win_rate_pct": 66.7, "total_pnl": 4.49, "roi_pct": 26.38,
                   "avg_edge_pct": 21.41}
        detail = [{"result": "win", "pnl": 1.54, "odds": -123, "bet_type": "prop",
                   "player": "A.J. Brown", "market": "player_receptions", "line": 5.5,
                   "pick_side": "under", "strategy": None,
                   "home_team": "SEA", "away_team": "NE"}]
        assert d.send_extras_results_to_discord(
            summary, "NFL", date(2026, 9, 10), pick_details=detail,
            alltime_summary=summary, starting_bankroll=250.0)
        assert all(url.endswith("extras-results") for url, _ in sent)
        body = "\n".join(e.get("description", "") for _, p in sent for e in p["embeds"])
        assert "6-3-0" in body and "A.J. Brown" in body
        assert "$250.00 → $254.49" in body

    def test_a_day_with_no_extras_posts_nothing(self, sent):
        assert d.send_extras_results_to_discord({"message": "No graded picks found"}, "NFL") is True
        assert sent == []
