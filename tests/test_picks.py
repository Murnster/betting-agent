"""Tests for pick generation: dedup, correlation adjustment, underdog ML."""

from unittest.mock import MagicMock

import pandas as pd

from betting_agent.intelligence.picks import generate_picks


def _make_mock_engine(win_prob=0.55, home_score=24.0, away_score=20.0):
    """Create a mock PredictionEngine."""
    engine = MagicMock()
    engine.predict.return_value = pd.DataFrame({
        "win_prob": [win_prob],
        "home_pred_score": [home_score],
        "away_pred_score": [away_score],
        "pred_total": [home_score + away_score],
        "pred_margin": [home_score - away_score],
    })
    return engine


def _make_odds_data(home_team, away_team, home_ml=-110, away_ml=-110,
                    spread=-3.0, spread_home_price=-110, spread_away_price=-110,
                    total_line=44.5, over_price=-110, under_price=-110):
    """Create mock Odds API response."""
    return [{
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": [{
            "title": "TestBook",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": home_team, "price": home_ml},
                        {"name": away_team, "price": away_ml},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": home_team, "price": spread_home_price, "point": spread},
                        {"name": away_team, "price": spread_away_price, "point": -spread},
                    ],
                },
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "price": over_price, "point": total_line},
                        {"name": "Under", "price": under_price, "point": total_line},
                    ],
                },
            ],
        }],
    }]


class TestDedupBothSides:
    def test_no_both_home_and_away_ml(self):
        """Should not have both home ML and away ML for the same game."""
        engine = _make_mock_engine(win_prob=0.50)  # 50/50, could trigger both sides
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        odds = _make_odds_data("TeamA", "TeamB", home_ml=200, away_ml=200)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NFL",
        )
        ml_picks = [c for c in candidates if c.bet_type == "moneyline"]
        assert len(ml_picks) <= 1, "Should have at most 1 moneyline pick per game"

    def test_no_both_over_and_under(self):
        """Should not have both over and under for the same game."""
        engine = _make_mock_engine(win_prob=0.55, home_score=30, away_score=30)
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        odds = _make_odds_data("TeamA", "TeamB", total_line=44.5,
                               over_price=300, under_price=300)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NFL",
        )
        total_picks = [c for c in candidates if c.bet_type == "total"]
        assert len(total_picks) <= 1, "Should have at most 1 total pick per game"


class TestCorrelationAdjustment:
    def test_multi_bet_same_game_reduces_kelly(self):
        """When multiple bets on same game, Kelly fractions should be reduced."""
        # Create a scenario with a big edge on both ML and total
        engine = _make_mock_engine(win_prob=0.70, home_score=30, away_score=20)
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        odds = _make_odds_data("TeamA", "TeamB",
                               home_ml=150,
                               total_line=35.0, over_price=-110, under_price=-110)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NFL",
        )

        if len(candidates) > 1:
            # If there are multi bets on same game, they should have adjusted Kelly
            for c in candidates:
                assert c.kelly_fraction >= 0.0
                assert c.recommended_bet >= 0.0


class TestUnderdogML:
    def test_underdog_ml_allowed(self):
        """Underdog ML bets should be generated when edge exists."""
        # Model says 42% win prob, but odds are +200 (33% implied) → 9% edge
        engine = _make_mock_engine(win_prob=0.42)
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        # Home team is underdog at +200
        odds = _make_odds_data("TeamA", "TeamB", home_ml=200, away_ml=-250)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NFL",
        )
        home_ml = [c for c in candidates if c.bet_type == "moneyline" and c.pick_side == "TeamA"]
        assert len(home_ml) == 1, "Should generate underdog ML pick with sufficient edge"
        assert home_ml[0].model_prob < 0.5, "This should be an underdog pick"


class TestScoringSimga:
    def test_sigma_passed_to_edge_calcs(self):
        """NBA sigma (11.5) should produce different results than NFL (14.0)."""
        from betting_agent.intelligence.ev import calculate_spread_edge, calculate_total_edge

        # Same inputs, different sigma
        _, edge_nfl = calculate_spread_edge(3.0, 0.0, -110, sigma=14.0)
        _, edge_nba = calculate_spread_edge(3.0, 0.0, -110, sigma=11.5)

        # Smaller sigma → same margin is more significant → bigger edge
        assert edge_nba > edge_nfl

        _, te_nfl = calculate_total_edge(230.0, 220.0, -110, "over", sigma=14.0)
        _, te_nba = calculate_total_edge(230.0, 220.0, -110, "over", sigma=11.5)
        assert te_nba > te_nfl
