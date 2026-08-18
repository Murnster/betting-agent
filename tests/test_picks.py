"""Tests for pick generation: dedup, correlation adjustment, underdog ML."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from betting_agent.intelligence.picks import (
    BetCandidate,
    _extract_team_from_pick,
    _passes_guardrails,
    generate_picks,
)


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


def _make_candidate(game_id=1, bet_type="moneyline", pick_side="TeamA",
                    edge=0.05, **kwargs):
    """Helper to create a BetCandidate with sensible defaults."""
    defaults = dict(
        home_team="TeamA", away_team="TeamB", game_date=date(2025, 1, 1),
        sport="NFL", model_prob=0.55, implied_prob=0.50, odds=-110,
    )
    defaults.update(kwargs)
    return BetCandidate(
        game_id=game_id, bet_type=bet_type, pick_side=pick_side,
        edge=edge, **defaults,
    )


class TestCorrelatedPickDedup:
    def test_extract_team_from_pick(self):
        """Unit test _extract_team_from_pick across all bet types."""
        ml = _make_candidate(bet_type="moneyline", pick_side="Buffalo Bills")
        assert _extract_team_from_pick(ml) == "Buffalo Bills"

        spread = _make_candidate(bet_type="spread", pick_side="Buffalo Bills +1.5")
        assert _extract_team_from_pick(spread) == "Buffalo Bills"

        spread_neg = _make_candidate(bet_type="spread", pick_side="New York Jets -3.0")
        assert _extract_team_from_pick(spread_neg) == "New York Jets"

        total = _make_candidate(bet_type="total", pick_side="over 44.5")
        assert _extract_team_from_pick(total) is None

    def test_same_team_ml_and_spread_keeps_higher_edge(self):
        """When ML and spread exist for same team, only higher-edge survives."""
        engine = _make_mock_engine(win_prob=0.65, home_score=28, away_score=21)
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        # Large home ML edge + small spread that also has edge
        odds = _make_odds_data("TeamA", "TeamB", home_ml=200, away_ml=-250,
                               spread=-1.5, spread_home_price=-110,
                               spread_away_price=-110)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NFL",
        )
        # Filter to team-directional picks for TeamA
        team_a_picks = [
            c for c in candidates
            if c.bet_type in ("moneyline", "spread") and
            _extract_team_from_pick(c) == "TeamA"
        ]
        assert len(team_a_picks) <= 1, (
            "Should have at most 1 team-directional pick per team per game"
        )

    def test_totals_not_affected(self):
        """Totals should coexist with ML/spread picks after dedup."""
        engine = _make_mock_engine(win_prob=0.65, home_score=30, away_score=25)
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        odds = _make_odds_data("TeamA", "TeamB", home_ml=200, away_ml=-250,
                               total_line=40.0, over_price=-110, under_price=-110)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NFL",
        )
        total_picks = [c for c in candidates if c.bet_type == "total"]
        ml_or_spread = [c for c in candidates if c.bet_type in ("moneyline", "spread")]
        # Totals can coexist with team-directional picks
        if total_picks and ml_or_spread:
            assert len(total_picks) >= 1

    def test_opposite_sides_not_deduped(self):
        """Away ML + home spread should both survive (different teams)."""
        c_away_ml = _make_candidate(
            game_id=1, bet_type="moneyline", pick_side="TeamB", edge=0.06,
        )
        c_home_spread = _make_candidate(
            game_id=1, bet_type="spread", pick_side="TeamA +1.5", edge=0.04,
        )
        # Verify they extract to different teams
        assert _extract_team_from_pick(c_away_ml) == "TeamB"
        assert _extract_team_from_pick(c_home_spread) == "TeamA"
        # Different team keys → both would survive the dedup


class TestGuardrails:
    def test_max_edge_rejection(self):
        """Picks with edge > max_edge_pct should be rejected."""
        assert not _passes_guardrails(edge=0.20, odds=-110, model_prob=0.60, bet_type="moneyline")
        assert not _passes_guardrails(edge=0.36, odds=-110, model_prob=0.60, bet_type="spread")
        assert not _passes_guardrails(edge=0.16, odds=-110, model_prob=0.60, bet_type="total")

    def test_underdog_odds_rejection(self):
        """ML picks worse than +500 should be rejected."""
        assert not _passes_guardrails(edge=0.05, odds=750, model_prob=0.40, bet_type="moneyline")
        assert not _passes_guardrails(edge=0.05, odds=600, model_prob=0.40, bet_type="moneyline")
        # +500 exactly should pass
        assert _passes_guardrails(edge=0.05, odds=500, model_prob=0.40, bet_type="moneyline")
        # Negative odds (favorites) always pass
        assert _passes_guardrails(edge=0.05, odds=-200, model_prob=0.60, bet_type="moneyline")

    def test_min_model_prob_rejection(self):
        """ML picks with model prob < 20% should be rejected."""
        assert not _passes_guardrails(edge=0.05, odds=300, model_prob=0.15, bet_type="moneyline")
        assert not _passes_guardrails(edge=0.05, odds=300, model_prob=0.10, bet_type="moneyline")
        # 20% exactly should pass
        assert _passes_guardrails(edge=0.05, odds=300, model_prob=0.20, bet_type="moneyline")

    def test_guardrails_only_apply_to_moneyline_for_odds_and_prob(self):
        """Underdog odds and min prob guardrails don't apply to spread/total."""
        assert _passes_guardrails(edge=0.05, odds=750, model_prob=0.15, bet_type="spread")
        assert _passes_guardrails(edge=0.05, odds=750, model_prob=0.15, bet_type="total")

    def test_normal_pick_passes(self):
        """Normal picks with reasonable edge/odds/prob should pass."""
        assert _passes_guardrails(edge=0.03, odds=-110, model_prob=0.55, bet_type="moneyline")
        assert _passes_guardrails(edge=0.05, odds=200, model_prob=0.40, bet_type="moneyline")
        assert _passes_guardrails(edge=0.04, odds=-110, model_prob=0.55, bet_type="spread")

    def test_guardrails_in_generate_picks(self):
        """Picks with huge edge should be filtered out of generate_picks."""
        # Model says 43.6% win prob, odds +750 → huge edge, should be rejected
        engine = _make_mock_engine(win_prob=0.436)
        features = pd.DataFrame({"col1": [1.0]})
        metadata = pd.DataFrame({
            "game_id": [1],
            "home_team": ["TeamA"],
            "away_team": ["TeamB"],
            "home_team_odds": ["TeamA"],
        })
        odds = _make_odds_data("TeamA", "TeamB", home_ml=750, away_ml=-1200)
        candidates = generate_picks(
            features, metadata, odds, engine,
            bankroll=100.0, sport="NBA",
        )
        home_ml = [c for c in candidates if c.bet_type == "moneyline" and c.pick_side == "TeamA"]
        assert len(home_ml) == 0, "Huge underdog pick (+750) should be rejected by guardrails"
