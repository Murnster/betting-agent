from __future__ import annotations

from argparse import Namespace
from datetime import date, datetime, timedelta, timezone

import pandas as pd

import scripts.picks as picks_script
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.validator.orchestrator import (
    AgentValidationRecord,
    AgentValidationSummary,
)


def test_main_saves_picks_before_agent_validations(monkeypatch, capsys):
    call_order = []
    candidate = BetCandidate(
        game_id=0,
        external_id="game-1",
        home_team="TeamA",
        away_team="TeamB",
        game_date=date(2026, 1, 15),
        sport="NFL",
        bet_type="moneyline",
        pick_side="TeamA",
        model_prob=0.58,
        implied_prob=0.50,
        edge=0.08,
        odds=-110,
        kelly_fraction=0.04,
        recommended_bet=40.0,
        bankroll_at_pick=1000.0,
    )
    validation_summary = AgentValidationSummary(
        validated_games=1,
        records=[
            AgentValidationRecord(
                game_id=None,
                external_id="game-1",
                pick_date=date(2026, 1, 15),
                sport="NFL",
                bet_type="moneyline",
                pick_side="TeamA",
                verdict="UNCHANGED",
                original_edge=0.08,
                adjusted_edge=0.08,
                reasons=["no material issues"],
                input_tokens=100,
                output_tokens=20,
                cost_usd=0.01,
            )
        ],
    )

    class _Config:
        sport_key = "americanfootball_nfl"

        @staticmethod
        def build_features(df: pd.DataFrame) -> pd.DataFrame:
            return df

    class _Engine:
        training_seasons: list[int] = []

        def load(self) -> bool:
            return True

        def get_sigma(self) -> dict[str, float]:
            return {"total_sigma": 14.0, "margin_sigma": 14.0}

    class _Client:
        def fetch_odds(self, sport_key: str) -> list[dict]:
            commence_time = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
            return [
                {
                    "id": "game-1",
                    "home_team": "TeamA",
                    "away_team": "TeamB",
                    "commence_time": commence_time,
                }
            ]

    monkeypatch.setattr(
        picks_script.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(
            bankroll=100.0,
            sport="NFL",
            save=True,
            agent_mode="top",
            agent_max_games=1,
        ),
    )
    monkeypatch.setattr(picks_script, "get_sport_config", lambda sport: _Config())
    monkeypatch.setattr(picks_script, "PredictionEngine", lambda sport: _Engine())
    monkeypatch.setattr(picks_script, "OddsAPIClient", lambda: _Client())
    monkeypatch.setattr(picks_script, "_load_recent_history", lambda config, seasons: pd.DataFrame())
    monkeypatch.setattr(picks_script, "generate_picks", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.validate_picks",
        lambda *args, **kwargs: ([candidate], validation_summary),
    )
    monkeypatch.setattr(picks_script, "format_picks_cli", lambda *args, **kwargs: "formatted")
    monkeypatch.setattr(picks_script, "save_picks_to_db", lambda candidates: call_order.append("picks"))
    monkeypatch.setattr(
        "betting_agent.intelligence.validator.save_agent_validations_to_db",
        lambda records: call_order.append("validations"),
    )
    monkeypatch.setattr(picks_script.settings, "discord_enabled", False)

    picks_script.main()

    assert call_order == ["picks", "validations"]
    assert "Saved 1 picks to DB." in capsys.readouterr().out
