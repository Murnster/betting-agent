from __future__ import annotations

import pandas as pd

from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.validator.schemas import DeterministicFlag


def build_deterministic_flags(
    candidates: list[BetCandidate],
    sport: str,
    history_df: pd.DataFrame | None = None,
    metadata_map: dict[str, dict] | None = None,
) -> list[DeterministicFlag]:
    flags: list[DeterministicFlag] = []
    if not candidates:
        return flags

    first = candidates[0]
    if sport in {"NBA", "NHL"}:
        flags.extend(_back_to_back_flags(first, history_df))

    if sport == "NFL":
        flags.extend(_weather_flags(first, metadata_map or {}))

    return flags


def _back_to_back_flags(candidate: BetCandidate, history_df: pd.DataFrame | None) -> list[DeterministicFlag]:
    if history_df is None or history_df.empty:
        return []

    flags: list[DeterministicFlag] = []
    target_date = pd.to_datetime(candidate.event_date).date()
    game_date_col = "game_date" if "game_date" in history_df.columns else "gameday"
    if game_date_col not in history_df.columns:
        return []

    hist = history_df.copy()
    hist[game_date_col] = pd.to_datetime(hist[game_date_col], errors="coerce").dt.date
    previous_day = target_date.fromordinal(target_date.toordinal() - 1)

    for team, side in ((candidate.home_team, "home"), (candidate.away_team, "away")):
        team_games = hist[
            (hist["home_team"] == team) | (hist["away_team"] == team)
        ]
        played_previous_day = not team_games[team_games[game_date_col] == previous_day].empty
        if played_previous_day:
            flags.append(
                DeterministicFlag(
                    type="back_to_back",
                    team=side,
                    severity="medium",
                    detail=f"{team} played on {previous_day.isoformat()}",
                )
            )

    return flags


def _weather_flags(candidate: BetCandidate, metadata_map: dict[str, dict]) -> list[DeterministicFlag]:
    key = candidate.external_id or str(candidate.game_id)
    metadata = metadata_map.get(key) or {}
    wind = metadata.get("wind_mph")
    weather = str(metadata.get("weather_desc") or "").lower()
    if wind is None and not weather:
        return []

    try:
        wind_val = float(wind) if wind is not None else 0.0
    except (TypeError, ValueError):
        wind_val = 0.0

    severe = wind_val >= 25 or any(token in weather for token in ("snow", "storm", "heavy rain"))
    if not severe:
        return []

    return [
        DeterministicFlag(
            type="weather",
            severity="high",
            detail=f"Existing weather data indicates elevated totals risk (wind={wind_val:.1f}, {weather})",
        )
    ]
