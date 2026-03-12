#!/usr/bin/env python
"""
Entry point: print today's +EV Picks of the Day to the terminal.

Usage:
    uv run python scripts/picks.py [--bankroll 100] [--sport NFL]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from betting_agent.config import settings
from betting_agent.intelligence.picks import format_picks_cli, generate_picks, save_picks_to_db
from betting_agent.models.engine import PredictionEngine
from betting_agent.sports.registry import get_sport_config, available_sports
from betting_agent.api.odds import OddsAPIClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_recent_history(config, history_seasons: list[int], n_games: int = 20) -> pd.DataFrame:
    """Load recent historical games so Elo/rolling avgs have warm-up data."""
    loader = config.loader_cls()
    df = loader.load_schedules(history_seasons).to_pandas()
    # Sort by game_date and take recent games for warm-up
    date_col = "game_date" if "game_date" in df.columns else "gameday"
    df = df.sort_values(date_col).tail(n_games * 32)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate today's POTD")
    parser.add_argument("--bankroll", type=float, default=settings.starting_bankroll)
    parser.add_argument(
        "--sport",
        type=str,
        default="NFL",
        help=f"Sport ({', '.join(available_sports())})",
    )
    parser.add_argument("--save", action="store_true", help="Save picks to DB")
    parser.add_argument(
        "--agent-mode",
        type=str,
        default=settings.agent_mode if settings.agent_enabled else "off",
        choices=("off", "top", "all"),
        help="Validator mode: off, top, or all",
    )
    parser.add_argument(
        "--agent-max-games",
        type=int,
        default=settings.agent_max_games_per_run,
        help="Maximum games to validate when agent mode is enabled",
    )
    args = parser.parse_args()

    sport = args.sport.upper()
    config = get_sport_config(sport)
    save_dir = Path(settings.saved_models_dir) / sport

    # ---- Load models ----
    engine = PredictionEngine(sport=sport)
    if not engine.load():
        print(f"\nModels not found in {save_dir}. Run:\n  uv run python scripts/train.py --sport {sport}\n")
        return

    # ---- Load upcoming games from Odds API ----
    client = OddsAPIClient()
    sport_key = config.sport_key
    logger.info("Fetching upcoming games from Odds API (%s)...", sport_key)
    raw_games = client.fetch_odds(sport_key)

    if not raw_games:
        print("\nNo upcoming games found from Odds API (check ODDS_API_KEY).\n")
        return

    # Filter out games that have already started (with 5 min buffer)
    # AND games that start more than 24 hours from now
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    max_lookahead = now_utc + timedelta(hours=24)
    filtered_games = []
    for g in raw_games:
        try:
            commence_time = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
            # If game starts more than 5 minutes from now AND within next 24 hours
            if (commence_time - now_utc).total_seconds() > -300 and commence_time <= max_lookahead:
                filtered_games.append(g)
        except (KeyError, ValueError):
            filtered_games.append(g)
    
    if not filtered_games:
        print("\nAll today's games have already started.\n")
        return
    
    raw_games = filtered_games
    logger.info("Filtered to %d upcoming games.", len(raw_games))

    # ---- Build feature rows for upcoming games ----
    current_year = date.today().year
    history_seasons = [current_year - 1, current_year]
    try:
        hist_df = _load_recent_history(config, history_seasons)
    except Exception as exc:
        logger.warning("Could not load history: %s", exc)
        hist_df = pd.DataFrame()

    # For NBA/NHL, we need to map Odds API full names → abbreviations
    team_name_map = None
    if sport == "NBA":
        from betting_agent.sports.nba.loader import NBA_FULL_TO_ABBREV
        team_name_map = NBA_FULL_TO_ABBREV
    elif sport == "NHL":
        from betting_agent.sports.nhl.loader import NHL_FULL_TO_ABBREV
        team_name_map = NHL_FULL_TO_ABBREV
    elif sport == "MLB":
        from betting_agent.sports.mlb.loader import MLB_FULL_TO_ABBREV
        team_name_map = MLB_FULL_TO_ABBREV

    # Build upcoming game rows (minimal info from Odds API)
    upcoming_rows = []
    for g in raw_games:
        from datetime import datetime
        dt_str = g.get("commence_time", "")
        gd = None
        try:
            gd = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).date()
        except Exception:
            pass

        home_raw = g.get("home_team", "")
        away_raw = g.get("away_team", "")

        # Map team names if needed (NBA: full name → abbreviation)
        home = team_name_map.get(home_raw, home_raw) if team_name_map else home_raw
        away = team_name_map.get(away_raw, away_raw) if team_name_map else away_raw

        upcoming_rows.append({
            "game_id": g.get("id", ""),
            "game_date": str(gd) if gd else str(date.today()),
            "season": current_year,
            "week": None,
            "home_team": home,
            "away_team": away,
            # Keep original Odds API names for odds matching
            "home_team_odds": home_raw,
            "away_team_odds": away_raw,
            "home_score": None,
            "away_score": None,
            "is_playoff": False,
            "neutral_site": False,
            "weather_desc": None,
            "temperature_f": None,
            "wind_mph": None,
            "external_id": g.get("id"),
        })

    upcoming_df = pd.DataFrame(upcoming_rows)
    if upcoming_df.empty:
        print("\nNo upcoming games parsed.\n")
        return

    # Combine history + upcoming for feature computation.
    # Tag upcoming rows so we can retrieve them after sort inside build_features.
    upcoming_df["_is_upcoming"] = True
    # Preserve original row order — build_features() sorts by game_date which can
    # reorder rows within the same date, breaking the metadata ↔ features alignment.
    upcoming_df["_upcoming_idx"] = range(len(upcoming_df))

    if not hist_df.empty:
        # Normalise history columns to canonical names for concat
        hist_trim = hist_df.copy()
        # Handle NFL-specific column renames
        if "gameday" in hist_trim.columns and "game_date" not in hist_trim.columns:
            hist_trim = hist_trim.rename(columns={"gameday": "game_date"})
        if "game_id" in hist_trim.columns and "external_id" not in hist_trim.columns:
            hist_trim = hist_trim.rename(columns={"game_id": "external_id"})
            hist_trim["game_id"] = hist_trim["external_id"]

        # Keep only columns the upcoming_df also has
        keep_cols = list(upcoming_df.columns)
        # Add any extra columns that exist in hist for normalisation
        extra = ["game_type", "location"]
        for c in extra:
            if c in hist_trim.columns:
                keep_cols.append(c)
        hist_trim = hist_trim[[c for c in keep_cols if c in hist_trim.columns]].copy()
        hist_trim["_is_upcoming"] = False
        combined = pd.concat([hist_trim, upcoming_df], ignore_index=True)
    else:
        combined = upcoming_df

    features_all = config.build_features(combined)
    # Retrieve upcoming rows by the tag column, then restore original order
    if "_is_upcoming" in features_all.columns:
        upcoming_mask = features_all["_is_upcoming"].astype(bool)
        upcoming_features = features_all[upcoming_mask].copy()
        if "_upcoming_idx" in upcoming_features.columns:
            upcoming_features = upcoming_features.sort_values("_upcoming_idx")
        features = upcoming_features.drop(columns=["_is_upcoming", "_upcoming_idx"], errors="ignore").reset_index(drop=True)
    else:
        features = features_all.tail(len(upcoming_df)).reset_index(drop=True)
    metadata = upcoming_df.drop(columns=["_is_upcoming", "_upcoming_idx"], errors="ignore").reset_index(drop=True)

    use_agent = args.agent_mode != "off"

    # ---- Sentiment + Game Analysis (optional legacy path) ----
    sentiment_scores: dict[str, float] | None = None
    game_analyses: dict[str, dict] = {}
    if not use_agent:
        try:
            from betting_agent.intelligence.sentiment import (
                analyze_game,
                fetch_team_context,
                is_ollama_available,
                store_sentiment,
            )
            import json

            if is_ollama_available():
                logger.info("Ollama available — running game analysis...")
                sentiment_scores = {}
                depth = settings.sentiment_depth
                for _, row in metadata.iterrows():
                    home_t = str(row.get("home_team", ""))
                    away_t = str(row.get("away_team", ""))
                    if not home_t or not away_t or home_t in sentiment_scores:
                        continue

                    home_ctx = fetch_team_context(home_t, current_year, depth=depth, sport=sport)
                    away_ctx = fetch_team_context(away_t, current_year, depth=depth, sport=sport)
                    result = analyze_game(home_t, away_t, home_ctx, away_ctx, sport=sport)

                    sentiment_scores[home_t] = result["home_score"]
                    sentiment_scores[away_t] = result["away_score"]
                    game_analyses[f"{home_t}|{away_t}"] = result
                    logger.info(
                        "  %s vs %s: home=%.2f away=%.2f edge=%s",
                        home_t, away_t,
                        result["home_score"], result["away_score"],
                        result["matchup_edge"],
                    )

                    game_id = row.get("game_id")
                    if game_id and args.save:
                        try:
                            summary_json = json.dumps(result)
                            store_sentiment(
                                game_id=int(game_id) if str(game_id).isdigit() else 0,
                                team=f"{home_t} vs {away_t}",
                                score=(result["home_score"] + result["away_score"]) / 2,
                                summary=summary_json,
                                analysis_type="game",
                            )
                        except Exception as exc:
                            logger.debug("Could not store game analysis: %s", exc)
            else:
                logger.info("Ollama not available — skipping sentiment")
        except Exception as exc:
            logger.warning("Sentiment failed, continuing without: %s", exc)

    # ---- Generate picks ----
    sigma = engine.get_sigma()
    logger.info(
        "Using sigma — total: %.2f, margin: %.2f",
        sigma["total_sigma"], sigma["margin_sigma"],
    )
    candidates = generate_picks(
        features=features,
        metadata=metadata,
        odds_data=raw_games,
        engine=engine,
        bankroll=args.bankroll,
        sport=sport,
        pick_date=date.today(),
        sentiment_scores=sentiment_scores,
        total_sigma=sigma["total_sigma"],
        margin_sigma=sigma["margin_sigma"],
        max_picks=settings.max_picks_per_run,
    )

    agent_summary: dict | None = None
    validation_records = []

    if use_agent and candidates:
        try:
            from betting_agent.intelligence.validator import validate_picks

            candidates, validation = validate_picks(
                candidates,
                sport=sport,
                history_df=hist_df,
                metadata_df=metadata,
                mode=args.agent_mode,
                max_games=args.agent_max_games,
            )
            validation_records = validation.records
            agent_summary = {
                "validated_games": validation.validated_games,
                "skipped_games": validation.skipped_games,
                "total_cost_usd": validation.total_cost_usd,
            }
        except Exception as exc:
            logger.warning("Validator failed, continuing without adjustments: %s", exc)
    else:
        for c in candidates:
            key = f"{c.home_team}|{c.away_team}"
            if key in game_analyses:
                c.extra["analysis"] = game_analyses[key]

    # ---- Output ----
    print(format_picks_cli(candidates, args.bankroll, sport=sport, agent_summary=agent_summary))

    if args.save and candidates:
        try:
            save_picks_to_db(candidates)
            print(f"Saved {len(candidates)} picks to DB.\n")
        except Exception as exc:
            logger.warning("Could not save picks to DB: %s", exc)

    if args.save and validation_records:
        try:
            from betting_agent.intelligence.validator import save_agent_validations_to_db

            save_agent_validations_to_db(validation_records)
        except Exception as exc:
            logger.warning("Could not save agent validations to DB: %s", exc)

    # ---- Discord notification ----
    if settings.discord_enabled:
        try:
            from betting_agent.notifications.discord import send_picks_to_discord, is_discord_configured
            if is_discord_configured(sport, "PICKS"):
                logger.info("Sending picks to Discord (%s)...", sport)
                send_picks_to_discord(candidates, args.bankroll, sport, agent_summary=agent_summary)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
