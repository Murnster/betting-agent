#!/usr/bin/env python
"""
Daily extraction orchestrator.

Subcommands:
    morning   — Fetch schedule + opening odds + weather
    closing   — Capture closing odds (run ~5 min before game time)
    postgame  — Fetch final scores for completed games

Usage:
    uv run python scripts/extract.py morning [--sport NFL]
    uv run python scripts/extract.py closing [--sport NBA]
    uv run python scripts/extract.py postgame [--sport NFL]
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

from betting_agent.api.odds import OddsAPIClient
from betting_agent.api.weather import WeatherClient
from betting_agent.db.queries import get_game_by_external_id, get_scheduled_games, upsert_game
from betting_agent.db.session import get_session
from betting_agent.sports.registry import get_sport_config, available_sports

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Stadium cities for weather lookups (team full name → city)
NFL_TEAM_CITIES: dict[str, str] = {
    "Arizona Cardinals": "Glendale, AZ",
    "Atlanta Falcons": "Atlanta, GA",
    "Baltimore Ravens": "Baltimore, MD",
    "Buffalo Bills": "Orchard Park, NY",
    "Carolina Panthers": "Charlotte, NC",
    "Chicago Bears": "Chicago, IL",
    "Cincinnati Bengals": "Cincinnati, OH",
    "Cleveland Browns": "Cleveland, OH",
    "Dallas Cowboys": "Arlington, TX",
    "Denver Broncos": "Denver, CO",
    "Detroit Lions": "Detroit, MI",
    "Green Bay Packers": "Green Bay, WI",
    "Houston Texans": "Houston, TX",
    "Indianapolis Colts": "Indianapolis, IN",
    "Jacksonville Jaguars": "Jacksonville, FL",
    "Kansas City Chiefs": "Kansas City, MO",
    "Las Vegas Raiders": "Las Vegas, NV",
    "Los Angeles Chargers": "Inglewood, CA",
    "Los Angeles Rams": "Inglewood, CA",
    "Miami Dolphins": "Miami Gardens, FL",
    "Minnesota Vikings": "Minneapolis, MN",
    "New England Patriots": "Foxborough, MA",
    "New Orleans Saints": "New Orleans, LA",
    "New York Giants": "East Rutherford, NJ",
    "New York Jets": "East Rutherford, NJ",
    "Philadelphia Eagles": "Philadelphia, PA",
    "Pittsburgh Steelers": "Pittsburgh, PA",
    "San Francisco 49ers": "Santa Clara, CA",
    "Seattle Seahawks": "Seattle, WA",
    "Tampa Bay Buccaneers": "Tampa, FL",
    "Tennessee Titans": "Nashville, TN",
    "Washington Commanders": "Landover, MD",
}

NBA_TEAM_CITIES: dict[str, str] = {
    "Atlanta Hawks": "Atlanta, GA",
    "Boston Celtics": "Boston, MA",
    "Brooklyn Nets": "Brooklyn, NY",
    "Charlotte Hornets": "Charlotte, NC",
    "Chicago Bulls": "Chicago, IL",
    "Cleveland Cavaliers": "Cleveland, OH",
    "Dallas Mavericks": "Dallas, TX",
    "Denver Nuggets": "Denver, CO",
    "Detroit Pistons": "Detroit, MI",
    "Golden State Warriors": "San Francisco, CA",
    "Houston Rockets": "Houston, TX",
    "Indiana Pacers": "Indianapolis, IN",
    "Los Angeles Clippers": "Inglewood, CA",
    "Los Angeles Lakers": "Los Angeles, CA",
    "Memphis Grizzlies": "Memphis, TN",
    "Miami Heat": "Miami, FL",
    "Milwaukee Bucks": "Milwaukee, WI",
    "Minnesota Timberwolves": "Minneapolis, MN",
    "New Orleans Pelicans": "New Orleans, LA",
    "New York Knicks": "New York, NY",
    "Oklahoma City Thunder": "Oklahoma City, OK",
    "Orlando Magic": "Orlando, FL",
    "Philadelphia 76ers": "Philadelphia, PA",
    "Phoenix Suns": "Phoenix, AZ",
    "Portland Trail Blazers": "Portland, OR",
    "Sacramento Kings": "Sacramento, CA",
    "San Antonio Spurs": "San Antonio, TX",
    "Toronto Raptors": "Toronto, ON",
    "Utah Jazz": "Salt Lake City, UT",
    "Washington Wizards": "Washington, DC",
}

NHL_TEAM_CITIES: dict[str, str] = {
    "Anaheim Ducks": "Anaheim, CA",
    "Arizona Coyotes": "Tempe, AZ",
    "Boston Bruins": "Boston, MA",
    "Buffalo Sabres": "Buffalo, NY",
    "Calgary Flames": "Calgary, AB",
    "Carolina Hurricanes": "Raleigh, NC",
    "Chicago Blackhawks": "Chicago, IL",
    "Colorado Avalanche": "Denver, CO",
    "Columbus Blue Jackets": "Columbus, OH",
    "Dallas Stars": "Dallas, TX",
    "Detroit Red Wings": "Detroit, MI",
    "Edmonton Oilers": "Edmonton, AB",
    "Florida Panthers": "Sunrise, FL",
    "Los Angeles Kings": "Los Angeles, CA",
    "Minnesota Wild": "Saint Paul, MN",
    "Montreal Canadiens": "Montreal, QC",
    "Nashville Predators": "Nashville, TN",
    "New Jersey Devils": "Newark, NJ",
    "New York Islanders": "Elmont, NY",
    "New York Rangers": "New York, NY",
    "Ottawa Senators": "Ottawa, ON",
    "Philadelphia Flyers": "Philadelphia, PA",
    "Pittsburgh Penguins": "Pittsburgh, PA",
    "San Jose Sharks": "San Jose, CA",
    "Seattle Kraken": "Seattle, WA",
    "St. Louis Blues": "St. Louis, MO",
    "Tampa Bay Lightning": "Tampa, FL",
    "Toronto Maple Leafs": "Toronto, ON",
    "Utah Hockey Club": "Salt Lake City, UT",
    "Vancouver Canucks": "Vancouver, BC",
    "Vegas Golden Knights": "Las Vegas, NV",
    "Washington Capitals": "Washington, DC",
    "Winnipeg Jets": "Winnipeg, MB",
}

MLB_TEAM_CITIES: dict[str, str] = {
    "Arizona Diamondbacks": "Phoenix, AZ",
    "Atlanta Braves": "Atlanta, GA",
    "Baltimore Orioles": "Baltimore, MD",
    "Boston Red Sox": "Boston, MA",
    "Chicago Cubs": "Chicago, IL",
    "Chicago White Sox": "Chicago, IL",
    "Cincinnati Reds": "Cincinnati, OH",
    "Cleveland Guardians": "Cleveland, OH",
    "Colorado Rockies": "Denver, CO",
    "Detroit Tigers": "Detroit, MI",
    "Houston Astros": "Houston, TX",
    "Kansas City Royals": "Kansas City, MO",
    "Los Angeles Angels": "Anaheim, CA",
    "Los Angeles Dodgers": "Los Angeles, CA",
    "Miami Marlins": "Miami, FL",
    "Milwaukee Brewers": "Milwaukee, WI",
    "Minnesota Twins": "Minneapolis, MN",
    "New York Mets": "Queens, NY",
    "New York Yankees": "Bronx, NY",
    "Athletics": "Sacramento, CA",
    "Philadelphia Phillies": "Philadelphia, PA",
    "Pittsburgh Pirates": "Pittsburgh, PA",
    "San Diego Padres": "San Diego, CA",
    "San Francisco Giants": "San Francisco, CA",
    "Seattle Mariners": "Seattle, WA",
    "St. Louis Cardinals": "St. Louis, MO",
    "Tampa Bay Rays": "St. Petersburg, FL",
    "Texas Rangers": "Arlington, TX",
    "Toronto Blue Jays": "Toronto, ON",
    "Washington Nationals": "Washington, DC",
}

SPORT_CITY_MAPS: dict[str, dict[str, str]] = {
    "NFL": NFL_TEAM_CITIES,
    "NBA": NBA_TEAM_CITIES,
    "NHL": NHL_TEAM_CITIES,
    "MLB": MLB_TEAM_CITIES,
}


def cmd_morning(args: argparse.Namespace) -> None:
    """Fetch schedule, opening odds, and weather for today's games."""
    sport = args.sport.upper()
    config = get_sport_config(sport)
    client = OddsAPIClient()
    sport_key = config.sport_key

    logger.info("Fetching upcoming %s odds...", sport)
    raw_games = client.fetch_odds(sport_key)
    if not raw_games:
        print("No upcoming games found.")
        return

    # Store opening odds + upsert game rows
    count = client.parse_and_store_odds(raw_games, is_closing=False, sport=sport)
    logger.info("Stored %d opening odds rows across %d games", count, len(raw_games))

    # Fetch weather for each game's stadium city
    city_map = SPORT_CITY_MAPS.get(sport, {})
    weather = WeatherClient()
    weather_count = 0
    with get_session() as session:
        for raw in raw_games:
            external_id = raw.get("id")
            home_team = raw.get("home_team", "")
            city = city_map.get(home_team)
            if not city:
                continue

            game = get_game_by_external_id(session, external_id)
            if game is None:
                continue

            wx = weather.get_weather(city)
            if wx:
                game.weather_desc = wx["description"]
                game.temperature_f = wx["temperature_f"]
                game.wind_mph = wx["wind_mph"]
                weather_count += 1

    logger.info("Updated weather for %d games", weather_count)
    print(f"Morning extraction complete: {len(raw_games)} games, {count} odds rows, {weather_count} weather updates.")


def cmd_closing(args: argparse.Namespace) -> None:
    """Capture closing odds for today's scheduled games, with optional line movement analysis."""
    sport = args.sport.upper()
    config = get_sport_config(sport)
    client = OddsAPIClient()
    sport_key = config.sport_key

    logger.info("Fetching closing odds for today's %s games...", sport)
    raw_games = client.fetch_odds(sport_key)
    if not raw_games:
        print("No games found for closing odds.")
        return

    # If --smart is used, filter games starting within the next window (e.g. 2 hours)
    if getattr(args, "smart", False):
        filtered_games = []
        now = datetime.now()
        window_hours = 2
        for g in raw_games:
            commence_time_str = g.get("commence_time", "")
            try:
                # Odds API uses ISO format Z
                commence_time = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
                # Convert to local time for comparison if needed, or keep UTC. 
                # Let's assume naive comparison or handle timezone.
                # Simplest: compare UTC now with UTC commence_time
                from datetime import timezone
                now_utc = datetime.now(timezone.utc)
                diff = (commence_time - now_utc).total_seconds() / 3600
                
                # If game starts in the next 2 hours, OR started in the last 15 mins (catch late starters)
                if -0.25 <= diff <= window_hours:
                    filtered_games.append(g)
            except (ValueError, AttributeError):
                continue
        
        if not filtered_games:
            logger.info("No games starting in the next %d hours. Skipping.", window_hours)
            return
        raw_games = filtered_games
        logger.info("Smart filter: found %d games starting soon.", len(raw_games))

    count = client.parse_and_store_odds(raw_games, is_closing=True, sport=sport)
    logger.info("Stored %d closing odds rows", count)
    print(f"Closing extraction complete: {count} closing odds rows stored.")

    # ---- Line movement analysis ----
    try:
        from betting_agent.intelligence.sentiment import (
            is_ollama_available,
            analyze_line_movement,
            line_moved_significantly,
            fetch_team_context,
            store_sentiment,
        )
        from betting_agent.db.models import Odds, Game
        from datetime import date as date_cls

        if not is_ollama_available():
            logger.info("Ollama not available — skipping line movement analysis")
            return

        logger.info("Analyzing line movements...")
        analyzed = 0
        with get_session() as session:
            today_games = (
                session.query(Game)
                .filter(Game.sport == sport, Game.game_date == date_cls.today())
                .all()
            )
            for game in today_games:
                opening = (
                    session.query(Odds)
                    .filter(Odds.game_id == game.id, Odds.is_closing == False, Odds.bet_type == "moneyline")
                    .first()
                )
                closing = (
                    session.query(Odds)
                    .filter(Odds.game_id == game.id, Odds.is_closing == True, Odds.bet_type == "moneyline")
                    .first()
                )
                if not opening or not closing:
                    continue

                open_dict = {
                    "home_price": opening.home_price,
                    "away_price": opening.away_price,
                    "spread_home": opening.spread_home,
                }
                close_dict = {
                    "home_price": closing.home_price,
                    "away_price": closing.away_price,
                    "spread_home": closing.spread_home,
                }

                if not line_moved_significantly(open_dict, close_dict):
                    continue

                current_year = date_cls.today().year
                ctx = fetch_team_context(game.home_team, current_year, depth="basic", sport=sport)
                ctx += "\n" + fetch_team_context(game.away_team, current_year, depth="basic", sport=sport)

                result = analyze_line_movement(
                    game.home_team, game.away_team, open_dict, close_dict, ctx
                )
                print(f"\n  {game.away_team} @ {game.home_team}")
                print(f"  Line movement: {result['explanation']}")
                if result["sharp_side"]:
                    print(f"  Sharp side: {result['sharp_side']}")

                import json as _json
                store_sentiment(
                    game_id=game.id,
                    team=f"{game.home_team} vs {game.away_team}",
                    score=0.0,
                    summary=_json.dumps(result),
                    analysis_type="line_movement",
                )
                analyzed += 1

        if analyzed:
            print(f"\nAnalyzed line movement for {analyzed} games.")
        else:
            print("\nNo significant line movements detected.")
    except Exception as exc:
        logger.warning("Line movement analysis failed: %s", exc)


def cmd_postgame(args: argparse.Namespace) -> None:
    """Fetch final scores for recently completed games."""
    sport = args.sport.upper()
    config = get_sport_config(sport)
    client = OddsAPIClient()
    sport_key = config.sport_key

    logger.info("Fetching %s scores from last 3 days...", sport)
    scores = client.fetch_scores(sport_key, days_from=3)
    if not scores:
        print("No scores returned.")
        return

    updated = 0
    with get_session() as session:
        for s in scores:
            if not s.get("completed"):
                continue

            external_id = s.get("id")
            game = get_game_by_external_id(session, external_id)
            if game is None:
                continue
            if game.status == "final":
                continue

            home_score = away_score = None
            for team_score in s.get("scores", []) or []:
                if team_score.get("name") == game.home_team:
                    home_score = int(team_score.get("score", 0))
                elif team_score.get("name") == game.away_team:
                    away_score = int(team_score.get("score", 0))

            if home_score is not None and away_score is not None:
                game.home_score = home_score
                game.away_score = away_score
                game.status = "final"
                updated += 1

    logger.info("Updated %d games with final scores", updated)
    print(f"Postgame extraction complete: {updated} games updated with final scores.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily extraction orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd_name, cmd_help in [
        ("morning", "Fetch schedule + opening odds + weather"),
        ("closing", "Capture closing odds"),
        ("postgame", "Fetch final scores"),
    ]:
        sub = subparsers.add_parser(cmd_name, help=cmd_help)
        sub.add_argument(
            "--sport",
            type=str,
            default="NFL",
            help=f"Sport ({', '.join(available_sports())})",
        )
        if cmd_name == "closing":
            sub.add_argument(
                "--smart",
                action="store_true",
                help="Only fetch odds for games starting in the next 2 hours",
            )

    args = parser.parse_args()

    commands = {
        "morning": cmd_morning,
        "closing": cmd_closing,
        "postgame": cmd_postgame,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
