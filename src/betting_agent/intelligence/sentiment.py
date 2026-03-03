"""
Ollama-based sentiment and matchup analysis.

Provides:
- fetch_team_context(): Rich team context from nflreadpy (injuries, depth chart, stats)
- analyze_sentiment(): Single-team health score (legacy, still used for backtesting)
- analyze_game(): Combined game analysis — 1 LLM call per matchup
- analyze_line_movement(): Explain line shifts with LLM reasoning
- get_game_sentiment(): Legacy wrapper returning (home_score, away_score)
- store_sentiment(): Persist analysis to DB

Scores range from -1.0 (devastating) to +1.0 (fully healthy/positive).
Gracefully degrades: if Ollama is unavailable, returns neutral scores.
"""

from __future__ import annotations

import json
import logging

import requests

from betting_agent.config import settings
from betting_agent.db.models import Sentiment
from betting_agent.db.session import get_session

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> str:
    """Extract JSON from an LLM response that may include preamble text or code fences."""
    cleaned = raw.strip()
    # Strip markdown code fences anywhere in the response
    if "```" in cleaned:
        start = cleaned.find("```")
        body_start = cleaned.find("\n", start)
        if body_start == -1:
            body_start = start + 3
        else:
            body_start += 1
        end = cleaned.find("```", body_start)
        if end != -1:
            cleaned = cleaned[body_start:end].strip()
        else:
            cleaned = cleaned[body_start:].strip()

    # If still not starting with {, find the first {
    if not cleaned.startswith("{"):
        brace = cleaned.find("{")
        if brace != -1:
            cleaned = cleaned[brace:]

    # Trim trailing text after the last matching }
    # (handles "Extra data" errors from LLMs appending commentary)
    if cleaned.startswith("{"):
        depth = 0
        end_pos = -1
        for i, ch in enumerate(cleaned):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        if end_pos != -1:
            cleaned = cleaned[: end_pos + 1]

    # Fix single quotes → double quotes (common LLM mistake)
    # Only do this if the string fails to parse as-is
    try:
        json.loads(cleaned)
    except json.JSONDecodeError:
        # Replace single-quoted keys/values with double quotes
        # Simple heuristic: replace ' with " when adjacent to { } , : [ ]
        import re
        fixed = re.sub(r"(?<=[\{,:\[])\s*'", ' "', cleaned)
        fixed = re.sub(r"'\s*(?=[\},:}\]])", '"', fixed)
        # Also handle keys at start: {'key' → {"key"
        fixed = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', fixed)
        try:
            json.loads(fixed)
            cleaned = fixed
        except json.JSONDecodeError:
            pass  # return best-effort cleaned string

    return cleaned


def is_ollama_available() -> bool:
    """Check if Ollama is running and reachable."""
    try:
        base = settings.ollama_url.replace("/api/generate", "")
        resp = requests.get(base, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step 1 — Enriched team context
# ---------------------------------------------------------------------------

def fetch_team_context(
    team: str, season: int, week: int | None = None, depth: str = "full",
    sport: str = "NFL",
) -> str:
    """
    Build context text for a team from sport-specific data sources.

    depth="basic" — minimal context (fast)
    depth="full"  — full context with player stats etc.
    """
    if sport.upper() == "NBA":
        return _fetch_nba_team_context(team, season, depth)
    return _fetch_nfl_team_context(team, season, week, depth)


def _fetch_nfl_team_context(
    team: str, season: int, week: int | None = None, depth: str = "full"
) -> str:
    """Build context text for an NFL team from nflreadpy data sources."""
    lines = [f"Team: {team}, Season: {season}"]

    # ---- Injuries (always included) ----
    injuries_df = None
    try:
        import nflreadpy as nfl
        injuries_df = nfl.load_injuries([season]).to_pandas()
        team_inj = injuries_df[
            injuries_df["team"].str.contains(team.split()[-1], case=False, na=False)
        ]
        if week is not None:
            team_inj = team_inj[team_inj["week"] == week]

        if not team_inj.empty:
            lines.append("\nInjury Report:")
            for _, row in team_inj.head(15).iterrows():
                name = row.get("full_name", row.get("player", "Unknown"))
                status = row.get("report_status", "Unknown")
                practice = row.get("practice_status", "")
                lines.append(f"  - {name}: {status} ({practice})")
        else:
            lines.append("\nNo injuries reported.")
    except Exception as exc:
        logger.debug("Could not load injuries for %s: %s", team, exc)
        lines.append("\nInjury data unavailable.")

    if depth == "basic":
        return "\n".join(lines)

    # ---- Depth chart cross-reference ----
    try:
        import nflreadpy as nfl
        dc = nfl.load_depth_charts([season]).to_pandas()
        team_dc = dc[dc["club_code"].str.contains(team.split()[-1][:3], case=False, na=False)]

        if week is not None:
            team_dc = team_dc[team_dc["week"] == week]

        if not team_dc.empty and injuries_df is not None:
            team_inj_names = set()
            team_inj_filtered = injuries_df[
                injuries_df["team"].str.contains(team.split()[-1], case=False, na=False)
            ]
            if week is not None:
                team_inj_filtered = team_inj_filtered[team_inj_filtered["week"] == week]
            for _, row in team_inj_filtered.iterrows():
                name = row.get("full_name", row.get("player", ""))
                if name:
                    team_inj_names.add(name.lower())

            # Flag injured starters (depth_team rank 1)
            starters = team_dc[team_dc["depth_team"] == 1]
            injured_starters = []
            for _, row in starters.iterrows():
                name = row.get("full_name", row.get("player_name", ""))
                if name and name.lower() in team_inj_names:
                    pos = row.get("position", "")
                    injured_starters.append(f"{pos} {name}")

            if injured_starters:
                lines.append("\nInjured Starters:")
                for s in injured_starters[:10]:
                    lines.append(f"  * {s}")
    except Exception as exc:
        logger.debug("Could not load depth charts for %s: %s", team, exc)

    # ---- Recent team performance ----
    try:
        import nflreadpy as nfl
        team_stats = nfl.load_team_stats([season]).to_pandas()
        # Filter to this team
        ts = team_stats[
            team_stats["recent_team"].str.contains(team.split()[-1], case=False, na=False)
        ]
        if not ts.empty:
            # Aggregate or take latest available
            latest = ts.tail(1).iloc[0]
            stat_parts = []
            for col, label in [
                ("points_for", "PF"),
                ("points_against", "PA"),
                ("total_yards", "YDS"),
                ("turnovers", "TO"),
                ("sacks", "SACKS"),
            ]:
                if col in latest.index and latest[col] is not None:
                    stat_parts.append(f"{label}={latest[col]}")
            if stat_parts:
                lines.append(f"\nSeason Stats: {', '.join(stat_parts)}")
    except Exception as exc:
        logger.debug("Could not load team stats for %s: %s", team, exc)

    # ---- Key player stats (QB, top RB, top WR) ----
    try:
        import nflreadpy as nfl
        player_stats = nfl.load_player_stats([season]).to_pandas()
        team_ps = player_stats[
            player_stats["recent_team"].str.contains(team.split()[-1], case=False, na=False)
        ]

        if not team_ps.empty:
            lines.append("\nKey Players (recent):")

            # Top QB by passing yards
            qbs = team_ps[team_ps["position"] == "QB"].sort_values(
                "passing_yards", ascending=False
            )
            if not qbs.empty:
                qb = qbs.iloc[0]
                name = qb.get("player_display_name", qb.get("player_name", "?"))
                pyds = int(qb.get("passing_yards", 0))
                ptd = int(qb.get("passing_tds", 0))
                ints = int(qb.get("interceptions", 0))
                lines.append(f"  QB {name}: {pyds} yds, {ptd} TD, {ints} INT")

            # Top RB by rushing yards
            rbs = team_ps[team_ps["position"] == "RB"].sort_values(
                "rushing_yards", ascending=False
            )
            if not rbs.empty:
                rb = rbs.iloc[0]
                name = rb.get("player_display_name", rb.get("player_name", "?"))
                ryds = int(rb.get("rushing_yards", 0))
                rtd = int(rb.get("rushing_tds", 0))
                lines.append(f"  RB {name}: {ryds} yds, {rtd} TD")

            # Top WR by receiving yards
            wrs = team_ps[team_ps["position"] == "WR"].sort_values(
                "receiving_yards", ascending=False
            )
            if not wrs.empty:
                wr = wrs.iloc[0]
                name = wr.get("player_display_name", wr.get("player_name", "?"))
                recyds = int(wr.get("receiving_yards", 0))
                rectd = int(wr.get("receiving_tds", 0))
                lines.append(f"  WR {name}: {recyds} yds, {rectd} TD")
    except Exception as exc:
        logger.debug("Could not load player stats for %s: %s", team, exc)

    # ---- Roster moves (weekly diff) ----
    try:
        import nflreadpy as nfl
        if week is not None and week > 1:
            rosters = nfl.load_rosters_weekly([season]).to_pandas()
            team_rost = rosters[
                rosters["team"].str.contains(team.split()[-1], case=False, na=False)
            ]
            curr = set(
                team_rost[team_rost["week"] == week]["player_id"].dropna()
            )
            prev = set(
                team_rost[team_rost["week"] == week - 1]["player_id"].dropna()
            )
            added = curr - prev
            dropped = prev - curr

            if added or dropped:
                lines.append("\nRoster Changes (vs prior week):")
                if added:
                    added_rows = team_rost[
                        (team_rost["week"] == week) & (team_rost["player_id"].isin(added))
                    ]
                    for _, r in added_rows.head(5).iterrows():
                        name = r.get("player_name", r.get("full_name", "?"))
                        pos = r.get("position", "")
                        lines.append(f"  + {pos} {name}")
                if dropped:
                    dropped_rows = team_rost[
                        (team_rost["week"] == week - 1) & (team_rost["player_id"].isin(dropped))
                    ]
                    for _, r in dropped_rows.head(5).iterrows():
                        name = r.get("player_name", r.get("full_name", "?"))
                        pos = r.get("position", "")
                        lines.append(f"  - {pos} {name}")
    except Exception as exc:
        logger.debug("Could not load roster data for %s: %s", team, exc)

    return "\n".join(lines)


def _fetch_nba_team_context(
    team: str, season: int, depth: str = "full",
) -> str:
    """
    Build context text for an NBA team from nba_api data sources.

    Uses LeagueDashTeamStats for team record/stats and LeagueDashPlayerStats
    for key player stats. Rate-limited with 0.6s between calls.
    """
    import time
    from betting_agent.sports.nba.loader import int_to_nba_season, NBA_ABBREV_TO_FULL
    from betting_agent.config import settings as cfg

    # Resolve team name — could be abbreviation or full name
    display_name = NBA_ABBREV_TO_FULL.get(team, team)
    lines = [f"Team: {display_name} ({team}), Season: {season}"]
    lines.append("\nNote: NBA injury data is not available via API.")

    nba_season = int_to_nba_season(season)

    if depth == "basic":
        return "\n".join(lines)

    # ---- Team stats ----
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats

        team_stats = LeagueDashTeamStats(season=nba_season).get_data_frames()[0]
        time.sleep(cfg.nba_api_rate_limit)

        ts = team_stats[team_stats["TEAM_ABBREVIATION"] == team]
        if ts.empty and len(team) > 3:
            # Try matching by full name
            ts = team_stats[team_stats["TEAM_NAME"].str.contains(team, case=False, na=False)]

        if not ts.empty:
            row = ts.iloc[0]
            wins = int(row.get("W", 0))
            losses = int(row.get("L", 0))
            ppg = float(row.get("PTS", 0)) / max(wins + losses, 1)
            opp_ppg = float(row.get("OPP_PTS", 0)) / max(wins + losses, 1) if "OPP_PTS" in row.index else 0
            fg_pct = float(row.get("FG_PCT", 0)) * 100
            fg3_pct = float(row.get("FG3_PCT", 0)) * 100
            reb = float(row.get("REB", 0)) / max(wins + losses, 1)
            ast = float(row.get("AST", 0)) / max(wins + losses, 1)
            tov = float(row.get("TOV", 0)) / max(wins + losses, 1)

            lines.append(f"\nRecord: {wins}-{losses}")
            lines.append(
                f"Season Stats: PPG={ppg:.1f}, OPP PPG={opp_ppg:.1f}, "
                f"FG%={fg_pct:.1f}, 3P%={fg3_pct:.1f}, "
                f"RPG={reb:.1f}, APG={ast:.1f}, TOPG={tov:.1f}"
            )
    except Exception as exc:
        logger.debug("Could not load NBA team stats for %s: %s", team, exc)

    # ---- Key players (top 5 by PPG) ----
    try:
        from nba_api.stats.endpoints import LeagueDashPlayerStats

        player_stats = LeagueDashPlayerStats(season=nba_season).get_data_frames()[0]
        time.sleep(cfg.nba_api_rate_limit)

        team_players = player_stats[player_stats["TEAM_ABBREVIATION"] == team]
        if team_players.empty and len(team) > 3:
            team_players = player_stats[
                player_stats["TEAM_ABBREVIATION"].str.contains(team[:3], case=False, na=False)
            ]

        if not team_players.empty:
            gp_col = "GP"
            pts_col = "PTS"
            if gp_col in team_players.columns and pts_col in team_players.columns:
                team_players = team_players.copy()
                team_players["PPG"] = team_players[pts_col] / team_players[gp_col].clip(lower=1)
                top5 = team_players.nlargest(5, "PPG")

                lines.append("\nKey Players (by PPG):")
                for _, p in top5.iterrows():
                    name = p.get("PLAYER_NAME", "?")
                    ppg = p["PPG"]
                    gp = int(p.get(gp_col, 0))
                    rpg = float(p.get("REB", 0)) / max(gp, 1)
                    apg = float(p.get("AST", 0)) / max(gp, 1)
                    lines.append(
                        f"  {name}: {ppg:.1f} PPG, {rpg:.1f} RPG, {apg:.1f} APG ({gp} GP)"
                    )
    except Exception as exc:
        logger.debug("Could not load NBA player stats for %s: %s", team, exc)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy single-team analysis (kept for backtesting compatibility)
# ---------------------------------------------------------------------------

def analyze_sentiment(context_text: str, team: str) -> dict:
    """
    Send context to Ollama and get a sentiment score + summary.

    Returns: {"score": float, "summary": str}
    Score range: -1.0 (devastating) to +1.0 (fully healthy/positive)
    """
    prompt = f"""Analyze the following NFL team context and return a JSON object with exactly two fields:
- "score": a float from -1.0 (severely negative outlook — key starters out, major injuries) to +1.0 (fully healthy, positive momentum)
- "summary": a brief one-sentence summary of the team's health/situation

Team context:
{context_text}

Respond with ONLY valid JSON, no other text. Example: {{"score": -0.3, "summary": "Missing starting QB and top WR"}}"""

    try:
        resp = requests.post(
            settings.ollama_url,
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=settings.ollama_timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        cleaned = _extract_json(raw)

        result = json.loads(cleaned)
        score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
        summary = str(result.get("summary", ""))
        return {"score": score, "summary": summary}
    except Exception as exc:
        logger.warning("Ollama sentiment failed for %s: %s", team, exc)
        return {"score": 0.0, "summary": "Unavailable"}


def get_game_sentiment(
    home: str, away: str, season: int, week: int | None = None
) -> tuple[float, float]:
    """
    Get sentiment scores for both teams in a game.
    Returns (home_score, away_score).
    """
    home_ctx = fetch_team_context(home, season, week, depth="basic")
    home_result = analyze_sentiment(home_ctx, home)

    away_ctx = fetch_team_context(away, season, week, depth="basic")
    away_result = analyze_sentiment(away_ctx, away)

    return home_result["score"], away_result["score"]


# ---------------------------------------------------------------------------
# Step 2 — Combined game analysis (1 LLM call per game)
# ---------------------------------------------------------------------------

def analyze_game(
    home_team: str,
    away_team: str,
    home_context: str,
    away_context: str,
    odds_summary: str | None = None,
    sport: str = "NFL",
) -> dict:
    """
    Analyze a full matchup in 1 LLM call.

    Returns:
        {
            "home_score": float,      # -1.0 to +1.0
            "away_score": float,      # -1.0 to +1.0
            "matchup_edge": str,      # "home" / "away" / "neutral"
            "key_factors": list[str], # top 3 factors
            "summary": str,           # 2-3 sentence narrative
        }
    """
    odds_section = ""
    if odds_summary:
        odds_section = f"\nCurrent Odds:\n{odds_summary}\n"

    prompt = f"""You are an expert {sport} analyst. Analyze this matchup and return structured JSON.

HOME TEAM CONTEXT:
{home_context}

AWAY TEAM CONTEXT:
{away_context}
{odds_section}
Analyze:
1. Each team's health and recent form
2. Key matchup dynamics (e.g. "elite pass rush vs weak O-line")
3. Any major edges one team holds

Return ONLY valid JSON with these exact fields:
{{
    "home_score": <float -1.0 to 1.0, team health/momentum>,
    "away_score": <float -1.0 to 1.0, team health/momentum>,
    "matchup_edge": "<home|away|neutral>",
    "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
    "summary": "<2-3 sentence matchup narrative>"
}}"""

    default = {
        "home_score": 0.0,
        "away_score": 0.0,
        "matchup_edge": "neutral",
        "key_factors": [],
        "summary": "Analysis unavailable",
    }

    try:
        resp = requests.post(
            settings.ollama_url,
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=settings.ollama_timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        cleaned = _extract_json(raw)

        result = json.loads(cleaned)
        return {
            "home_score": max(-1.0, min(1.0, float(result.get("home_score", 0.0)))),
            "away_score": max(-1.0, min(1.0, float(result.get("away_score", 0.0)))),
            "matchup_edge": str(result.get("matchup_edge", "neutral")),
            "key_factors": [str(f) for f in result.get("key_factors", [])][:5],
            "summary": str(result.get("summary", "")),
        }
    except Exception as exc:
        logger.warning("Ollama game analysis failed (%s vs %s): %s", home_team, away_team, exc)
        return default


# ---------------------------------------------------------------------------
# Step 3 — Line movement analysis
# ---------------------------------------------------------------------------

def analyze_line_movement(
    home_team: str,
    away_team: str,
    opening_odds: dict,
    closing_odds: dict,
    game_context: str,
) -> dict:
    """
    Explain why a line moved between open and close.

    Only call when the line moved significantly (spread >= 1.5 pts or ML >= 15 cents).

    Returns: {"explanation": str, "sharp_side": str | None}
    """
    prompt = f"""You are an expert NFL odds analyst. Explain why the betting line moved for this game.

MATCHUP: {away_team} @ {home_team}

OPENING ODDS:
{json.dumps(opening_odds, indent=2)}

CLOSING ODDS:
{json.dumps(closing_odds, indent=2)}

GAME CONTEXT:
{game_context}

Analyze the line movement and identify:
1. What changed between open and close (injuries, weather, sharp money, public money)
2. Which side appears to be the "sharp" side (where professional bettors moved the line)

Return ONLY valid JSON:
{{
    "explanation": "<2-3 sentence explanation of why the line moved>",
    "sharp_side": "<home|away|null if unclear>"
}}"""

    default = {"explanation": "Line movement analysis unavailable", "sharp_side": None}

    try:
        resp = requests.post(
            settings.ollama_url,
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=settings.ollama_timeout,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        cleaned = _extract_json(raw)

        result = json.loads(cleaned)
        sharp = result.get("sharp_side")
        if sharp and sharp.lower() == "null":
            sharp = None
        return {
            "explanation": str(result.get("explanation", "")),
            "sharp_side": sharp,
        }
    except Exception as exc:
        logger.warning("Line movement analysis failed (%s vs %s): %s", home_team, away_team, exc)
        return default


def line_moved_significantly(opening: dict, closing: dict) -> bool:
    """Check if the line moved enough to warrant analysis."""
    # Spread shift >= 1.5 points
    open_spread = opening.get("spread_home")
    close_spread = closing.get("spread_home")
    if open_spread is not None and close_spread is not None:
        if abs(float(close_spread) - float(open_spread)) >= 1.5:
            return True

    # Moneyline shift >= 15 cents (e.g. -130 to -145)
    for key in ("home_price", "away_price"):
        open_ml = opening.get(key)
        close_ml = closing.get(key)
        if open_ml is not None and close_ml is not None:
            if abs(int(close_ml) - int(open_ml)) >= 15:
                return True

    return False


# ---------------------------------------------------------------------------
# Step 4 — Persist analysis to DB
# ---------------------------------------------------------------------------

def store_sentiment(
    game_id: int,
    team: str,
    score: float,
    summary: str,
    model: str | None = None,
    analysis_type: str = "injury",
) -> None:
    """Save sentiment/analysis to the DB."""
    with get_session() as session:
        row = Sentiment(
            game_id=game_id,
            team=team,
            sentiment_score=score,
            summary=summary,
            model_used=model or settings.ollama_model,
            source="ollama",
            analysis_type=analysis_type,
        )
        session.add(row)
