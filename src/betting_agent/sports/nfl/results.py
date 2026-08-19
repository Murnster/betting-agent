"""
Final scores for NFL games, sourced from nflreadpy.

Picks created by props.py carry an Odds API event id but no score, so their
Game rows are written as `status="scheduled"`. Grading refuses to settle a
pick whose game is not final, which would leave every prop pick pending
forever unless something fills the scores in.

`extract.py postgame` can do that from the Odds API, but it costs credits and
only reaches back three days. nflreadpy publishes the same scores for free
and for the whole season, so the grading path uses this instead.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from betting_agent.db.models import Game
from betting_agent.db.session import get_session
from betting_agent.sports.teams import same_team

logger = logging.getLogger(__name__)

#: A schedule row must land this close to the stored game date to match.
DATE_TOLERANCE = pd.Timedelta(days=2)


def _season_for(d: date) -> int:
    return d.year if d.month >= 8 else d.year - 1


def finalize_nfl_games(target_date: date | None = None) -> int:
    """
    Fill in scores and mark NFL games final from published schedules.

    Only touches games that have already kicked off and are still unfinished.
    Returns the number of games finalized.
    """
    today = date.today()
    with get_session() as session:
        pending = (
            session.query(Game)
            .filter(Game.sport == "NFL", Game.status != "final")
            .filter(Game.game_date <= today)
            .all()
        )
        if target_date is not None:
            pending = [g for g in pending if g.game_date == target_date]
        if not pending:
            return 0

        seasons = sorted({_season_for(g.game_date) for g in pending})
        logger.info("Looking up final scores for %d NFL games (seasons %s)",
                    len(pending), seasons)

        from betting_agent.sports.nfl.features import normalise_raw_schedules
        from betting_agent.sports.nfl.loader import NFLLoader

        try:
            sched = normalise_raw_schedules(NFLLoader().load_schedules(seasons).to_pandas())
        except Exception as exc:
            logger.warning("Could not load NFL schedules: %s", exc)
            return 0
        if sched.empty:
            return 0
        played = sched[sched["home_score"].notna() & sched["away_score"].notna()]

        finalized = 0
        for game in pending:
            stamp = pd.Timestamp(game.game_date)
            near = played[(played["game_date"] - stamp).abs() <= DATE_TOLERANCE]
            for row in near.itertuples(index=False):
                if same_team("NFL", row.home_team, game.home_team) and same_team(
                    "NFL", row.away_team, game.away_team
                ):
                    game.home_score = int(row.home_score)
                    game.away_score = int(row.away_score)
                    game.status = "final"
                    finalized += 1
                    break

    if finalized:
        logger.info("Finalized %d NFL games from published schedules", finalized)
    return finalized
