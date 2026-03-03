"""
Common reusable DB queries.
"""

from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from betting_agent.db.models import Game, Odds, Pick


def get_game_by_external_id(session: Session, external_id: str) -> Optional[Game]:
    return session.query(Game).filter(Game.external_id == external_id).first()


def get_ungraded_picks(session: Session) -> list[Pick]:
    return (
        session.query(Pick)
        .filter(Pick.result.is_(None))
        .join(Game)
        .filter(Game.status == "final")
        .all()
    )


def get_picks_for_date(session: Session, pick_date: date, sport: str = "NFL") -> list[Pick]:
    return (
        session.query(Pick)
        .filter(Pick.pick_date == pick_date, Pick.sport == sport)
        .all()
    )


def get_closing_odds(session: Session, game_id: int, bet_type: str) -> Optional[Odds]:
    return (
        session.query(Odds)
        .filter(
            Odds.game_id == game_id,
            Odds.bet_type == bet_type,
            Odds.is_closing == True,
        )
        .first()
    )


def get_scheduled_games(session: Session, sport: str = "NFL", target_date: Optional[date] = None) -> list[Game]:
    q = session.query(Game).filter(Game.sport == sport, Game.status == "scheduled")
    if target_date:
        q = q.filter(Game.game_date == target_date)
    return q.all()


def upsert_game(session: Session, game_data: dict) -> Game:
    """Insert or update a game by external_id."""
    external_id = game_data.get("external_id")
    if external_id:
        existing = get_game_by_external_id(session, external_id)
        if existing:
            for k, v in game_data.items():
                setattr(existing, k, v)
            return existing
    game = Game(**game_data)
    session.add(game)
    session.flush()
    return game
