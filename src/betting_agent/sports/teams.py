"""
Canonical team naming.

Three write paths put team names into the games table, and they disagreed:
loaders seed sport-native abbreviations ("KC", "BOS"), while anything created
from an Odds API event carried full club names ("Kansas City Chiefs"). Picks
always store the Odds API spelling. Grading and CLV then compare those strings
directly, so a pick on a game row seeded from the other path graded against the
wrong team — wins recorded as losses, spreads measured from the wrong side.

The canonical form is the sport-native abbreviation: it is what every loader
already writes, what all historical rows use, and what the models train on.
Odds API names are converted on entry.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _full_to_abbrev(sport: str) -> dict[str, str]:
    """Full club name → abbreviation for one sport ({} if unsupported)."""
    sport = sport.upper()
    try:
        if sport == "NFL":
            from betting_agent.sports.nfl.loader import (
                NFL_FULL_TO_ABBREV,
                NFL_LEGACY_FULL_TO_ABBREV,
            )
            return {**NFL_FULL_TO_ABBREV, **NFL_LEGACY_FULL_TO_ABBREV}
        if sport == "NBA":
            from betting_agent.sports.nba.loader import NBA_FULL_TO_ABBREV
            return dict(NBA_FULL_TO_ABBREV)
        if sport == "NHL":
            from betting_agent.sports.nhl.loader import NHL_FULL_TO_ABBREV
            return dict(NHL_FULL_TO_ABBREV)
        if sport == "MLB":
            from betting_agent.sports.mlb.loader import MLB_FULL_TO_ABBREV
            return dict(MLB_FULL_TO_ABBREV)
    except ImportError as exc:  # pragma: no cover - defensive
        logger.warning("Could not load %s team names: %s", sport, exc)
    return {}


@lru_cache(maxsize=None)
def _abbrev_to_full(sport: str) -> dict[str, str]:
    """Abbreviation → full club name. Built from the forward map so the two
    can never drift apart."""
    out: dict[str, str] = {}
    for full, abbrev in _full_to_abbrev(sport).items():
        out.setdefault(abbrev, full)
    return out


def canonical_team(sport: str, name: str | None) -> str | None:
    """
    Return the canonical abbreviation for a team.

    Accepts an abbreviation (returned unchanged) or a full club name. Unknown
    names are returned stripped but otherwise untouched, so callers can still
    compare like with like.
    """
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return key
    if key in _abbrev_to_full(sport):
        return key
    return _full_to_abbrev(sport).get(key, key)


def display_team(sport: str, name: str | None) -> str | None:
    """Full club name for display; falls back to the input when unknown."""
    if name is None:
        return None
    key = str(name).strip()
    return _abbrev_to_full(sport).get(key, key)


def same_team(sport: str, a: str | None, b: str | None) -> bool:
    """True when two spellings refer to the same club."""
    if a is None or b is None:
        return False
    return canonical_team(sport, a) == canonical_team(sport, b)
