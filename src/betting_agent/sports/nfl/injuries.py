"""
Deterministic injury / QB1 checks for NFL player props.

Books price injury designations in; the projection model does not see them.
This module turns two free nflreadpy feeds into flags on prop candidates:

- `load_injuries` — the official game-status report (Out / Doubtful /
  Questionable). nflreadpy gates it on its own notion of the current
  season, which flips on the Thursday after Labor Day, so requesting the
  new season before then raises — every loader here returns an EMPTY frame
  on any failure and the flag builder is then a no-op.
- `load_depth_charts` — identifies each team's QB1. The season argument is
  ignored by nflreadpy (the whole current-format dataset comes back), and
  teams publish on different days, so the latest chart is taken PER TEAM.

Policy (applied in scripts/props.py): a player listed Out or Doubtful is a
`drop` — a line still hanging on game day is a void or a limited-snap loss.
Questionable and a QB1 who is Out cut both ways, so they are surfaced on the
pick card and to the validator but not acted on.
"""

from __future__ import annotations

import logging

import pandas as pd

from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.validator.schemas import DeterministicFlag
from betting_agent.sports.nfl.props import normalize_player
from betting_agent.sports.teams import canonical_team

logger = logging.getLogger(__name__)

INJURY_COLUMNS = [
    "full_name", "team", "position", "report_status", "practice_status", "gsis_id",
]
DROP_STATUSES = ("Out", "Doubtful")
FLAG_STATUSES = ("Questionable",)


def load_injury_report(season: int, week: int) -> pd.DataFrame:
    """
    Official injury report rows for one week, keyed by `player_key`.
    Empty frame (same columns) when the feed is unavailable or gated.
    """
    import nflreadpy as nfl

    empty = pd.DataFrame(columns=INJURY_COLUMNS + ["player_key"])
    try:
        inj = nfl.load_injuries([season]).to_pandas()
    except Exception as exc:
        logger.warning("Injury report unavailable for %s week %s: %s", season, week, exc)
        return empty
    if inj.empty or "week" not in inj.columns:
        return empty
    inj = inj[inj["week"] == week]
    cols = [c for c in INJURY_COLUMNS if c in inj.columns]
    inj = inj[cols].copy()
    for col in INJURY_COLUMNS:
        if col not in inj.columns:
            inj[col] = None
    inj["player_key"] = inj["full_name"].map(normalize_player)
    return inj.reset_index(drop=True)


def qb1_by_team(season: int) -> dict[str, tuple[str, str | None]]:
    """
    {team_abbrev: (qb1_name, gsis_id)} from each team's most recent depth
    chart. `pos_abb == "QB"` is the position column — `pos_grp` is a
    formation label ("3WR 1TE") and must not be used. {} on failure.
    """
    import nflreadpy as nfl

    try:
        dc = nfl.load_depth_charts([season]).to_pandas()
    except Exception as exc:
        logger.warning("Depth charts unavailable for %s: %s", season, exc)
        return {}
    return qb1_from_depth_chart(dc)


def qb1_from_depth_chart(dc: pd.DataFrame) -> dict[str, tuple[str, str | None]]:
    """Pure part of qb1_by_team(), for testing against a synthetic frame."""
    needed = {"team", "dt", "pos_abb", "pos_rank", "player_name"}
    if dc is None or dc.empty or not needed <= set(dc.columns):
        return {}
    latest = dc.groupby("team")["dt"].transform("max")
    current = dc[dc["dt"] == latest]
    qbs = current[(current["pos_abb"] == "QB") & (current["pos_rank"] == 1)]
    out: dict[str, tuple[str, str | None]] = {}
    for row in qbs.itertuples(index=False):
        gsis = getattr(row, "gsis_id", None)
        out[str(row.team)] = (str(row.player_name), None if pd.isna(gsis) else str(gsis))
    return out


def _status_by_key(injuries: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """(player_key → report_status, gsis_id → report_status) for listed players."""
    by_key: dict[str, str] = {}
    by_gsis: dict[str, str] = {}
    if injuries is None or injuries.empty:
        return by_key, by_gsis
    for row in injuries.itertuples(index=False):
        status = getattr(row, "report_status", None)
        if status is None or pd.isna(status):
            continue
        key = getattr(row, "player_key", None) or normalize_player(getattr(row, "full_name", ""))
        if key:
            by_key[key] = str(status)
        gsis = getattr(row, "gsis_id", None)
        if gsis is not None and not pd.isna(gsis):
            by_gsis[str(gsis)] = str(status)
    return by_key, by_gsis


def prop_injury_flags(
    candidates: list[BetCandidate],
    injuries: pd.DataFrame | None,
    qb1: dict[str, tuple[str, str | None]] | None,
    player_teams: dict[str, str] | None = None,
) -> list[DeterministicFlag]:
    """
    Flags for prop candidates from the injury report and depth charts.

    - player Out/Doubtful  → type="player_injury", severity="high", drop=True
    - player Questionable  → type="player_injury", severity="medium"
    - team QB1 Out/Doubtful → type="qb_out", severity="high" on every prop
      for that team (flag only)

    `player_teams` (player_key → abbrev) says which side each player is on;
    without it the QB check is skipped for that player.
    """
    if not candidates:
        return []
    by_key, by_gsis = _status_by_key(injuries)
    if not by_key and not by_gsis:
        return []

    flags: list[DeterministicFlag] = []
    qb_flag_by_team: dict[str, DeterministicFlag | None] = {}

    def qb_flag(team: str) -> DeterministicFlag | None:
        if team in qb_flag_by_team:
            return qb_flag_by_team[team]
        flag = None
        entry = (qb1 or {}).get(team)
        if entry:
            name, gsis = entry
            status = (by_gsis.get(gsis) if gsis else None) or by_key.get(normalize_player(name))
            if status in DROP_STATUSES:
                flag = DeterministicFlag(
                    type="qb_out", team=team, severity="high", player=name,
                    detail=f"{team} QB1 {name} is {status}",
                )
        qb_flag_by_team[team] = flag
        return flag

    for cand in candidates:
        if cand.bet_type != "prop" or not cand.player:
            continue
        key = normalize_player(cand.player)
        status = by_key.get(key)
        if status in DROP_STATUSES:
            flags.append(DeterministicFlag(
                type="player_injury", severity="high", player=cand.player, drop=True,
                detail=f"{cand.player} listed {status} on the official injury report",
            ))
        elif status in FLAG_STATUSES:
            flags.append(DeterministicFlag(
                type="player_injury", severity="medium", player=cand.player,
                detail=f"{cand.player} listed {status}",
            ))

        team = (player_teams or {}).get(key)
        if team is None:
            continue
        team = canonical_team("NFL", team)
        qf = qb_flag(team)
        if qf is not None:
            flags.append(qf.model_copy(update={"player": cand.player}))
    return flags


def apply_injury_policy(
    candidates: list[BetCandidate], flags: list[DeterministicFlag]
) -> list[BetCandidate]:
    """
    Drop candidates carrying a drop=True flag (logged) and attach the
    remaining flags to `candidate.extra["flags"]` so they print, reach
    Discord, and feed the validator payload.
    """
    if not flags:
        return candidates
    by_player: dict[str, list[DeterministicFlag]] = {}
    for f in flags:
        by_player.setdefault(normalize_player(f.player), []).append(f)

    kept: list[BetCandidate] = []
    for cand in candidates:
        mine = by_player.get(normalize_player(cand.player), [])
        drops = [f for f in mine if f.drop]
        if drops:
            logger.info("Dropping %s %s %s — %s", cand.player, cand.market,
                        cand.pick_side, drops[0].detail)
            continue
        if mine:
            cand.extra["flags"] = [f.model_dump(exclude_none=True) for f in mine]
            cand.agent_reasons = list(cand.agent_reasons or []) + [f.detail for f in mine]
        kept.append(cand)
    return kept
