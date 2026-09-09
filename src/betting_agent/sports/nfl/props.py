"""
NFL player props: data access and distributional projection models.

Markets modeled (Phase 3 MVP): receptions and receiving yards.

Receptions are counts with variance above the mean, so they get a negative
binomial whose dispersion is fit per position from history. Receiving yards
are non-negative and heavy-tailed, so they get a shifted lognormal whose
residual location/scale vary with projection size (low-volume players carry
fatter zero-side tails). Both center on an exponentially-weighted
average of the player's recent games, shrunk toward the position mean
(few games = mostly prior), scaled by how the opponent's defense treats
that position relative to league average.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import defaultdict
from dataclasses import dataclass

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

RECEIVING_POSITIONS = ("WR", "TE", "RB", "FB")
RUSHING_POSITIONS = ("RB", "FB", "QB", "WR")

# Odds API market key → player-stats column (grading supports more markets
# than the models project). The `_alternate` keys are the books' ladder
# boards — the same stat at other numbers — and grade off the same column.
MARKET_STAT_COLUMNS: dict[str, str] = {
    "player_receptions": "receptions",
    "player_receptions_alternate": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_reception_yds_alternate": "receiving_yards",
    "player_rush_yds": "rushing_yards",
    "player_rush_yds_alternate": "rushing_yards",
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_anytime_td": "anytime_tds",   # derived: see td_props.add_anytime_td_column
}

#: Markets the main card prices (both sides, floors from props_diagnostic.py).
MODELED_MARKETS = ("player_receptions", "player_reception_yds")
#: Markets ReceivingPropsModel can build a distribution for. Rushing yards is
#: projected for the ladder only — it has no main-card diagnostic.
DISTRIBUTION_MARKETS = ("player_receptions", "player_reception_yds", "player_rush_yds")
ALTERNATE_MARKETS = {m: f"{m}_alternate" for m in DISTRIBUTION_MARKETS}


def base_market(key: str | None) -> str:
    """'player_reception_yds_alternate' → 'player_reception_yds'."""
    return (key or "").removesuffix("_alternate")


def is_alternate_market(key: str | None) -> bool:
    return bool(key) and key.endswith("_alternate")

# Sentinel returned by stat lookups when the week's stats ARE published but
# the player has no row — a DNP. Distinct from None (stats not yet available),
# so the grader can void the pick instead of leaving it pending forever.
DNP = "DNP"

# Pseudo-line offsets around a projected mean — the neighbourhood real books
# quote in. Used to tune and to validate probability calibration.
RECEPTION_OFFSETS = (-1.5, -0.5, 0.5, 1.5)
YARDS_MULTIPLIERS = (0.75, 0.9, 1.1, 1.25)


def pseudo_lines(market: str, mean: float) -> list[float]:
    if market == "player_receptions":
        return [round(mean) + off for off in RECEPTION_OFFSETS if round(mean) + off > 0]
    lines = [round(mean * m * 2) / 2 for m in YARDS_MULTIPLIERS]
    return [ln + 0.5 if ln == int(ln) else ln for ln in lines if ln > 0]


#: Per-market edge floors, set from the walk-forward pick diagnostic (2020-23
#: train, 2024-25 eval). Realized hit rate rises monotonically with the floor
#: in both markets, and receiving yards runs ~5pp worse than receptions at
#: every floor with a wider overconfidence gap, so it carries the stricter
#: floor. Both sit above the ~4.6-6.0pp residual overconfidence measured after
#: recalibration, so the floor is real cushion rather than drift.
PROP_EDGE_FLOORS = {
    "player_receptions": 0.10,
    "player_reception_yds": 0.15,
    "player_anytime_td": 0.08,
}
DEFAULT_EDGE_FLOOR = 0.10

#: Per-market edge CAPS (exclusive). scripts/td_props_diagnostic.py (2024-25
#: walk-forward vs a usage-aware proxy board, long shots excluded): anytime-TD
#: picks in the 8-15% window hit 30.4% vs 34.4% claimed (n=655, +10% flat ROI
#: at the measured 25% board hold, the same in both seasons) but realised
#: probability FALLS above 15% (claimed 38%, realised 24%) — when the model
#: disagrees with the market that strongly, the model is the one that is
#: wrong. The receiving markets show no such turnover and carry no cap.
PROP_EDGE_CAPS = {
    "player_anytime_td": 0.15,
}


def edge_floor(market: str, override: float | None = None) -> float:
    """Edge a prop must clear. `override` (CLI --min-edge) wins when given."""
    if override is not None:
        return override
    return PROP_EDGE_FLOORS.get(market, DEFAULT_EDGE_FLOOR)


def edge_cap(market: str) -> float | None:
    """Edge at or above which a prop is NOT bet (None = no cap)."""
    return PROP_EDGE_CAPS.get(market)


# ---- Ladder hits (scripts/props.py generate_ladder_candidates) ----
#
# The "best overs" section: the player reaching a milestone — 60+ receiving
# yards, 6+ receptions, 40+ rushing — priced on the books' alternate boards
# (Over-only ladders of 10-18 rungs per player) plus the main-line Over.
# Policy is set by scripts/ladder_diagnostic.py; re-run it after any change
# to the distributions or the tail calibrator.

#: Rungs the tail calibrator trains on and the diagnostic evaluates — the
#: numbers DraftKings/FanDuel actually hang.
LADDER_RUNGS: dict[str, tuple[float, ...]] = {
    "player_receptions": tuple(x + 0.5 for x in range(1, 12)),
    "player_reception_yds": (9.5, 14.5, 19.5, 24.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5,
                             89.5, 99.5, 109.5, 124.5, 149.5),
    "player_rush_yds": (9.5, 14.5, 19.5, 24.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5,
                        89.5, 99.5, 109.5, 124.5, 149.5),
}
#: A rung must be a milestone: at least this number. Below it the "edge" the
#: diagnostic finds is on 10+/15+ yard rungs for low-usage players — depth
#: chart guesses, not milestones, and not numbers books hang anyway.
LADDER_MIN_RUNG = {
    "player_receptions": 4.5,
    "player_reception_yds": 39.5,
    "player_rush_yds": 39.5,
}
#: Edge a ladder rung must clear to be a pick, by base market. The ladder
#: is an EXPERIMENTAL pick pool by the user's decision (Sep 8 2026: "I want
#: the ladders and the potential overs to be picks in their own separate
#: pool. I don't care if overall long term they lose money, I just want to
#: experiment with the models and see if they somehow pick good overs or
#: ladders like the main picks for the unders") — so the floors are set
#: where the model starts to disagree with the book at all (the main card's
#: MIN_EDGE tier), not where ladder_diagnostic.py says the edge is real.
#: For the record, that diagnostic (2024-25 walk-forward on the ladder
#: projection, milestone rungs, fair 20-65%, best rung per player, vs a
#: trailing-mean proxy book) found: rushing yards calibrated from ~5% up
#: (44.2% hit vs 43.5% claimed, +31% flat ROI at a 10% hold); receiving
#: yards over-claim ~6pp at 10% (44.1% vs 50.1%, +17%) and ~3pp at 12%;
#: receptions over-claim 10-20pp at every floor. Expect the pool to run
#: under its claimed probabilities; the point is to see by how much.
LADDER_EDGE_FLOORS = {
    "player_receptions": 0.03,
    "player_reception_yds": 0.03,
    "player_rush_yds": 0.03,
}
#: Edge at or above which a rung is NOT bet: realised probability turns
#: over above 15% claimed edge for rushing yards (claimed 41%, realised 35%
#: at 15-20%; 44% vs 33% at 20-30%) — the model, not the book, is wrong
#: there. Receiving yards show no turnover through 25%.
LADDER_EDGE_CAPS = {
    "player_receptions": 0.15,
    "player_reception_yds": 0.25,
    "player_rush_yds": 0.15,
}
#: Fair-probability window for a rung. Below the floor is long-shot country
#: (favourite-longshot bias lives in the book's hold there, which a uniform
#: de-vig cannot see); above the cap it is a chalky main-line over, not a
#: milestone.
LADDER_MIN_FAIR_PROB = 0.20
LADDER_MAX_FAIR_PROB = 0.65
#: Hold applied to an Over-only rung when the player has no main-line pair
#: to measure the book's hold from (DK/FD price main lines at -110/-114).
DEFAULT_LADDER_HOLD = 0.05


def ladder_edge_floor(market: str, override: float | None = None) -> float:
    if override is not None:
        return override
    return LADDER_EDGE_FLOORS.get(base_market(market), DEFAULT_EDGE_FLOOR)


def ladder_edge_cap(market: str) -> float:
    return LADDER_EDGE_CAPS.get(base_market(market), 0.15)


def ladder_min_rung(market: str) -> float:
    return LADDER_MIN_RUNG.get(base_market(market), 0.0)


LADDER_STAT_LABELS = {
    "player_receptions": "receptions",
    "player_reception_yds": "receiving yds",
    "player_rush_yds": "rushing yds",
}

#: Straight overs (user, Sep 8 2026: "there should be at least one straight
#: over line to add in these primetime games"): the book's main-line Over on
#: the receiving markets, priced by the ladder projection, own paper book
#: (Pick.strategy = "overs"). The game's best over is ALWAYS a pick — when
#: the model has no edge on any over it is staked flat at OVERS_MIN_STAKE_PCT
#: of the overs bankroll so the selection is still tracked; further overs in
#: the game need OVERS_EDGE_FLOOR. Same experiment as the ladder: the main
#: projection is built to find unders and sits under the book on most
#: starters, so expect many flat-stake entries.
OVERS_EDGE_FLOOR = 0.03
OVERS_MIN_STAKE_PCT = 0.01


def over_label(player: str | None, market: str | None, line: float | None) -> str:
    """'Romeo Doubs over 36.5 receiving yds' — a straight main-line over."""
    stat = LADDER_STAT_LABELS.get(base_market(market), _market_label_text(market))
    if line is None:
        return f"{player or '?'} over {stat}"
    return f"{player or '?'} over {float(line):g} {stat}"


def ladder_label(player: str | None, market: str | None, line: float | None) -> str:
    """'A.J. Brown 60+ receiving yds' — the milestone the rung pays on."""
    stat = LADDER_STAT_LABELS.get(base_market(market), _market_label_text(market))
    if line is None:
        return f"{player or '?'} {stat}"
    return f"{player or '?'} {int(np.floor(float(line))) + 1}+ {stat}"


def _market_label_text(market: str | None) -> str:
    return base_market(market).removeprefix("player_").replace("_", " ")


#: Smallest line worth treating as quotable — books don't hang numbers below
#: these, so lines under them are noise for both calibration and betting.
MIN_QUOTABLE_LINE = {"player_receptions": 1.5, "player_reception_yds": 10.0,
                     "player_rush_yds": 10.0}


def book_proxy_line(prior_values: list[float], market: str) -> float | None:
    """
    Approximate the number a book would hang: the player's trailing median,
    quoted to a half point.

    Books price close to recent central tendency, which sits systematically
    ABOVE our shrunk projection. Calibrating only on offsets around our own
    mean therefore trains the probability map in a region real lines rarely
    occupy — this gives us the realistic region as well.
    """
    if not prior_values:
        return None
    med = float(np.median(prior_values[-8:]))
    if market == "player_receptions":
        line = float(int(med)) + 0.5
    else:
        line = round(med * 2) / 2
        if line == int(line):
            line += 0.5
    return line if line >= MIN_QUOTABLE_LINE.get(market, 0.0) else None

_SUFFIXES = re.compile(r"\s+(jr|sr|ii|iii|iv|v)\.?$")


def normalize_player(name: str | None) -> str:
    """Canonical form for matching Odds API names to nflreadpy names."""
    if not name:
        return ""
    s = str(name).strip().lower()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIXES.sub("", s)
    return re.sub(r"\s+", " ", s)


def load_player_stats(seasons: list[int]) -> pd.DataFrame:
    """
    Load weekly player stats from nflreadpy. Seasons are fetched one at a
    time so a season that has no data yet (e.g. the upcoming one before
    kickoff) doesn't sink the whole load.
    """
    frames = []
    for season in seasons:
        try:
            frames.append(nfl.load_player_stats([season]).to_pandas())
        except Exception as exc:
            logger.warning("No player stats for %s: %s", season, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_stat_history(stats: pd.DataFrame, positions: tuple[str, ...],
                       stat_cols: tuple[str, ...]) -> pd.DataFrame:
    """
    Filter weekly stats to `positions` and the columns a model needs,
    ordered by time. Adds `t` — a global game-order index used for
    "everything before week X" splits.
    """
    if stats.empty:
        return stats
    df = stats[stats["position"].isin(positions)].copy()
    keep = [
        "player_id", "player_display_name", "position", "season", "week",
        "season_type", "team", "opponent_team", *stat_cols,
    ]
    df = df[[c for c in keep if c in df.columns]]
    df["player_key"] = df["player_display_name"].map(normalize_player)
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["t"] = df["season"] * 100 + df["week"]
    return df


def build_receiving_history(stats: pd.DataFrame) -> pd.DataFrame:
    """Pass-catchers' receptions / targets / receiving yards by game."""
    return build_stat_history(stats, RECEIVING_POSITIONS,
                              ("receptions", "targets", "receiving_yards"))


def build_rushing_history(stats: pd.DataFrame) -> pd.DataFrame:
    """Ball-carriers' (backs and quarterbacks, plus receivers who get
    carries) rushing yards / carries by game — the ladder's rushing model."""
    return build_stat_history(stats, RUSHING_POSITIONS, ("rushing_yards", "carries"))


#: How many distinct slates back a player must have appeared to count as
#: someone a book would hang a line on.
ACTIVE_WINDOW_SLATES = 3


def active_player_keys(history: pd.DataFrame, asof_t: int,
                       window: int = ACTIVE_WINDOW_SLATES) -> set[str]:
    """
    Players seen in the last `window` distinct slates strictly before `asof_t`.

    `t = season*100 + week` is not contiguous across seasons, so the window
    is taken by RANK of the distinct `t` values in history, never by
    subtraction: `asof_t - 3` in week 1 of a new season names a `t` no game
    has and silently excludes every player.

    Postseason slates carry only the surviving teams, so when the history
    tags `season_type` the window counts regular-season slates only —
    otherwise week 1 would look back at the divisional round through the
    Super Bowl and find two teams' worth of players.
    """
    if history is None or history.empty or "t" not in history.columns:
        return set()
    if "season_type" in history.columns:
        regular = history[history["season_type"] == "REG"]
        if not regular.empty:
            history = regular
    past_ts = sorted(t for t in history["t"].unique() if t < asof_t)
    if not past_ts:
        return set()
    recent = set(past_ts[-window:])
    rows = history[history["t"].isin(recent)]
    return set(rows["player_key"].unique())


def current_teams(season: int, positions: tuple[str, ...] = RECEIVING_POSITIONS,
                  ) -> dict[str, str]:
    """
    Player key → current team from the published roster, for `positions`
    (default pass-catchers) on active status. Stats rows only know the team a player LAST PLAYED for,
    so offseason movers would otherwise be attributed to the wrong side.
    Returns {} on any failure — callers fall back to the stats-derived team.
    """
    try:
        roster = nfl.load_rosters([season]).to_pandas()
    except Exception as exc:
        logger.warning("No roster for %s (using last-played teams): %s", season, exc)
        return {}
    if roster.empty or not {"status", "position", "full_name", "team"} <= set(roster.columns):
        return {}
    active = roster[
        (roster["status"] == "ACT") & (roster["position"].isin(positions))
    ]
    return {
        normalize_player(name): str(team)
        for name, team in active[["full_name", "team"]].itertuples(index=False)
        if name and pd.notna(team)
    }


def load_season_schedule(season: int) -> pd.DataFrame:
    """Normalised nflreadpy schedule for one season; empty frame on failure."""
    from betting_agent.sports.nfl.features import normalise_raw_schedules
    from betting_agent.sports.nfl.loader import NFLLoader

    try:
        sched = normalise_raw_schedules(NFLLoader().load_schedules([season]).to_pandas())
    except Exception as exc:
        logger.warning("Could not load %s schedule: %s", season, exc)
        return pd.DataFrame()
    if not sched.empty and "game_date" in sched.columns:
        sched = sched.assign(game_date=pd.to_datetime(sched["game_date"]))
    return sched


#: A schedule row must land this close to the requested date to match.
SCHEDULE_DATE_TOLERANCE = pd.Timedelta(days=2)


def schedule_row_for(
    schedule: pd.DataFrame,
    game_date,
    home: str | None = None,
    away: str | None = None,
) -> pd.Series | None:
    """
    Nearest schedule row to `game_date` (within two days), optionally
    restricted to a matchup given as nflreadpy abbreviations.
    """
    if schedule is None or schedule.empty or "game_date" not in schedule.columns:
        return None
    rows = schedule
    if home is not None and away is not None:
        rows = rows[(rows["home_team"] == home) & (rows["away_team"] == away)]
    if rows.empty:
        return None
    stamp = pd.Timestamp(game_date)
    gaps = (pd.to_datetime(rows["game_date"]) - stamp).abs()
    idx = gaps.idxmin()
    if gaps.loc[idx] > SCHEDULE_DATE_TOLERANCE:
        return None
    return rows.loc[idx]


def approximate_nfl_week(event_date, season: int) -> int:
    """Date-only fallback for the NFL week, used when no schedule is available."""
    from datetime import date as _date

    season_start = _date(season, 9, 1)
    days = (pd.Timestamp(event_date).date() - season_start).days
    return max(1, min(22, days // 7 + 1))


def nfl_week_for(
    event_date,
    season: int,
    schedule: pd.DataFrame | None,
    home: str | None = None,
    away: str | None = None,
) -> int:
    """
    NFL week for a game, from the published schedule when possible (exact
    matchup first, then any game within two days), else the date heuristic.
    """
    if schedule is not None and not schedule.empty and "week" in schedule.columns:
        row = schedule_row_for(schedule, event_date, home, away)
        if row is None and home is not None:
            row = schedule_row_for(schedule, event_date)
        if row is not None and pd.notna(row["week"]):
            return int(row["week"])
    return approximate_nfl_week(event_date, season)


def prop_bookmaker_order() -> list[str]:
    """
    Books to request and, in this order, to price against: the preferred
    book(s) first (default bet365, the book actually bet at), then the
    fallbacks. The Odds API does not carry bet365 player props (Sep 2026:
    none in any region on the opener while DraftKings/FanDuel/Pinnacle all
    posted), so fetching only the preferred book returns an empty response
    and no pick is ever produced. Up to 10 books cost the same one credit
    per market, so requesting them all is free; `books_in_preference`
    then decides which book's prices a game is priced against.
    """
    from betting_agent.config import settings

    order = list(settings.preferred_bookmaker_list) or ["bet365"]
    for key in settings.prop_fallback_bookmaker_list:
        if key not in order:
            order.append(key)
    return order


def _book_has_markets(book: dict) -> bool:
    return any(m.get("outcomes") for m in book.get("markets", []))


def books_in_preference(event: dict, order: list[str] | None) -> list[dict]:
    """
    The bookmaker entries of one per-event odds response to price against:
    only the first book in `order` that actually posted markets, so a game
    is never priced best-of-N across books (that inflates edges — the
    floors were set against a single book). Falls back to every book in
    the response when none of the ordered keys posted.
    """
    books = [b for b in event.get("bookmakers", []) if _book_has_markets(b)]
    for key in order or []:
        chosen = [b for b in books if b.get("key") == key]
        if chosen:
            return chosen
    return books


def pair_outcomes(market: dict) -> dict[tuple[str, float], dict[str, dict]]:
    """Group a market's outcomes into {(player, line): {"Over": o, "Under": o}}."""
    pairs: dict[tuple[str, float], dict[str, dict]] = {}
    for outcome in market.get("outcomes", []):
        player = outcome.get("description")
        point = outcome.get("point")
        name = outcome.get("name")
        if player is None or point is None or name not in ("Over", "Under"):
            continue
        pairs.setdefault((player, float(point)), {})[name] = outcome
    return pairs


YARDS_SHIFT = 5.0   # lognormal shift so zero-yard games stay in support


class _ShiftedDist:
    """A scipy frozen distribution on (X + shift), exposed on X's scale."""

    def __init__(self, dist, shift: float):
        self._dist = dist
        self._shift = shift

    def sf(self, x):
        return self._dist.sf(np.asarray(x, dtype=float) + self._shift)

    def cdf(self, x):
        return self._dist.cdf(np.asarray(x, dtype=float) + self._shift)

    def ppf(self, q):
        return self._dist.ppf(q) - self._shift


def calibrated_prob(calibrator, raw: float) -> float:
    """
    Apply an isotonic raw-P(over) → empirical map only where it is trustworthy.

    The calibrators are fit on lines near the projection (pseudo_lines and the
    book-proxy median), so their fitted range is narrow — receiving yards sees
    raw P(over) in roughly [0.11, 0.73]. sklearn snaps anything outside that
    range to the end bins, and an isotonic end bin is exactly 0 or 1 whenever
    its few most extreme samples all missed or all hit. A real book line far
    from the projection (11.5 on a player the model has at 30, or at 6) lands
    there, and the model then claimed 99% — a bigger Kelly stake precisely
    where it disagrees with the market hardest. Outside the fitted range, or
    in a collapsed end bin, the distribution's own tail is the honest number.
    """
    raw = float(raw)
    lo = getattr(calibrator, "X_min_", None)
    hi = getattr(calibrator, "X_max_", None)
    if lo is not None and hi is not None and not (lo <= raw <= hi):
        return raw
    cal = float(calibrator.predict([raw])[0])
    if cal <= 0.0 or cal >= 1.0:
        return raw
    return cal


@dataclass
class Projection:
    player: str
    market: str
    mean: float
    games: int              # player games the mean is built on
    _dist: object           # frozen scipy distribution
    _calibrator: object = None   # isotonic map raw P(over) → empirical
    _tail_calibrator: object = None   # same, trained on the ladder rungs

    def _raw_over(self, line: float) -> float:
        if self.market == "player_receptions":
            # Counts: P(X > line). For half-lines this is P(X >= ceil(line)).
            return float(self._dist.sf(np.floor(line)))
        return float(self._dist.sf(line))

    def _push_mass(self, line: float) -> float:
        if self.market == "player_receptions" and float(line) == int(line):
            return float(self._dist.pmf(int(line)))
        return 0.0

    def prob_over(self, line: float) -> float:
        p = self._raw_over(line)
        if self._calibrator is not None:
            p = calibrated_prob(self._calibrator, p)
        return float(np.clip(p, 0.01, 0.99))

    def prob_under(self, line: float) -> float:
        return max(0.0, 1.0 - self.prob_over(line) - self._push_mass(line))

    def prob_hit(self, line: float) -> float:
        """
        P(stat > line) for a ladder rung. The main calibrator only saw
        pseudo-lines near the projection and the book's main number, so it
        clips in the tails; the tail calibrator was fit on the rungs the
        books hang (LADDER_RUNGS) and covers the whole range.
        """
        p = self._raw_over(line)
        cal = self._tail_calibrator if self._tail_calibrator is not None else self._calibrator
        if cal is not None:
            p = calibrated_prob(cal, p)
        return float(np.clip(p, 0.01, 0.99))


class ReceivingPropsModel:
    """
    Projection model for one receiving market.

    fit() learns global shape parameters (dispersion / CV per position) and
    keeps the history; project() uses only games strictly before `asof`
    for the player mean and defense factor, so walk-forward evaluation is
    leak-free for first moments. Shape parameters are second moments fit on
    whatever history fit() was given — pass only training seasons to fit()
    and add later rows via extend_history() when evaluating.
    """

    HALFLIFE = 6.0          # games; recency weighting for the player mean
    PRIOR_GAMES = 4.0       # pseudo-games of shrinkage toward position mean
    #: The ladder's projection shrinks far less. Four pseudo-games toward the
    #: position mean pull a WR1 with nine effective games ~17 yards under his
    #: own average — books hang the milestone boards on exactly those
    #: players, so the main projection can never see an over there. The
    #: tail calibrator is fit on this projection, not the main one.
    LADDER_PRIOR_GAMES = 1.0
    MIN_GAMES = 4           # fewer prior games than this → no projection
    DEF_WINDOW = 8          # defensive games in the opponent factor
    DEF_CLIP = (0.8, 1.2)

    def __init__(self, market: str):
        if market not in DISTRIBUTION_MARKETS:
            raise ValueError(f"No projection model for market '{market}'")
        self.market = market
        self.stat_col = MARKET_STAT_COLUMNS[market]
        self.history: pd.DataFrame | None = None
        self.dispersion: dict[str, float] = {}   # position → NB overdispersion alpha
        self.dispersion_scale = 1.0              # set by tune_dispersion()
        # Yards: lognormal on (yards + YARDS_SHIFT). Residual location/scale
        # in log space vary with projection size — low-volume players have a
        # heavier zero-side tail than the WR1s books actually quote — so they
        # are stored per projected-mean level and interpolated. Defaults are
        # NFL-wide empirical values, replaced by tune_dispersion().
        self.yards_levels = np.array([8.0, 17.0, 27.0, 50.0])
        self.yards_mu = np.array([-0.41, -0.50, -0.48, -0.17])
        self.yards_sigma = np.array([0.72, 0.83, 0.89, 0.76])
        self.position_means: dict[str, float] = {}
        self.prob_calibrator = None   # isotonic raw P(over) → empirical, from tuning
        self.tail_calibrator = None   # same, on the ladder rungs (see prob_hit)

    # ---- fitting ----

    def fit(self, history: pd.DataFrame) -> "ReceivingPropsModel":
        self.history = history.copy()
        col = self.stat_col
        for pos, grp in history.groupby("position"):
            self.position_means[pos] = float(grp[col].mean())
            per_player = grp.groupby("player_id")[col].agg(["mean", "var", "count"])
            per_player = per_player[(per_player["count"] >= 6) & (per_player["mean"] > 0)]
            if per_player.empty:
                self.dispersion[pos] = 0.5
                continue
            # NB overdispersion: var = mean + alpha * mean^2 (receptions only;
            # yards use the lognormal residual parameters instead).
            alpha = (per_player["var"] - per_player["mean"]) / per_player["mean"] ** 2
            self.dispersion[pos] = float(np.clip(alpha.median(), 0.01, 2.0))
        return self

    def tune_dispersion(self, seasons: list[int], sample: int = 2000) -> float:
        """
        Fit the spread AROUND OUR PROJECTION from walk-forward residuals on
        the given (training) seasons. The raw within-player variance from
        fit() measures marginal spread, which is the wrong quantity once the
        mean is conditioned on recent form.

        Receptions: scales the NB overdispersion to maximise held-out
        log-likelihood. Yards: sets the lognormal residual location/scale
        directly, then trims the scale for central-interval coverage.
        """
        if self.history is None:
            raise RuntimeError("fit() before tune_dispersion()")
        rows = self.history[self.history["season"].isin(seasons)]
        if len(rows) > sample:
            rows = rows.sample(sample, random_state=0)

        # Per-player prior-game series, so each cached row can carry the line a
        # book would have hung on it.
        series: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
        ordered = self.history.sort_values("t")
        for pk, t, val in ordered[["player_key", "t", self.stat_col]].itertuples(index=False):
            ts, vs = series[pk]
            ts.append(int(t))
            vs.append(max(0.0, float(val)) if pd.notna(val) else 0.0)

        # Project once per row; retuning only changes the distribution shape,
        # so cache (mean, position, actual, book_line) and rebuild
        # distributions per k.
        cached = []
        for _, g in rows.iterrows():
            proj = self.project(g["player_key"], int(g["season"]), int(g["week"]))
            if proj is not None:
                actual = max(0.0, float(g[self.stat_col]))
                ts, vs = series[g["player_key"]]
                cut = bisect.bisect_left(ts, int(g["season"]) * 100 + int(g["week"]))
                book_line = book_proxy_line(vs[:cut], self.market)
                cached.append((proj.mean, g["position"], actual, book_line))
        if len(cached) < 200:
            logger.warning("tune_dispersion: only %d usable rows — keeping defaults",
                           len(cached))
            return self.dispersion_scale

        if self.market != "player_receptions":
            df = pd.DataFrame(
                [(mean, np.log(actual + YARDS_SHIFT) - np.log(mean + YARDS_SHIFT))
                 for mean, _, actual, _ in cached],
                columns=["mean", "res"],
            )
            df["bin"] = pd.qcut(df["mean"], 4, duplicates="drop")
            grouped = df.groupby("bin", observed=True)
            self.yards_levels = grouped["mean"].mean().to_numpy()
            self.yards_mu = grouped["res"].mean().to_numpy()
            self.yards_sigma = grouped["res"].std().to_numpy()

        def score(k: float) -> float:
            if self.market == "player_receptions":
                # Proper score: mean NB log-pmf of the actual counts.
                ll = 0.0
                for mean, position, actual, _ in cached:
                    ll += self._make_dist(mean, position, k).logpmf(round(actual))
                return -ll
            in50 = in80 = 0
            for mean, position, actual, _ in cached:
                dist = self._make_dist(mean, position, k)
                if dist.ppf(0.25) <= actual <= dist.ppf(0.75):
                    in50 += 1
                if dist.ppf(0.10) <= actual <= dist.ppf(0.90):
                    in80 += 1
            n = len(cached)
            return abs(in50 / n - 0.5) + abs(in80 / n - 0.8)

        grid = (0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.8)
        self.dispersion_scale = min(grid, key=score)
        logger.info("%s dispersion scale tuned to %.2f", self.market, self.dispersion_scale)

        # Final layer: isotonic map from the distribution's raw P(over) to the
        # empirical over-rate at pseudo-lines. Absorbs family-shape misfit
        # (e.g. the dud-game/normal-game bimodality of receiving yards) that
        # no location/scale tuning can express.
        from sklearn.isotonic import IsotonicRegression

        raw_p, hits = [], []
        for mean, position, actual, book_line in cached:
            dist = self._make_dist(mean, position, self.dispersion_scale)
            proj = Projection(player="", market=self.market, mean=mean,
                              games=0, _dist=dist)
            lines = list(pseudo_lines(self.market, mean))
            # The book-proxy line is where we actually bet, so it must be in
            # the calibration sample; weight it up so the fit is anchored
            # there rather than dominated by the offsets around our own mean.
            if book_line is not None:
                lines.extend([book_line] * len(lines))
            for line in lines:
                raw_p.append(proj._raw_over(line))
                hits.append(float(actual > line))
        self.prob_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(raw_p, hits)

        # Ladder rungs: the same kind of map for the ladder's own projection
        # (LADDER_PRIOR_GAMES), fit where the alternate boards live — well
        # into the tails the main calibrator never sees and would clip.
        raw_t, hits_t = [], []
        for _, g in rows.iterrows():
            proj = self.project_ladder(g["player_key"], int(g["season"]), int(g["week"]))
            if proj is None:
                continue
            actual = max(0.0, float(g[self.stat_col]))
            for rung in LADDER_RUNGS.get(self.market, ()):
                p = proj._raw_over(rung)
                if 0.005 <= p <= 0.995:
                    raw_t.append(p)
                    hits_t.append(float(actual > rung))
        if len(raw_t) >= 200:
            self.tail_calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            ).fit(raw_t, hits_t)
        return self.dispersion_scale

    def extend_history(self, later_rows: pd.DataFrame) -> None:
        """Add rows for walk-forward evaluation without refitting shapes."""
        self.history = (
            pd.concat([self.history, later_rows], ignore_index=True)
            .drop_duplicates(subset=["player_id", "season", "week"], keep="first")
            .sort_values("t")
            .reset_index(drop=True)
        )

    # ---- projecting ----

    def _make_dist(self, mean: float, position: str, scale: float):
        if self.market == "player_receptions":
            # Guard: NB needs var > mean, i.e. alpha > 0.
            alpha = max(self.dispersion.get(position, 0.5) * scale, 1e-3)
            r = 1.0 / alpha
            return sps.nbinom(r, r / (r + mean))
        # Yards: lognormal on (Y + shift), located by the walk-forward
        # residual mean at this projection level so the median sits where
        # actuals do.
        mu = float(np.interp(mean, self.yards_levels, self.yards_mu))
        sigma = float(np.interp(mean, self.yards_levels, self.yards_sigma))
        sigma = max(sigma * scale, 0.05)
        log_scale = np.exp(np.log(mean + YARDS_SHIFT) + mu)
        return _ShiftedDist(sps.lognorm(sigma, scale=log_scale), YARDS_SHIFT)

    def _defense_factor(self, opponent: str, position: str, before_t: int) -> float:
        """How the opponent treats this position vs league average."""
        h = self.history
        col = self.stat_col
        pos_rows = h[(h["position"] == position) & (h["t"] < before_t)]
        if pos_rows.empty:
            return 1.0
        league_pg = pos_rows.groupby(["opponent_team", "t"])[col].sum().mean()
        faced = pos_rows[pos_rows["opponent_team"] == opponent]
        if faced.empty or league_pg <= 0:
            return 1.0
        recent = (
            faced.groupby("t")[col].sum()
            .sort_index()
            .tail(self.DEF_WINDOW)
        )
        if recent.empty:
            return 1.0
        return float(np.clip(recent.mean() / league_pg, *self.DEF_CLIP))

    def project_ladder(
        self,
        player_key: str,
        asof_season: int,
        asof_week: int,
        opponent: str | None = None,
    ) -> Projection | None:
        """The ladder's projection: lighter shrinkage, tail calibrator attached."""
        return self.project(player_key, asof_season, asof_week, opponent=opponent,
                            prior_games=self.LADDER_PRIOR_GAMES)

    def project(
        self,
        player_key: str,
        asof_season: int,
        asof_week: int,
        opponent: str | None = None,
        prior_games: float | None = None,
    ) -> Projection | None:
        """
        Distribution for the player's stat in (asof_season, asof_week),
        using only games strictly before it. `prior_games` overrides the
        shrinkage (the ladder passes LADDER_PRIOR_GAMES via project_ladder);
        only such projections carry the tail calibrator — it was fit on them.
        """
        if self.history is None:
            raise RuntimeError("fit() before project()")
        asof_t = asof_season * 100 + asof_week
        past = self.history[
            (self.history["player_key"] == player_key)
            & (self.history["t"] < asof_t)
        ].sort_values("t")
        if len(past) < self.MIN_GAMES:
            return None

        values = past[self.stat_col].to_numpy(dtype=float)
        weights = 0.5 ** (np.arange(len(values))[::-1] / self.HALFLIFE)
        ew_mean = float(np.average(values, weights=weights))
        n_eff = float(weights.sum())

        position = past["position"].iloc[-1]
        pos_mean = self.position_means.get(position, float(np.mean(values)))
        prior = self.PRIOR_GAMES if prior_games is None else prior_games
        mean = (n_eff * ew_mean + prior * pos_mean) / (n_eff + prior)

        if opponent:
            mean *= self._defense_factor(opponent, position, asof_t)
        mean = max(mean, 0.1)

        dist = self._make_dist(mean, position, self.dispersion_scale)

        return Projection(
            player=player_key, market=self.market, mean=mean,
            games=len(past), _dist=dist, _calibrator=self.prob_calibrator,
            _tail_calibrator=self.tail_calibrator if prior_games is not None else None,
        )


# ---- grading support ----

def make_stat_lookup(seasons: list[int]):
    """
    Build a stat_lookup(player, market, game) callable for
    grade_prop_picks(). Resolves the game's NFL week from the schedule by
    (home, away, nearest date), then finds the player's row for that week.

    Returns the stat as float; None when the week's stats aren't published
    yet (leave ungraded); or the DNP sentinel when stats for the game's
    teams ARE published but the player has no row — the pick voids, matching
    how books settle a prop on a player who didn't play. A name that never
    matches nflreadpy's spelling voids the same way, so check the void log
    line if a star player's pick voids unexpectedly.
    """
    stats = load_player_stats(seasons)
    if stats.empty:
        return lambda player, market, game: None
    from betting_agent.sports.nfl.td_props import add_anytime_td_column

    stats = add_anytime_td_column(stats)
    stats["player_key"] = stats["player_display_name"].map(normalize_player)

    schedules = pd.concat(
        [load_season_schedule(s) for s in seasons], ignore_index=True
    ) if seasons else pd.DataFrame()

    def lookup(player: str | None, market: str | None, game) -> float | None:
        col = MARKET_STAT_COLUMNS.get(market or "")
        if col is None or not player:
            return None
        row = schedule_row_for(schedules, game.game_date, game.home_team, game.away_team)
        if row is None or "week" not in row.index:
            return None
        season, week = int(row["season"]), int(row["week"])

        key = normalize_player(player)
        week_stats = stats[
            (stats["season"] == season)
            & (stats["week"] == week)
            & (stats["team"].isin([game.home_team, game.away_team]))
        ]
        if week_stats.empty or col not in stats.columns:
            return None  # stats for this game not published yet
        hits = week_stats[week_stats["player_key"] == key]
        if hits.empty:
            return DNP  # week is published, player absent — void
        return float(hits.iloc[0][col])

    return lookup


# ---- odds fetching ----

def fetch_prop_odds(
    sport_key: str = "americanfootball_nfl",
    markets: list[str] | None = None,
    bookmakers: list[str] | None = None,
    max_events: int | None = None,
    events: list[dict] | None = None,
) -> list[dict]:
    """
    Fetch player prop odds. Props are only served by the per-event endpoint,
    so this costs one API call per event. Pass `events` (from
    OddsAPIClient.fetch_events, a free call) to fetch odds for a chosen
    subset instead of every upcoming event.
    """
    from betting_agent.api.odds import OddsAPIClient

    if markets is None:
        markets = list(MODELED_MARKETS)
    client = OddsAPIClient()
    if events is None:
        events = client.fetch_events(sport_key)
    if max_events:
        events = events[:max_events]

    out = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        data = client.fetch_event_odds(sport_key, event_id, markets, bookmakers)
        if data:
            out.append(data)
    return out
