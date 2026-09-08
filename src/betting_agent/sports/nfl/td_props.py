"""
Anytime-touchdown-scorer props (`player_anytime_td`).

A Yes-only market: the book posts one price per player and nothing to
de-vig it against, so two things differ from the over/under props in
`props.py`.

Fair price: the board is de-vigged as a whole. With TDs per player Poisson
at rate lambda, the Yes price implies lambda_book = -ln(1 - p_book). The
sum of those over every listed player should equal the touchdowns the
market expects in the game, which follows from the spread and total
(`expected_team_tds`: a linear fit of team rushing+receiving+return TDs on
market-implied points, 2020-2025). Scaling every lambda_book by one factor
until the board sums to that target gives the fair Yes probability per
player. The hold the scaling removes is reported alongside each pick.

Model: a player's TD rate is the exponentially weighted mean of his own
TDs per game, shrunk toward a USAGE-scaled prior (the position's TDs per
touch times his own touches per game — shrinking toward the bare position
rate hands a half-touch-a-game backup the same 10% floor as a starter,
which is exactly the player the book prices at +4000 and knows better),
moved for the scoring environment (market-implied team TDs vs the team's
trailing TDs) and the opponent (TDs allowed to the position vs league),
through a Poisson to P(>= 1), then an isotonic calibration layer fit
walk-forward. Long shots are refused regardless of edge (MIN_FAIR_PROB):
the favourite-longshot bias lives there and so do the depth-chart facts
the model cannot see. Realised P(any TD) by trailing rate
runs from 10% (no recent TDs) to 50% (the busiest scorers), so the signal
is real but noisy — hence heavy shrinkage and a strict edge floor set by
`scripts/td_props_diagnostic.py`.

Picks are stored as bet_type="prop", market="player_anytime_td",
pick_side="yes", line=0.5 (so the grader's over/under logic and the
closing-line capture work unchanged), graded on the derived
`anytime_tds` column = rushing + receiving + return + fumble-recovery TDs.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from betting_agent.intelligence.ev import american_to_implied_prob
from betting_agent.sports.nfl.props import normalize_player

logger = logging.getLogger(__name__)

TD_MARKET = "player_anytime_td"
TD_STAT_COL = "anytime_tds"
#: Positions books list on an anytime-TD board (QBs for rushing scores).
TD_POSITIONS = ("QB", "RB", "WR", "TE", "FB")
#: nflreadpy weekly-stat columns that settle an anytime-TD bet (any TD the
#: player scores himself; passing TDs do not count).
ANYTIME_TD_COLUMNS = ("rushing_tds", "receiving_tds", "special_teams_tds",
                      "fumble_recovery_tds")

#: Team non-passing TDs per game as a linear function of market-implied
#: points ((total ± spread) / 2), fit on 3,230 team-games 2020-2025.
TDS_INTERCEPT = -0.671
TDS_PER_POINT = 0.1407
LEAGUE_TEAM_TDS = 2.38          # fallback when no market line is known
#: Share of a team's TDs scored by skill-position players (the ones on the
#: board); the rest are linemen / defenders on returns.
BOARD_COVERAGE = 0.98
#: A player the de-vigged board prices below this is not bet whatever the
#: model says: long shots are where the favourite-longshot bias sits and
#: where the book's depth-chart knowledge beats a stats-only model.
MIN_FAIR_PROB = 0.10


def add_anytime_td_column(stats: pd.DataFrame) -> pd.DataFrame:
    """Add the derived `anytime_tds` column (sum of the TD columns present)."""
    if stats.empty:
        return stats
    cols = [c for c in ANYTIME_TD_COLUMNS if c in stats.columns]
    out = stats.copy()
    out[TD_STAT_COL] = out[cols].fillna(0).sum(axis=1) if cols else 0.0
    return out


def build_td_history(stats: pd.DataFrame) -> pd.DataFrame:
    """Skill-position weekly rows with `anytime_tds`, `player_key` and `t`."""
    if stats.empty:
        return stats
    df = add_anytime_td_column(stats[stats["position"].isin(TD_POSITIONS)])
    keep = ["player_id", "player_display_name", "position", "season", "week",
            "season_type", "team", "opponent_team", "carries", "targets", TD_STAT_COL]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["player_key"] = df["player_display_name"].map(normalize_player)
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["t"] = df["season"] * 100 + df["week"]
    return df


# ---- market-implied touchdowns ----

def expected_team_tds(points: float | None) -> float:
    """Expected non-passing TDs for a team expected to score `points`."""
    if points is None or not np.isfinite(points):
        return LEAGUE_TEAM_TDS
    return max(0.3, TDS_INTERCEPT + TDS_PER_POINT * float(points))


def expected_game_tds(spread_line: float | None, total_line: float | None,
                      ) -> tuple[float, float]:
    """
    (home, away) expected TDs from the market. `spread_line` follows
    nflreadpy: points the HOME team is favoured by (positive = home
    favourite). Missing lines fall back to the league average.
    """
    if total_line is None or not np.isfinite(total_line):
        return LEAGUE_TEAM_TDS, LEAGUE_TEAM_TDS
    spread = 0.0 if spread_line is None or not np.isfinite(spread_line) else float(spread_line)
    home_pts = (float(total_line) + spread) / 2.0
    away_pts = (float(total_line) - spread) / 2.0
    return expected_team_tds(home_pts), expected_team_tds(away_pts)


def team_expected_tds_lookup(schedule: pd.DataFrame | None) -> dict[tuple[int, int, str], float]:
    """{(season, week, team): expected TDs} from a normalised schedule's lines."""
    out: dict[tuple[int, int, str], float] = {}
    if schedule is None or schedule.empty:
        return out
    need = {"season", "week", "home_team", "away_team", "spread_line", "total_line"}
    if not need <= set(schedule.columns):
        return out
    for r in schedule[list(need)].itertuples(index=False):
        home, away = expected_game_tds(r.spread_line, r.total_line)
        out[(int(r.season), int(r.week), str(r.home_team))] = home
        out[(int(r.season), int(r.week), str(r.away_team))] = away
    return out


# ---- the board ----

def yes_outcomes(market: dict) -> dict[str, dict]:
    """{player: outcome} for the Yes side of an anytime-TD market."""
    out: dict[str, dict] = {}
    for outcome in market.get("outcomes", []):
        player = outcome.get("description")
        if player and outcome.get("name") == "Yes" and outcome.get("price") is not None:
            out[player] = outcome
    return out


def fair_yes_probabilities(prices: dict[str, int], expected_tds: float,
                           ) -> tuple[dict[str, float], float]:
    """
    De-vig a Yes-only board against the touchdowns the market expects.

    Returns ({player: fair P(yes)}, hold) where hold is the board's summed
    implied probability over its fair sum minus one — how much the book is
    charging. `expected_tds` should already be scaled by BOARD_COVERAGE.
    """
    if not prices or expected_tds <= 0:
        return {}, 0.0
    lam_book: dict[str, float] = {}
    for player, price in prices.items():
        p = min(max(american_to_implied_prob(price), 1e-4), 0.995)
        lam_book[player] = -np.log1p(-p)
    scale = expected_tds / sum(lam_book.values())
    fair = {pl: float(1.0 - np.exp(-lam * scale)) for pl, lam in lam_book.items()}
    hold = sum(american_to_implied_prob(pr) for pr in prices.values()) / sum(fair.values()) - 1.0
    return fair, float(hold)


# ---- the model ----

def _touches(history: pd.DataFrame) -> pd.Series:
    """Carries + targets per row (NaN when neither column exists)."""
    cols = [c for c in ("carries", "targets") if c in history.columns]
    if not cols:
        return pd.Series(np.nan, index=history.index)
    return history[cols].fillna(0).sum(axis=1).astype(float)


@dataclass
class TdProjection:
    player: str
    rate: float             # expected TDs (Poisson lambda)
    games: int
    raw_prob: float         # 1 - exp(-rate)
    prob: float             # calibrated P(any TD)


class TouchdownPropsModel:
    """
    fit() indexes the history and learns position priors; calibrate() fits
    the isotonic layer from walk-forward projections on training seasons;
    project() uses games strictly before the as-of week only.
    """

    HALFLIFE = 8.0          # games — TDs are rare, so lean on more history
    PRIOR_GAMES = 8.0       # pseudo-games of shrinkage to the usage-scaled prior
    TOUCH_PRIOR_GAMES = 2.0 # pseudo-games of shrinkage on the player's touches
    MIN_GAMES = 4
    TEAM_WINDOW = 10        # team games in the scoring-environment baseline
    DEF_WINDOW = 8
    ENV_CLIP = (0.6, 1.6)
    DEF_CLIP = (0.8, 1.2)

    def __init__(self):
        self.history: pd.DataFrame | None = None
        self.position_rates: dict[str, float] = {}
        self.position_touches: dict[str, float] = {}      # touches per game
        self.position_td_per_touch: dict[str, float] = {}
        self.league_pos_pg: dict[str, float] = {}
        self.calibrator = None
        # player_key → (t, tds, touches, teams, position)
        self._players: dict[str, tuple[list[int], np.ndarray, np.ndarray, list[str], str]] = {}
        self._teams: dict[str, tuple[list[int], np.ndarray]] = {}
        self._defense: dict[tuple[str, str], tuple[list[int], np.ndarray]] = {}

    # -- fitting --

    def fit(self, history: pd.DataFrame) -> "TouchdownPropsModel":
        self.history = history.copy()
        touches = _touches(history)
        for pos, grp in history.groupby("position"):
            self.position_rates[pos] = float(grp[TD_STAT_COL].mean())
            pos_touches = touches.loc[grp.index]
            self.position_touches[pos] = float(pos_touches.mean())
            self.position_td_per_touch[pos] = float(
                grp[TD_STAT_COL].sum() / max(pos_touches.sum(), 1.0)
            )
            self.league_pos_pg[pos] = float(
                grp.groupby(["opponent_team", "t"])[TD_STAT_COL].sum().mean()
            )
        self._index()
        return self

    def extend_history(self, later_rows: pd.DataFrame) -> None:
        """Add rows for walk-forward evaluation without refitting priors."""
        self.history = (
            pd.concat([self.history, later_rows], ignore_index=True)
            .drop_duplicates(subset=["player_id", "season", "week"], keep="first")
            .sort_values("t")
            .reset_index(drop=True)
        )
        self._index()

    def _index(self) -> None:
        h = self.history.sort_values("t")
        h = h.assign(_touches=_touches(h))
        self._players = {}
        for pk, g in h.groupby("player_key", sort=False):
            self._players[pk] = (
                g["t"].astype(int).tolist(),
                g[TD_STAT_COL].fillna(0).clip(lower=0).to_numpy(dtype=float),
                g["_touches"].to_numpy(dtype=float),
                g["team"].astype(str).tolist(),
                str(g["position"].iloc[-1]),
            )
        team_games = h.groupby(["team", "t"])[TD_STAT_COL].sum().reset_index()
        self._teams = {
            str(team): (g["t"].astype(int).tolist(), g[TD_STAT_COL].to_numpy(dtype=float))
            for team, g in team_games.groupby("team", sort=False)
        }
        allowed = h.groupby(["opponent_team", "position", "t"])[TD_STAT_COL].sum().reset_index()
        self._defense = {
            (str(opp), str(pos)): (g["t"].astype(int).tolist(),
                                   g[TD_STAT_COL].to_numpy(dtype=float))
            for (opp, pos), g in allowed.groupby(["opponent_team", "position"], sort=False)
        }

    def calibrate(self, seasons: list[int], schedule: pd.DataFrame | None = None,
                  sample: int = 6000) -> "TouchdownPropsModel":
        """
        Isotonic map from raw Poisson P(>=1) to the realised rate, from
        walk-forward projections on `seasons` (training seasons only). With
        a schedule the projections use the market scoring environment, as
        production does.
        """
        from sklearn.isotonic import IsotonicRegression

        if self.history is None:
            raise RuntimeError("fit() before calibrate()")
        rows = self.history[self.history["season"].isin(seasons)]
        if len(rows) > sample:
            rows = rows.sample(sample, random_state=0)
        env = team_expected_tds_lookup(schedule)
        raw, hits = [], []
        for r in rows.itertuples(index=False):
            exp_tds = env.get((int(r.season), int(r.week), str(r.team)))
            proj = self.project(r.player_key, int(r.season), int(r.week),
                                opponent=r.opponent_team, team=r.team,
                                team_expected_tds=exp_tds, calibrated=False)
            if proj is None:
                continue
            raw.append(proj.raw_prob)
            hits.append(float(getattr(r, TD_STAT_COL) >= 1))
        if len(raw) < 200:
            logger.warning("TD calibrate: only %d usable rows — no calibrator", len(raw))
            return self
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip"
                                             ).fit(raw, hits)
        logger.info("Anytime-TD calibrator fit on %d player-games", len(raw))
        return self

    # -- projecting --

    def _usage_prior(self, position: str, touches: np.ndarray, weights: np.ndarray,
                     n_eff: float, values: np.ndarray) -> float:
        """Position TDs-per-touch × the player's (lightly shrunk) touches per game."""
        per_touch = self.position_td_per_touch.get(position)
        pos_touch = self.position_touches.get(position)
        if per_touch is None or pos_touch is None or np.isnan(touches).all():
            return self.position_rates.get(position, float(values.mean()))
        ew_touch = float(np.average(np.nan_to_num(touches, nan=0.0), weights=weights))
        touch = (n_eff * ew_touch + self.TOUCH_PRIOR_GAMES * pos_touch) / (
            n_eff + self.TOUCH_PRIOR_GAMES)
        return per_touch * touch

    def _team_environment(self, team: str | None, before_t: int, expected: float | None) -> float:
        if team is None or expected is None or team not in self._teams:
            return 1.0
        ts, tds = self._teams[team]
        cut = bisect.bisect_left(ts, before_t)
        recent = tds[max(0, cut - self.TEAM_WINDOW):cut]
        if len(recent) < 4 or recent.mean() <= 0:
            return 1.0
        return float(np.clip(expected / recent.mean(), *self.ENV_CLIP))

    def _defense_factor(self, opponent: str | None, position: str, before_t: int) -> float:
        league = self.league_pos_pg.get(position)
        if opponent is None or not league or (opponent, position) not in self._defense:
            return 1.0
        ts, tds = self._defense[(opponent, position)]
        cut = bisect.bisect_left(ts, before_t)
        recent = tds[max(0, cut - self.DEF_WINDOW):cut]
        if len(recent) < 4:
            return 1.0
        return float(np.clip(recent.mean() / league, *self.DEF_CLIP))

    def project(
        self,
        player_key: str,
        asof_season: int,
        asof_week: int,
        opponent: str | None = None,
        team: str | None = None,
        team_expected_tds: float | None = None,
        calibrated: bool = True,
    ) -> TdProjection | None:
        """P(player scores a TD) in (asof_season, asof_week), from prior games only."""
        if self.history is None:
            raise RuntimeError("fit() before project()")
        entry = self._players.get(player_key)
        if entry is None:
            return None
        ts, tds, touches, teams, position = entry
        asof_t = asof_season * 100 + asof_week
        cut = bisect.bisect_left(ts, asof_t)
        if cut < self.MIN_GAMES:
            return None
        values = tds[:cut]
        weights = 0.5 ** (np.arange(cut)[::-1] / self.HALFLIFE)
        ew_rate = float(np.average(values, weights=weights))
        n_eff = float(weights.sum())
        prior = self._usage_prior(position, touches[:cut], weights, n_eff, values)
        rate = (n_eff * ew_rate + self.PRIOR_GAMES * prior) / (n_eff + self.PRIOR_GAMES)

        team = team or teams[cut - 1]
        rate *= self._team_environment(team, asof_t, team_expected_tds)
        rate *= self._defense_factor(opponent, position, asof_t)
        rate = max(rate, 0.01)

        raw = float(1.0 - np.exp(-rate))
        prob = raw
        if calibrated and self.calibrator is not None:
            prob = float(self.calibrator.predict([raw])[0])
        return TdProjection(player=player_key, rate=rate, games=cut, raw_prob=raw,
                            prob=float(np.clip(prob, 0.01, 0.95)))
