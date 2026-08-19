"""
NFL player props: data access and distributional projection models.

Markets modeled (Phase 3 MVP): receptions and receiving yards.

Receptions are counts with variance above the mean, so they get a negative
binomial whose dispersion is fit per position from history. Receiving yards
are non-negative and right-skewed, so they get a gamma whose coefficient of
variation is fit per position. Both center on an exponentially-weighted
average of the player's recent games, shrunk toward the position mean
(few games = mostly prior), scaled by how the opponent's defense treats
that position relative to league average.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import nflreadpy as nfl
import numpy as np
import pandas as pd
from scipy import stats as sps

logger = logging.getLogger(__name__)

RECEIVING_POSITIONS = ("WR", "TE", "RB", "FB")

# Odds API market key → player-stats column (grading supports more markets
# than the models project).
MARKET_STAT_COLUMNS: dict[str, str] = {
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_rush_yds": "rushing_yards",
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
}

MODELED_MARKETS = ("player_receptions", "player_reception_yds")

# Pseudo-line offsets around a projected mean — the neighbourhood real books
# quote in. Used to tune and to validate probability calibration.
RECEPTION_OFFSETS = (-1.5, -0.5, 0.5, 1.5)
YARDS_MULTIPLIERS = (0.75, 0.9, 1.1, 1.25)


def pseudo_lines(market: str, mean: float) -> list[float]:
    if market == "player_receptions":
        return [round(mean) + off for off in RECEPTION_OFFSETS if round(mean) + off > 0]
    lines = [round(mean * m * 2) / 2 for m in YARDS_MULTIPLIERS]
    return [ln + 0.5 if ln == int(ln) else ln for ln in lines if ln > 0]

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


def build_receiving_history(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Filter weekly stats to pass-catchers and the columns the models need,
    ordered by time. Adds `t` — a global game-order index used for
    "everything before week X" splits.
    """
    if stats.empty:
        return stats
    df = stats[stats["position"].isin(RECEIVING_POSITIONS)].copy()
    keep = [
        "player_id", "player_display_name", "position", "season", "week",
        "season_type", "team", "opponent_team", "receptions", "targets",
        "receiving_yards",
    ]
    df = df[[c for c in keep if c in df.columns]]
    df["player_key"] = df["player_display_name"].map(normalize_player)
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["t"] = df["season"] * 100 + df["week"]
    return df


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


@dataclass
class Projection:
    player: str
    market: str
    mean: float
    games: int              # player games the mean is built on
    _dist: object           # frozen scipy distribution
    _calibrator: object = None   # isotonic map raw P(over) → empirical

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
            p = float(self._calibrator.predict([p])[0])
        return float(np.clip(p, 0.01, 0.99))

    def prob_under(self, line: float) -> float:
        return max(0.0, 1.0 - self.prob_over(line) - self._push_mass(line))


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
    MIN_GAMES = 4           # fewer prior games than this → no projection
    DEF_WINDOW = 8          # defensive games in the opponent factor
    DEF_CLIP = (0.8, 1.2)

    def __init__(self, market: str):
        if market not in MODELED_MARKETS:
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

        # Project once per row; retuning only changes the distribution shape,
        # so cache (mean, position, actual) and rebuild distributions per k.
        cached = []
        for _, g in rows.iterrows():
            proj = self.project(g["player_key"], int(g["season"]), int(g["week"]))
            if proj is not None:
                actual = max(0.0, float(g[self.stat_col]))
                cached.append((proj.mean, g["position"], actual))
        if len(cached) < 200:
            logger.warning("tune_dispersion: only %d usable rows — keeping defaults",
                           len(cached))
            return self.dispersion_scale

        if self.market == "player_reception_yds":
            df = pd.DataFrame(
                [(mean, np.log(actual + YARDS_SHIFT) - np.log(mean + YARDS_SHIFT))
                 for mean, _, actual in cached],
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
                for mean, position, actual in cached:
                    ll += self._make_dist(mean, position, k).logpmf(round(actual))
                return -ll
            in50 = in80 = 0
            for mean, position, actual in cached:
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
        for mean, position, actual in cached:
            dist = self._make_dist(mean, position, self.dispersion_scale)
            proj = Projection(player="", market=self.market, mean=mean,
                              games=0, _dist=dist)
            for line in pseudo_lines(self.market, mean):
                raw_p.append(proj._raw_over(line))
                hits.append(float(actual > line))
        self.prob_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(raw_p, hits)
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

    def project(
        self,
        player_key: str,
        asof_season: int,
        asof_week: int,
        opponent: str | None = None,
    ) -> Projection | None:
        """
        Distribution for the player's stat in (asof_season, asof_week),
        using only games strictly before it.
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
        mean = (n_eff * ew_mean + self.PRIOR_GAMES * pos_mean) / (n_eff + self.PRIOR_GAMES)

        if opponent:
            mean *= self._defense_factor(opponent, position, asof_t)
        mean = max(mean, 0.1)

        dist = self._make_dist(mean, position, self.dispersion_scale)

        return Projection(
            player=player_key, market=self.market, mean=mean,
            games=len(past), _dist=dist, _calibrator=self.prob_calibrator,
        )


# ---- grading support ----

def make_stat_lookup(seasons: list[int]):
    """
    Build a stat_lookup(player, market, game) callable for
    grade_prop_picks(). Resolves the game's NFL week from the schedule by
    (home, away, nearest date), then finds the player's row for that week.
    Returns None (leave ungraded) when the player has no row — DNP or stats
    not yet published.
    """
    from betting_agent.sports.nfl.loader import NFLDataLoader

    stats = load_player_stats(seasons)
    if stats.empty:
        return lambda player, market, game: None
    stats = stats.copy()
    stats["player_key"] = stats["player_display_name"].map(normalize_player)

    schedules = NFLDataLoader().load_schedules(seasons).to_pandas()
    from betting_agent.sports.nfl.features import normalise_raw_schedules
    schedules = normalise_raw_schedules(schedules)

    def lookup(player: str | None, market: str | None, game) -> float | None:
        col = MARKET_STAT_COLUMNS.get(market or "")
        if col is None or not player:
            return None
        sched = schedules[
            (schedules["home_team"] == game.home_team)
            & (schedules["away_team"] == game.away_team)
        ]
        if sched.empty or "week" not in sched.columns:
            return None
        game_date = pd.Timestamp(game.game_date)
        sched = sched.assign(_gap=(sched["game_date"] - game_date).abs())
        row = sched.sort_values("_gap").iloc[0]
        if row["_gap"] > pd.Timedelta(days=2):
            return None
        season, week = int(row["season"]), int(row["week"])

        key = normalize_player(player)
        hits = stats[
            (stats["player_key"] == key)
            & (stats["season"] == season)
            & (stats["week"] == week)
            & (stats["team"].isin([game.home_team, game.away_team]))
        ]
        if hits.empty or col not in hits.columns:
            return None
        return float(hits.iloc[0][col])

    return lookup


# ---- odds fetching ----

def fetch_prop_odds(
    sport_key: str = "americanfootball_nfl",
    markets: list[str] | None = None,
    bookmakers: list[str] | None = None,
    max_events: int | None = None,
) -> list[dict]:
    """
    Fetch player prop odds. Props are only served by the per-event endpoint,
    so this costs one API call per event.
    """
    from betting_agent.api.odds import OddsAPIClient

    if markets is None:
        markets = list(MODELED_MARKETS)
    client = OddsAPIClient()
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
