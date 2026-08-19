"""
Real historical NFL closing lines.

nflreadpy ships the closing spread, total, and moneyline — with prices — on
every schedule row, going back decades and costing nothing. That makes a
genuine backtest possible: bet the model against the number the market
actually closed at, and grade against the actual score.

Betting at the close is the hardest available test. The closing line is the
market's most informed price, and a model that clears it is showing real edge
rather than the appearance of one. The synthetic-odds path this replaces
centred the fake market on the model's own prediction, so it measured nothing
about the market at all.

Column semantics, per nflreadpy:
  spread_line   points the HOME team is favoured by (positive = home favourite)
  total_line    combined points line
  *_moneyline   American odds
  *_spread_odds American odds on each spread side
  over/under_odds American odds on each total side
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Closing-line columns carried on nflreadpy schedule rows.
LINE_COLUMNS = (
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
    "home_spread_odds",
    "away_spread_odds",
    "over_odds",
    "under_odds",
)

PUSH = "push"
WIN = "win"
LOSS = "loss"


def has_closing_lines(df: pd.DataFrame) -> bool:
    """True when the frame carries the closing-line columns."""
    return all(c in df.columns for c in ("spread_line", "total_line", "home_moneyline"))


def line_coverage(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows with a usable line for each market."""
    if df.empty:
        return {"moneyline": 0.0, "spread": 0.0, "total": 0.0}
    return {
        "moneyline": float(
            (df.get("home_moneyline").notna() & df.get("away_moneyline").notna()).mean()
        ),
        "spread": float(
            (df.get("spread_line").notna() & df.get("home_spread_odds").notna()).mean()
        ),
        "total": float(
            (df.get("total_line").notna() & df.get("over_odds").notna()).mean()
        ),
    }


def grade_moneyline(home_score: float, away_score: float, side: str) -> str:
    """Grade a moneyline bet on "home" or "away"."""
    if home_score == away_score:
        return PUSH
    home_won = home_score > away_score
    picked_home = side == "home"
    return WIN if home_won == picked_home else LOSS


def grade_spread(
    home_score: float, away_score: float, spread_line: float, side: str
) -> str:
    """
    Grade a spread bet. `spread_line` is the home team's handicap in
    nflreadpy's convention: positive means the home team is favoured by that
    many points, so the home side needs a winning margin above it.
    """
    margin = home_score - away_score
    if margin == spread_line:
        return PUSH
    home_covered = margin > spread_line
    picked_home = side == "home"
    return WIN if home_covered == picked_home else LOSS


def grade_total(
    home_score: float, away_score: float, total_line: float, side: str
) -> str:
    """Grade an over/under bet."""
    total = home_score + away_score
    if total == total_line:
        return PUSH
    went_over = total > total_line
    picked_over = side == "over"
    return WIN if went_over == picked_over else LOSS


def payout(bet_amount: float, american_odds: float, result: str) -> float:
    """Profit or loss on a settled bet (0 on a push — the stake comes back)."""
    if result == PUSH:
        return 0.0
    if result == LOSS:
        return -bet_amount
    if american_odds > 0:
        return bet_amount * (american_odds / 100.0)
    return bet_amount * (100.0 / abs(american_odds))


def breakeven_rate(american_odds: float) -> float:
    """Win rate needed to break even at a price, ignoring pushes."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)
