"""
Edge (+EV) calculation.
Converts American odds to implied probability, then computes model edge.
"""

from __future__ import annotations


def american_to_implied_prob(american_odds: int | float) -> float:
    """
    Convert American odds to implied win probability (no vig removed).
    E.g. -110 → 0.5238, +150 → 0.4000
    """
    o = float(american_odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return abs(o) / (abs(o) + 100.0)


def remove_vig(home_implied: float, away_implied: float) -> tuple[float, float]:
    """
    Remove bookmaker vig from a two-way market.
    Returns fair implied probabilities that sum to 1.0.
    """
    total = home_implied + away_implied
    return home_implied / total, away_implied / total


def calculate_edge(model_prob: float, american_odds: int | float) -> float:
    """
    Edge = model probability - implied probability from market odds (vig included).
    Use calculate_edge_fair() for moneylines where both sides' odds are available.
    """
    implied = american_to_implied_prob(american_odds)
    return model_prob - implied


def calculate_edge_fair(
    model_prob: float,
    home_odds: int | float,
    away_odds: int | float,
    pick_home: bool = True,
) -> float:
    """
    Edge vs vig-removed fair probability.
    Requires both sides' odds to accurately remove vig.
    Only positive values represent real edge against the fair line.
    """
    home_imp = american_to_implied_prob(home_odds)
    away_imp = american_to_implied_prob(away_odds)
    fair_home, fair_away = remove_vig(home_imp, away_imp)
    return model_prob - (fair_home if pick_home else fair_away)


def calculate_spread_edge(
    predicted_margin: float,
    spread_line: float,
    american_odds: int | float,
    away_odds: int | float | None = None,
    sigma: float = 14.0,
) -> tuple[float, float]:
    """
    For spread bets, convert the predicted margin into a cover probability
    using a simple normal approximation.
    sigma: scoring variance (14.0 for NFL, 11.5 for NBA).
    If away_odds is provided, vig is removed before computing edge.
    Otherwise assumes a symmetric market (same odds on both sides).
    Returns (cover_prob, edge).
    """
    import math
    z = (predicted_margin - spread_line) / sigma
    cover_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    other_odds = away_odds if away_odds is not None else american_odds
    home_imp = american_to_implied_prob(american_odds)
    away_imp = american_to_implied_prob(other_odds)
    fair_home, _ = remove_vig(home_imp, away_imp)
    edge = cover_prob - fair_home
    return cover_prob, edge


def calculate_total_edge(
    predicted_total: float,
    total_line: float,
    american_odds: int | float,
    side: str = "over",
    other_odds: int | float | None = None,
    sigma: float = 14.0,
) -> tuple[float, float]:
    """
    Over/under edge using normal approximation.
    sigma: scoring variance (14.0 for NFL, 11.5 for NBA).
    side: 'over' or 'under'
    If other_odds (the opposite side) is provided, vig is removed.
    Otherwise assumes a symmetric market (same odds on both sides).
    Returns (cover_prob, edge).
    """
    import math
    z = (predicted_total - total_line) / sigma
    over_prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    cover_prob = over_prob if side == "over" else 1.0 - over_prob

    opp_odds = other_odds if other_odds is not None else american_odds
    this_imp = american_to_implied_prob(american_odds)
    opp_imp = american_to_implied_prob(opp_odds)
    fair_this, _ = remove_vig(this_imp, opp_imp)
    edge = cover_prob - fair_this
    return cover_prob, edge
