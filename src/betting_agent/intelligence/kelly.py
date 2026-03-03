"""
Kelly Criterion bet sizing with hard caps.
"""

from __future__ import annotations

from betting_agent.config import settings


def kelly_fraction(prob: float, american_odds: int | float) -> float:
    """
    Full Kelly fraction (uncapped).
    b = net odds received per unit wagered.
    q = 1 - prob
    Kelly = (b*p - q) / b
    """
    o = float(american_odds)
    b = o / 100.0 if o > 0 else 100.0 / abs(o)
    q = 1.0 - prob
    k = (prob * b - q) / b
    return max(0.0, k)


def scaled_kelly(
    prob: float,
    american_odds: int | float,
    edge: float,
    max_kelly: float | None = None,
) -> float:
    """
    Conservative Kelly with scaling based on edge size.
    Larger edge → less aggressive scaling (more conservative near true Kelly).
    Applies MAX_KELLY_PCT cap.
    """
    if max_kelly is None:
        max_kelly = settings.max_kelly_pct

    k = kelly_fraction(prob, american_odds)
    # Large edges against efficient markets are most likely model errors —
    # scale down more aggressively for larger claimed edges
    if edge > 0.05:
        k *= 0.4
    elif edge > 0.03:
        k *= 0.6
    else:
        k *= 0.8

    return max(0.0, min(k, max_kelly))


def recommended_bet(
    prob: float,
    american_odds: int | float,
    edge: float,
    bankroll: float,
    max_kelly: float | None = None,
    max_bet_pct: float | None = None,
    min_bet_pct: float | None = None,
) -> tuple[float, float]:
    """
    Returns (kelly_fraction_used, dollar_bet_amount).
    Applies both a Kelly cap and a bankroll % cap.
    """
    if max_kelly is None:
        max_kelly = settings.max_kelly_pct
    if max_bet_pct is None:
        max_bet_pct = settings.max_bet_pct
    if min_bet_pct is None:
        min_bet_pct = settings.min_bet_pct

    kf = scaled_kelly(prob, american_odds, edge, max_kelly=max_kelly)
    dollar_amount = kf * bankroll

    # Apply bankroll % caps
    dollar_amount = min(dollar_amount, max_bet_pct * bankroll)
    dollar_amount = max(dollar_amount, min_bet_pct * bankroll)

    return kf, dollar_amount
