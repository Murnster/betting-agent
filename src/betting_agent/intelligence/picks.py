"""
POTD (Picks of the Day) generation + CLI formatting.
Generates best pick per bet type per game, filtered by edge threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from betting_agent.config import settings
from betting_agent.db.models import Game, Pick
from betting_agent.db.session import get_session
from betting_agent.intelligence.ev import (
    american_to_implied_prob,
    calculate_edge,
    calculate_edge_fair,
    calculate_spread_edge,
    calculate_total_edge,
)
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.models.engine import PredictionEngine

logger = logging.getLogger(__name__)


@dataclass
class BetCandidate:
    game_id: int
    home_team: str
    away_team: str
    game_date: date
    sport: str
    bet_type: str          # moneyline | spread | total
    pick_side: str          # team name or over/under
    model_prob: float
    implied_prob: float
    edge: float
    odds: int
    kelly_fraction: float = 0.0
    recommended_bet: float = 0.0
    bankroll_at_pick: float = 0.0
    extra: dict = field(default_factory=dict)  # spread_line, total_line, etc.


def generate_picks(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    odds_data: list[dict],
    engine: PredictionEngine,
    bankroll: float | None = None,
    sport: str = "NFL",
    pick_date: date | None = None,
    sentiment_scores: dict[str, float] | None = None,
    total_sigma: float = 14.0,
    margin_sigma: float = 14.0,
) -> list[BetCandidate]:
    """
    Main pick generation pipeline.

    features: model feature matrix (one row per game)
    metadata: raw game metadata aligned by index (home_team, away_team, game_id, etc.)
    odds_data: raw Odds API response
    engine: loaded PredictionEngine

    Returns list of BetCandidate objects that clear the edge threshold.
    """
    if bankroll is None:
        bankroll = settings.starting_bankroll
    if pick_date is None:
        pick_date = date.today()

    # Run predictions
    predictions = engine.predict(features)
    candidates: list[BetCandidate] = []

    for i, (feat_idx, pred_row) in enumerate(predictions.iterrows()):
        meta = metadata.iloc[i] if i < len(metadata) else {}
        raw_game_id = meta.get("game_id", 0)
        try:
            game_id = int(raw_game_id)
        except (TypeError, ValueError):
            game_id = 0
        home = str(meta.get("home_team", ""))
        away = str(meta.get("away_team", ""))

        # Find best odds for this game (use Odds API name if available)
        home_odds_name = str(meta.get("home_team_odds", home))
        best_odds = _find_best_odds(odds_data, home_odds_name)

        win_prob = float(pred_row["win_prob"])
        home_pred = float(pred_row["home_pred_score"])
        away_pred = float(pred_row["away_pred_score"])
        pred_margin = float(pred_row["pred_margin"])
        pred_total = float(pred_row["pred_total"])

        # Sentiment adjustment helper
        def _sent_adj(team: str) -> float:
            if sentiment_scores is None:
                return 0.0
            return sentiment_scores.get(team, 0.0) * settings.sentiment_weight

        # ---- Moneyline ----
        ml_home = best_odds.get("home_price")
        ml_away = best_odds.get("away_price")

        if ml_home:
            if ml_away:
                edge = calculate_edge_fair(win_prob, ml_home, ml_away, pick_home=True)
            else:
                edge = calculate_edge(win_prob, ml_home)
            edge += _sent_adj(home)
            if edge >= settings.min_edge_pct:
                implied = american_to_implied_prob(ml_home)
                kf, bet = recommended_bet(win_prob, ml_home, edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    bet_type="moneyline", pick_side=home,
                    model_prob=win_prob, implied_prob=implied, edge=edge,
                    odds=ml_home, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll,
                ))

        away_win_prob = 1.0 - win_prob
        if ml_away:
            if ml_home:
                edge = calculate_edge_fair(away_win_prob, ml_home, ml_away, pick_home=False)
            else:
                edge = calculate_edge(away_win_prob, ml_away)
            edge += _sent_adj(away)
            if edge >= settings.min_edge_pct:
                implied = american_to_implied_prob(ml_away)
                kf, bet = recommended_bet(away_win_prob, ml_away, edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    bet_type="moneyline", pick_side=away,
                    model_prob=away_win_prob, implied_prob=implied, edge=edge,
                    odds=ml_away, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll,
                ))

        # ---- Spread ----
        spread_line = best_odds.get("spread_home")
        spread_home_price = best_odds.get("spread_home_price")
        spread_away_price = best_odds.get("spread_away_price")
        if spread_line is not None and spread_home_price:
            cover_prob, s_edge = calculate_spread_edge(
                pred_margin, -spread_line, spread_home_price,
                away_odds=spread_away_price, sigma=margin_sigma,
            )
            s_edge += _sent_adj(home)
            if s_edge >= settings.min_edge_pct:
                implied = american_to_implied_prob(spread_home_price)
                kf, bet = recommended_bet(cover_prob, spread_home_price, s_edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    bet_type="spread", pick_side=f"{home} {spread_line:+.1f}",
                    model_prob=cover_prob, implied_prob=implied, edge=s_edge,
                    odds=spread_home_price, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll,
                    extra={"spread_line": spread_line},
                ))

        # ---- Total (Over/Under) ----
        total_line = best_odds.get("total_line")
        over_price = best_odds.get("over_price")
        under_price = best_odds.get("under_price")

        if total_line and over_price:
            over_prob, o_edge = calculate_total_edge(pred_total, total_line, over_price, "over", other_odds=under_price, sigma=total_sigma)
            o_edge += (_sent_adj(home) + _sent_adj(away)) / 2
            if o_edge >= settings.min_edge_pct:
                implied = american_to_implied_prob(over_price)
                kf, bet = recommended_bet(over_prob, over_price, o_edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    bet_type="total", pick_side=f"over {total_line}",
                    model_prob=over_prob, implied_prob=implied, edge=o_edge,
                    odds=over_price, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll,
                    extra={"total_line": total_line},
                ))

        if total_line and under_price:
            under_prob, u_edge = calculate_total_edge(pred_total, total_line, under_price, "under", other_odds=over_price, sigma=total_sigma)
            u_edge += (_sent_adj(home) + _sent_adj(away)) / 2
            if u_edge >= settings.min_edge_pct:
                implied = american_to_implied_prob(under_price)
                kf, bet = recommended_bet(under_prob, under_price, u_edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    bet_type="total", pick_side=f"under {total_line}",
                    model_prob=under_prob, implied_prob=implied, edge=u_edge,
                    odds=under_price, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll,
                    extra={"total_line": total_line},
                ))

    # Dedup: keep only highest-edge candidate per (game_id, bet_type)
    seen: dict[tuple[int, str], BetCandidate] = {}
    for c in candidates:
        key = (c.game_id, c.bet_type)
        if key not in seen or c.edge > seen[key].edge:
            seen[key] = c
    candidates = list(seen.values())

    # Same-game correlation adjustment: scale Kelly down by 1/sqrt(n)
    # where n = number of bets on the same game
    from collections import Counter
    import math
    game_counts = Counter(c.game_id for c in candidates)
    for c in candidates:
        n = game_counts[c.game_id]
        if n > 1:
            corr_factor = 1.0 / math.sqrt(n)
            c.kelly_fraction *= corr_factor
            c.recommended_bet *= corr_factor

    # Sort by edge descending
    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates


def _find_best_odds(odds_data: list[dict], home_team: str) -> dict:
    """Find best available odds for the home_team matchup in raw Odds API data."""
    from betting_agent.api.odds import OddsAPIClient
    client = OddsAPIClient()
    return client.get_best_odds(odds_data, home_team)


def save_picks_to_db(candidates: list[BetCandidate]) -> None:
    """
    Persist POTD candidates to the picks table.
    Prevents duplicates by checking if (game_id, bet_type, pick_date) already exists.
    """
    if not candidates:
        return

    with get_session() as session:
        # Pre-fetch existing picks for these games/date to minimize queries
        game_ids = {c.game_id for c in candidates}
        pick_date = candidates[0].game_date  # Assuming all candidates share the date

        existing_rows = (
            session.query(Pick)
            .filter(
                Pick.pick_date == pick_date,
                Pick.game_id.in_(game_ids)
            )
            .all()
        )
        
        # Create set of (game_id, bet_type) that already exist
        existing_keys = {(r.game_id, r.bet_type) for r in existing_rows}

        added = 0
        skipped = 0
        for c in candidates:
            # If we already have a pick for this game + bet type (e.g. spread), skip it.
            # This prevents double-betting if the script re-runs or line moves slightly.
            if (c.game_id, c.bet_type) in existing_keys:
                skipped += 1
                continue

            pick = Pick(
                game_id=c.game_id,
                sport=c.sport,
                pick_date=c.game_date,
                bet_type=c.bet_type,
                pick_side=c.pick_side,
                model_prob=c.model_prob,
                implied_prob=c.implied_prob,
                edge=c.edge,
                odds=c.odds,
                kelly_fraction=c.kelly_fraction,
                recommended_bet=c.recommended_bet,
                bankroll_at_pick=c.bankroll_at_pick,
            )
            session.add(pick)
            # Add to local set so we don't try to add duplicates within the same batch
            existing_keys.add((c.game_id, c.bet_type))
            added += 1

    if added > 0 or skipped > 0:
        logger.info("Saved %d picks to DB (skipped %d duplicates)", added, skipped)


def _pick_label(pick: BetCandidate) -> str:
    """Build a human-readable label for a pick (e.g. 'OVER 215.5', 'WAS +14.5')."""
    if pick.bet_type == "total":
        return pick.pick_side.upper()
    if pick.bet_type == "spread":
        return pick.pick_side
    # moneyline
    return f"{pick.pick_side} Moneyline"


def _confidence_stars(edge: float) -> str:
    """Return 1-5 star rating based on edge magnitude."""
    if edge >= 0.20:
        return "\u2605\u2605\u2605\u2605\u2605"
    if edge >= 0.12:
        return "\u2605\u2605\u2605\u2605\u2606"
    if edge >= 0.07:
        return "\u2605\u2605\u2605\u2606\u2606"
    if edge >= 0.04:
        return "\u2605\u2605\u2606\u2606\u2606"
    return "\u2605\u2606\u2606\u2606\u2606"


def format_picks_cli(candidates: list[BetCandidate], bankroll: float) -> str:
    """Format picks for terminal output in card-style ranked by confidence."""
    if not candidates:
        return "\nNo +EV picks found for today.\n"

    # Sort by edge descending (best pick first)
    ranked = sorted(candidates, key=lambda c: c.edge, reverse=True)

    total_wagered = sum(c.recommended_bet for c in ranked)
    pct_bankroll = (total_wagered / bankroll * 100) if bankroll > 0 else 0

    lines = [
        "",
        "=" * 58,
        f"  PICKS OF THE DAY \u2014 {ranked[0].game_date}",
        f"  Bankroll: ${bankroll:,.2f}  |  "
        f"{len(ranked)} picks  |  "
        f"Total action: ${total_wagered:.2f} ({pct_bankroll:.1f}%)",
        "=" * 58,
    ]

    # Collect LLM analyses by game_id (show after cards)
    analyses: list[str] = []

    W = 54  # inner card width
    for rank, pick in enumerate(ranked, 1):
        matchup = f"{pick.away_team} @ {pick.home_team}"
        label = _pick_label(pick)
        stars = _confidence_stars(pick.edge)

        def _card_row(left: str, right: str) -> str:
            padding = W - len(left) - len(right)
            return f"  \u2502{left}{' ' * max(padding, 1)}{right}\u2502"

        lines.append("")
        lines.append("  \u250c" + "\u2500" * W + "\u2510")
        lines.append(_card_row(
            f"  #{rank}  {stars}  {label}",
            f"{matchup}  ",
        ))
        lines.append(_card_row(
            f"       Odds: {pick.odds:+d}",
            f"Bet: ${pick.recommended_bet:.2f}  ",
        ))
        lines.append(_card_row(
            f"       Edge: {pick.edge:+.1%}",
            f"Kelly: {pick.kelly_fraction:.2%}  ",
        ))
        lines.append(_card_row(
            f"       Model: {pick.model_prob:.1%}",
            f"Implied: {pick.implied_prob:.1%}  ",
        ))
        lines.append("  \u2514" + "\u2500" * W + "\u2518")

        # Collect LLM analysis if present (only once per game)
        analysis = pick.extra.get("analysis")
        if analysis and analysis.get("summary"):
            summary = f"  {matchup}: {analysis['summary']}"
            if summary not in analyses:
                analyses.append(summary)

    # LLM analyses at the bottom
    if analyses:
        lines.append("")
        for a in analyses:
            lines.append(f"  LLM {a}")

    lines.append("")
    lines.append("=" * 58)
    lines.append("")
    return "\n".join(lines)
