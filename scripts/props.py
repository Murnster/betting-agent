#!/usr/bin/env python
"""
NFL player prop picks (Phase 3 — PAPER TRADING).

Fetches per-event prop odds (bet365 by default), projects each player's stat
distribution from nflreadpy weekly stats, and surfaces overs/unders where the
model probability beats the de-vigged market probability by the edge
threshold. Picks save as bet_type="prop" with recommended (paper) stakes —
this phase logs and grades picks, it does not tell you to bet money.

Usage:
    uv run python scripts/props.py                     # print picks
    uv run python scripts/props.py --save              # log to picks table
    uv run python scripts/props.py --max-events 5      # cap API spend
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

from betting_agent.config import settings
from betting_agent.intelligence.ev import american_to_implied_prob, remove_vig
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.intelligence.picks import BetCandidate, save_picks_to_db
from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    ReceivingPropsModel,
    build_receiving_history,
    fetch_prop_odds,
    load_player_stats,
    normalize_player,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _current_nfl_season(today: date) -> int:
    # NFL seasons run Sep–Feb; Jan/Feb games belong to the previous year's season.
    return today.year if today.month >= 8 else today.year - 1


def _pair_outcomes(market: dict) -> dict[tuple[str, float], dict[str, dict]]:
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


def _nfl_week(event_date: date, season: int) -> int:
    """Approximate NFL week from the date — only used to time-split history."""
    season_start = date(season, 9, 1)
    days = (event_date - season_start).days
    return max(1, min(22, days // 7 + 1))


def generate_prop_candidates(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    bankroll: float,
    min_edge: float,
    season: int,
) -> list[BetCandidate]:
    from betting_agent.sports.teams import canonical_team

    candidates: list[BetCandidate] = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        home_abbrev = canonical_team("NFL", home)
        away_abbrev = canonical_team("NFL", away)
        try:
            event_date = datetime.fromisoformat(
                event.get("commence_time", "").replace("Z", "+00:00")
            ).date()
        except ValueError:
            event_date = date.today()
        week = _nfl_week(event_date, season)

        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                key = market.get("key")
                model = models.get(key)
                if model is None:
                    continue
                for (player, line), pair in _pair_outcomes(market).items():
                    over = pair.get("Over")
                    under = pair.get("Under")
                    if not over or not under:
                        continue
                    player_key = normalize_player(player)
                    team_rows = model.history[model.history["player_key"] == player_key]
                    player_team = team_rows["team"].iloc[-1] if len(team_rows) else None
                    opponent = None
                    if player_team == home_abbrev:
                        opponent = away_abbrev
                    elif player_team == away_abbrev:
                        opponent = home_abbrev

                    proj = model.project(player_key, season, week, opponent=opponent)
                    if proj is None:
                        continue

                    fair_over, fair_under = remove_vig(
                        american_to_implied_prob(over["price"]),
                        american_to_implied_prob(under["price"]),
                    )
                    p_over = proj.prob_over(line)
                    for side, price, model_p, fair_p in (
                        ("over", over["price"], p_over, fair_over),
                        ("under", under["price"], proj.prob_under(line), fair_under),
                    ):
                        edge = model_p - fair_p
                        if edge < min_edge:
                            continue
                        kelly, bet = recommended_bet(model_p, int(price), edge, bankroll)
                        candidates.append(BetCandidate(
                            game_id=0,
                            external_id=event.get("id"),
                            home_team=home,
                            away_team=away,
                            game_date=date.today(),
                            scheduled_game_date=event_date,
                            sport="NFL",
                            bet_type="prop",
                            pick_side=side,
                            player=player,
                            market=key,
                            line=line,
                            model_prob=model_p,
                            implied_prob=fair_p,
                            edge=edge,
                            odds=int(price),
                            kelly_fraction=kelly,
                            recommended_bet=bet,
                            bankroll_at_pick=bankroll,
                            extra={"projection_mean": round(proj.mean, 2),
                                   "projection_games": proj.games,
                                   "bookmaker": book.get("key")},
                        ))
    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL prop picks (paper trading)")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--save", action="store_true", help="Persist picks to DB")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Cap per-event odds calls (API credit control)")
    # Calibration (2024-25 eval) shows ~2-3pp residual drift in P(over), so
    # props need a fatter edge floor than game markets or the drift itself
    # would read as edge.
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Minimum prop edge (default 0.05)")
    parser.add_argument("--max-picks", type=int, default=10)
    args = parser.parse_args()

    bankroll = args.bankroll or settings.starting_bankroll
    min_edge = args.min_edge
    season = _current_nfl_season(date.today())

    logger.info("Loading player stats (%s-%s)...", season - 2, season)
    stats = load_player_stats([season - 2, season - 1, season])
    history = build_receiving_history(stats)
    if history.empty:
        raise SystemExit("No player stats available — cannot project props.")
    models = {}
    for m in MODELED_MARKETS:
        model = ReceivingPropsModel(m).fit(history)
        model.tune_dispersion([season - 1])
        models[m] = model

    books = settings.preferred_bookmaker_list or ["bet365"]
    logger.info("Fetching prop odds (books: %s)...", ",".join(books))
    events = fetch_prop_odds(bookmakers=books, max_events=args.max_events)
    if not events:
        raise SystemExit("No prop odds returned — check ODDS_API_KEY / season timing.")

    candidates = generate_prop_candidates(events, models, bankroll, min_edge, season)
    candidates = candidates[:args.max_picks]

    if not candidates:
        print("\nNo prop edges clear the threshold today.")
        return

    print(f"\n{'player':<24} {'market':<22} {'side':<6} {'line':>6} {'odds':>6} "
          f"{'model':>7} {'fair':>7} {'edge':>7} {'paper $':>8}")
    for c in candidates:
        print(f"{c.player:<24} {c.market:<22} {c.pick_side:<6} {c.line:>6.1f} "
              f"{c.odds:>+6} {c.model_prob:>6.1%} {c.implied_prob:>6.1%} "
              f"{c.edge:>+6.1%} {c.recommended_bet:>8.2f}")
    print(f"\n{len(candidates)} paper picks. These are NOT bets — Phase 3 "
          "validates the projections first.")

    if args.save:
        save_picks_to_db(candidates)
        print("Saved to picks table (bet_type='prop').")


if __name__ == "__main__":
    main()
