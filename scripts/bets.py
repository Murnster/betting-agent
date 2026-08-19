#!/usr/bin/env python
"""
Record the bets actually placed at the book, so the ledger tracks real money
rather than recommendations.

Usage:
    uv run python scripts/bets.py list                 # open (ungraded) picks
    uv run python scripts/bets.py list --all           # include settled picks
    uv run python scripts/bets.py set 42 --stake 25    # bet $25 on pick 42
    uv run python scripts/bets.py set 42 --stake 25 --odds -115
    uv run python scripts/bets.py set 42 --stake 0     # logged but not placed
    uv run python scripts/bets.py ledger [--sport NFL] # equity curve
"""

from __future__ import annotations

import argparse
import logging

from betting_agent.accounting.grader import _calculate_pnl
from betting_agent.accounting.ledger import equity_curve, ledger_summary
from betting_agent.db.models import Game, Pick
from betting_agent.db.session import get_session

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _describe(pick: Pick, game: Game) -> str:
    matchup = f"{game.away_team}@{game.home_team}"
    label = pick.pick_side
    if pick.bet_type == "prop":
        label = f"{pick.player} {pick.pick_side} {pick.line} {pick.market}"
    return f"{matchup:<14} {pick.bet_type:<9} {label}"


def cmd_list(args) -> None:
    with get_session() as session:
        q = session.query(Pick, Game).join(Game)
        if not args.all:
            q = q.filter(Pick.result.is_(None))
        if args.sport:
            q = q.filter(Pick.sport == args.sport.upper())
        rows = q.order_by(Pick.pick_date.desc(), Pick.id).limit(args.limit).all()

        if not rows:
            print("No picks found.")
            return
        print(f"{'id':>5} {'date':<11} {'sport':<5} {'pick':<58} "
              f"{'odds':>6} {'rec $':>7} {'actual $':>9} {'result':<7}")
        for pick, game in rows:
            actual = f"{pick.actual_bet:.2f}" if pick.actual_bet is not None else "-"
            odds = pick.actual_odds if pick.actual_odds is not None else pick.odds
            print(f"{pick.id:>5} {str(pick.pick_date):<11} {pick.sport:<5} "
                  f"{_describe(pick, game):<58} {odds:>+6} "
                  f"{(pick.recommended_bet or 0):>7.2f} {actual:>9} "
                  f"{pick.result or 'open':<7}")


def cmd_set(args) -> None:
    with get_session() as session:
        pick = session.get(Pick, args.pick_id)
        if pick is None:
            raise SystemExit(f"No pick with id {args.pick_id}")
        pick.actual_bet = args.stake
        if args.odds is not None:
            pick.actual_odds = args.odds
        if pick.result is not None:
            # Already settled — rebook the P&L at the real stake and price.
            pick.pnl = _calculate_pnl(pick, pick.result)
            print(f"Pick {pick.id} already graded {pick.result}; "
                  f"P&L rebooked to {pick.pnl:+.2f} at the recorded bet.")
        print(f"Pick {pick.id}: actual stake ${args.stake:.2f}"
              + (f" at {args.odds:+d}" if args.odds is not None else ""))


def cmd_ledger(args) -> None:
    curve = equity_curve(sport=args.sport.upper() if args.sport else None)
    if not curve:
        print("No settled picks yet.")
        return
    print(f"{'date':<11} {'pick':>5} {'type':<9} {'stake':>7} {'result':<6} "
          f"{'pnl':>8} {'equity':>9}")
    for e in curve[-args.limit:]:
        print(f"{str(e.event_date):<11} {e.pick_id:>5} {e.bet_type:<9} "
              f"{e.stake:>7.2f} {e.result:<6} {e.pnl:>+8.2f} {e.equity:>9.2f}")
    s = ledger_summary(sport=args.sport.upper() if args.sport else None)
    print(f"\nStart ${s['starting_bankroll']:,.2f} → ${s['current_bankroll']:,.2f} "
          f"({s['total_pnl']:+,.2f}) | peak ${s['peak_equity']:,.2f} | "
          f"max drawdown ${s['max_drawdown']:,.2f} | {s['settled_picks']} picks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Track bets actually placed")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Show picks and their recorded stakes")
    p_list.add_argument("--all", action="store_true", help="Include settled picks")
    p_list.add_argument("--sport", type=str, default=None)
    p_list.add_argument("--limit", type=int, default=40)
    p_list.set_defaults(func=cmd_list)

    p_set = sub.add_parser("set", help="Record the stake actually placed")
    p_set.add_argument("pick_id", type=int)
    p_set.add_argument("--stake", type=float, required=True,
                       help="Dollars actually bet (0 = not placed)")
    p_set.add_argument("--odds", type=int, default=None,
                       help="American odds actually taken, if different")
    p_set.set_defaults(func=cmd_set)

    p_ledger = sub.add_parser("ledger", help="Equity curve from settled picks")
    p_ledger.add_argument("--sport", type=str, default=None)
    p_ledger.add_argument("--limit", type=int, default=30)
    p_ledger.set_defaults(func=cmd_ledger)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
