"""
ROI and performance reporting.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


from betting_agent.accounting.prop_clv import line_moved_for_pick
from betting_agent.db.models import Pick
from betting_agent.db.session import get_session


def _pct(num: int, denom: int) -> float:
    return (num / denom * 100.0) if denom else 0.0


# NFL game-market picks are market leans (paper, stake 0 unless the edge is
# positive), reported separately from the prop picks that carry the money.
LEAN_BET_TYPES = ("moneyline", "spread", "total")

#: Sports where the card IS the offer: only picks that made a card count in the
#: reported record, ROI and bankroll. props.py saves every candidate that clears
#: the floors, but the ones that never reached a card were never offered to bet
#: — they stay in the table (and keep grading) as model evaluation only.
CARDED_SPORTS = frozenset({"NFL"})


def carded_only(sport: str | None) -> bool | None:
    """
    The `on_card` filter a sport's reports should run with.

    True  -> carded picks only (the offer).
    None  -> every saved pick (sports with no card, e.g. the NBA/NHL game
             picks, whose rows all predate on_card and are all False).
    """
    return True if (sport or "").upper() in CARDED_SPORTS else None


def get_summary(
    sport: str | None = None,
    since: date | None = None,
    until: date | None = None,
    bet_type: str | list[str] | tuple[str, ...] | None = None,
    season: int | None = None,
    graded_since: datetime | None = None,
    graded_until: datetime | None = None,
    market: str | None = None,
    exclude_market: str | None = None,
    strategy: str | None = None,
    exclude_strategy: str | list[str] | tuple[str, ...] | None = None,
    on_card: bool | None = None,
) -> dict[str, Any]:
    """
    Aggregate ROI report.

    market / exclude_market filter props by Odds API market key — the
    anytime-TD scorers are reported apart from the receiving props (they are
    lean-like: saved at stake 0 unless the edge is inside the window).
    strategy / exclude_strategy filter on Pick.strategy — the ladder-hits
    and straight-overs sections keep their own paper books (`strategy=
    "ladder"` / `"overs"`, see ledger.SIDE_BOOKS); NULL rows are the default
    strategies and pass an exclude filter (one name or several).
    Returns dict with: total_bets, wins, losses, pushes, win_rate, total_pnl,
                       total_wagered, roi_pct, avg_edge, avg_clv.

    since/until filter on pick_date (the day the pick was made). graded_since /
    graded_until filter on graded_at instead — the daily results post uses
    graded_since, because an NFL pick is made days before its game and graded
    days after; a re-post of an earlier day bounds both ends.

    on_card=True reports only the picks that made a card — the ones actually
    offered. Off-card picks stay saved and graded for model evaluation but do
    not belong in a record or a bankroll. Use carded_only(sport) to pick the
    right value for a sport; None counts every saved pick.
    """
    from betting_agent.db.models import Game
    with get_session() as session:
        q = session.query(Pick).join(Game).filter(Pick.result.isnot(None))
        if sport:
            q = q.filter(Pick.sport == sport)
        if since:
            q = q.filter(Pick.pick_date >= since)
        if until:
            q = q.filter(Pick.pick_date <= until)
        if graded_since is not None:
            q = q.filter(Pick.graded_at >= graded_since)
        if graded_until is not None:
            q = q.filter(Pick.graded_at <= graded_until)
        if bet_type:
            if isinstance(bet_type, str):
                q = q.filter(Pick.bet_type == bet_type)
            else:
                q = q.filter(Pick.bet_type.in_(list(bet_type)))
        if season:
            q = q.filter(Game.season == season)
        if market:
            q = q.filter(Pick.market == market)
        if exclude_market:
            q = q.filter((Pick.market.is_(None)) | (Pick.market != exclude_market))
        if strategy:
            q = q.filter(Pick.strategy == strategy)
        if exclude_strategy:
            from betting_agent.accounting.ledger import strategy_exclusion
            q = q.filter(strategy_exclusion(Pick.strategy, exclude_strategy))
        if on_card is not None:
            q = q.filter(Pick.on_card.is_(on_card))

        picks = q.all()

    if not picks:
        return {"message": "No graded picks found"}

    total = len(picks)
    wins = sum(1 for p in picks if p.result == "win")
    losses = sum(1 for p in picks if p.result == "loss")
    pushes = sum(1 for p in picks if p.result == "push")
    voids = sum(1 for p in picks if p.result == "void")
    total_pnl = sum(p.pnl or 0.0 for p in picks)
    # A voided bet never happened (DNP — stake returned), so it doesn't
    # count as money wagered.
    total_wagered = sum(p.stake for p in picks if p.result != "void")
    avg_edge = sum(p.edge for p in picks) / total
    fair = [getattr(p, "implied_prob", None) for p in picks if p.result in ("win", "loss")]
    fair = [f for f in fair if f is not None]
    avg_fair = sum(fair) / len(fair) if fair else None
    clv_picks = [p for p in picks if p.clv is not None]
    avg_clv = sum(p.clv for p in clv_picks) / len(clv_picks) if clv_picks else None
    clv_hits = sum(1 for p in clv_picks if p.clv > 0)

    # Props: the book may have moved the NUMBER rather than the price. Count
    # moves for/against the pick from the captured closing line.
    moves_for = moves_against = 0
    for p in picks:
        closing_line = getattr(p, "closing_line", None)
        line = getattr(p, "line", None)
        if closing_line is None or line is None:
            continue
        moved = line_moved_for_pick(p.pick_side or "", float(line), float(closing_line))
        if moved is True:
            moves_for += 1
        elif moved is False:
            moves_against += 1

    return {
        "line_moves_for": moves_for,
        "line_moves_against": moves_against,
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "voids": voids,
        "win_rate_pct": _pct(wins, wins + losses),
        "total_pnl": round(total_pnl, 2),
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round((total_pnl / total_wagered * 100.0) if total_wagered else 0.0, 2),
        "avg_edge_pct": round(avg_edge * 100.0, 2),
        # Mean fair probability of the decided picks — the number a hit rate
        # must beat (for anytime TD "yes" picks it is far from 50%).
        "avg_fair_pct": round(avg_fair * 100.0, 1) if avg_fair is not None else None,
        "avg_clv_pct": round(avg_clv * 100.0, 2) if avg_clv is not None else None,
        "clv_hit_rate_pct": round(_pct(clv_hits, len(clv_picks)), 1) if clv_picks else None,
        "clv_sample": len(clv_picks),
    }


def get_graded_picks_detail(
    sport: str,
    since: date | None = None,
    until: date | None = None,
    graded_since: datetime | None = None,
    graded_until: datetime | None = None,
    on_card: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Return per-pick detail for graded picks, joined with Game for team names.

    on_card=True lists only the picks that made a card (see get_summary).
    """
    from betting_agent.db.models import Game
    with get_session() as session:
        q = (
            session.query(Pick, Game)
            .join(Game)
            .filter(Pick.result.isnot(None), Pick.sport == sport)
        )
        if since:
            q = q.filter(Pick.pick_date >= since)
        if until:
            q = q.filter(Pick.pick_date <= until)
        if graded_since is not None:
            q = q.filter(Pick.graded_at >= graded_since)
        if graded_until is not None:
            q = q.filter(Pick.graded_at <= graded_until)
        if on_card is not None:
            q = q.filter(Pick.on_card.is_(on_card))

        rows = q.all()

    return [
        {
            "pick_side": pick.pick_side,
            "bet_type": pick.bet_type,
            "odds": pick.odds,
            "result": pick.result,
            "pnl": pick.pnl or 0.0,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "player": pick.player,
            "market": pick.market,
            "line": pick.line,
            "clv": pick.clv,
            "closing_line": pick.closing_line,
            "on_card": bool(getattr(pick, "on_card", False)),
            "strategy": getattr(pick, "strategy", None),
        }
        for pick, game in rows
    ]


def get_breakdown_by_bet_type(
    sport: str | None = None,
    since: date | None = None,
    until: date | None = None,
    season: int | None = None,
    graded_since: datetime | None = None,
    graded_until: datetime | None = None,
    exclude_market: str | None = None,
    exclude_strategy: str | list[str] | tuple[str, ...] | None = None,
    on_card: bool | None = None,
) -> list[dict]:
    """Per-bet-type breakdown, scoped the same way as get_summary — a caller
    reporting the main book alone must be able to narrow this too, or the
    breakdown contradicts the headline above it."""
    bet_types = ["moneyline", "spread", "total", "prop"]
    rows = []
    for bt in bet_types:
        summary = get_summary(sport=sport, since=since, until=until, bet_type=bt, season=season,
                              graded_since=graded_since, graded_until=graded_until,
                              exclude_market=exclude_market, exclude_strategy=exclude_strategy,
                              on_card=on_card)
        if "total_bets" in summary and summary["total_bets"] > 0:
            rows.append({"bet_type": bt, **summary})
    return rows


def format_roi_report(
    sport: str | None = None,
    since: date | None = None,
    season: int | None = None,
    on_card: bool | None = None,
) -> str:
    """
    Human-readable ROI report.

    on_card scopes the report exactly as in get_summary: True = the picks that
    were offered on a card, False = the off-card evaluation set, None = both.
    Callers pass carded_only(sport) for a sport's normal scope.

    The headline is the MAIN book only. The ladder hits, the straight overs
    and the anytime-TD scorers each run on their own paper bankroll and get
    their own section below — one headline blending four books with different
    stakes and different purposes is not a record of anything.
    """
    from betting_agent.accounting.ledger import (
        LADDER_STRATEGY,
        OVERS_STRATEGY,
        SIDE_BOOKS,
        SIDE_MARKETS,
        ledger_summary,
    )
    from betting_agent.config import settings
    from betting_agent.sports.nfl.td_props import TD_MARKET

    main_scope = {"exclude_strategy": SIDE_BOOKS, "exclude_market": TD_MARKET}
    summary = get_summary(sport=sport, since=since, season=season, on_card=on_card,
                          **main_scope)
    if "message" in summary:
        return f"\n{summary['message']}\n"

    breakdown = get_breakdown_by_bet_type(sport=sport, since=since, season=season,
                                          on_card=on_card, **main_scope)

    lines = [
        "",
        "=" * 60,
        f"  ROI REPORT — Sport: {sport or 'ALL'} | Season: {season or 'ALL'}"
        + {True: "  (picks offered on a card)",
           False: "  (OFF-CARD picks — model evaluation, never offered)"}.get(on_card, ""),
        "=" * 60,
        f"  Total bets:   {summary['total_bets']}",
        f"  Record:       {summary['wins']}-{summary['losses']}-{summary['pushes']}"
        + (f" ({summary['voids']} void)" if summary.get("voids") else ""),
        f"  Win rate:     {summary['win_rate_pct']:.1f}%",
        f"  Total wagered: ${summary['total_wagered']:,.2f}",
        f"  Total P&L:    ${summary['total_pnl']:+,.2f}",
        f"  ROI:          {summary['roi_pct']:+.2f}%",
        f"  Avg Edge:     {summary['avg_edge_pct']:+.2f}%",
    ]
    if summary.get("avg_clv_pct") is not None:
        lines.append(f"  Avg CLV:      {summary['avg_clv_pct']:+.2f}%")
    if summary.get("clv_hit_rate_pct") is not None:
        lines.append(
            f"  CLV hit rate: {summary['clv_hit_rate_pct']:.1f}% "
            f"(beat the close on {summary['clv_sample']} priced picks)"
        )
    if summary.get("line_moves_for") or summary.get("line_moves_against"):
        lines.append(
            f"  Line moves:   {summary['line_moves_for']} for / "
            f"{summary['line_moves_against']} against (props whose number moved by close)"
        )

    ledger = ledger_summary(sport=sport, exclude_strategy=SIDE_BOOKS,
                            exclude_market=SIDE_MARKETS, on_card=on_card)
    if ledger["settled_picks"]:
        lines += [
            "",
            "  Bankroll:",
            "  " + "-" * 50,
            f"  Start ${ledger['starting_bankroll']:,.2f} → now ${ledger['current_bankroll']:,.2f} "
            f"({ledger['total_pnl']:+,.2f})",
            f"  Peak ${ledger['peak_equity']:,.2f}, max drawdown ${ledger['max_drawdown']:,.2f} "
            f"over {ledger['settled_picks']} settled picks",
        ]
    # The side books, each on its own bankroll. The TD board is addressed by
    # market rather than by strategy: its saved picks carry a NULL strategy
    # and the past is not rewritten to give them one.
    side_books: list[tuple[str, dict]] = [
        ("Straight overs", {"strategy": OVERS_STRATEGY}),
        ("Ladder hits", {"strategy": LADDER_STRATEGY}),
        ("TD scorers", {"market": TD_MARKET, "starting_bankroll": settings.td_bankroll}),
    ]
    for label, scope in side_books:
        side = ledger_summary(sport=sport, on_card=on_card, **scope)
        if side["settled_picks"]:
            lines += [
                "",
                f"  {label} (own bankroll):",
                "  " + "-" * 50,
                f"  Start ${side['starting_bankroll']:,.2f} → now ${side['current_bankroll']:,.2f} "
                f"({side['total_pnl']:+,.2f}) over {side['settled_picks']} settled picks, "
                f"max drawdown ${side['max_drawdown']:,.2f}",
            ]

    if breakdown:
        lines.append("")
        lines.append("  By Bet Type:")
        lines.append("  " + "-" * 50)
        for row in breakdown:
            lines.append(
                f"  {row['bet_type'].upper():<12} "
                f"{row['wins']}-{row['losses']}  "
                f"WR={row['win_rate_pct']:.1f}%  "
                f"ROI={row['roi_pct']:+.1f}%"
            )

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)
