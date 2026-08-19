"""
ROI and performance reporting.
"""

from __future__ import annotations

from datetime import date
from typing import Any


from betting_agent.db.models import Pick
from betting_agent.db.session import get_session


def _pct(num: int, denom: int) -> float:
    return (num / denom * 100.0) if denom else 0.0


def get_summary(
    sport: str | None = None,
    since: date | None = None,
    until: date | None = None,
    bet_type: str | None = None,
    season: int | None = None,
) -> dict[str, Any]:
    """
    Aggregate ROI report.
    Returns dict with: total_bets, wins, losses, pushes, win_rate, total_pnl,
                       total_wagered, roi_pct, avg_edge, avg_clv.
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
        if bet_type:
            q = q.filter(Pick.bet_type == bet_type)
        if season:
            q = q.filter(Game.season == season)

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
    clv_picks = [p for p in picks if p.clv is not None]
    avg_clv = sum(p.clv for p in clv_picks) / len(clv_picks) if clv_picks else None
    clv_hits = sum(1 for p in clv_picks if p.clv > 0)

    return {
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
        "avg_clv_pct": round(avg_clv * 100.0, 2) if avg_clv is not None else None,
        "clv_hit_rate_pct": round(_pct(clv_hits, len(clv_picks)), 1) if clv_picks else None,
        "clv_sample": len(clv_picks),
    }


def get_graded_picks_detail(
    sport: str,
    since: date | None = None,
    until: date | None = None,
) -> list[dict[str, Any]]:
    """
    Return per-pick detail for graded picks, joined with Game for team names.
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
        }
        for pick, game in rows
    ]


def get_breakdown_by_bet_type(
    sport: str | None = None,
    since: date | None = None,
    until: date | None = None,
    season: int | None = None,
) -> list[dict]:
    """Per-bet-type breakdown."""
    bet_types = ["moneyline", "spread", "total", "prop"]
    rows = []
    for bt in bet_types:
        summary = get_summary(sport=sport, since=since, until=until, bet_type=bt, season=season)
        if "total_bets" in summary and summary["total_bets"] > 0:
            rows.append({"bet_type": bt, **summary})
    return rows


def format_roi_report(
    sport: str | None = None,
    since: date | None = None,
    season: int | None = None,
) -> str:
    """Human-readable ROI report."""
    summary = get_summary(sport=sport, since=since, season=season)
    if "message" in summary:
        return f"\n{summary['message']}\n"

    breakdown = get_breakdown_by_bet_type(sport=sport, since=since, season=season)

    lines = [
        "",
        "=" * 60,
        f"  ROI REPORT — Sport: {sport or 'ALL'} | Season: {season or 'ALL'}",
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

    from betting_agent.accounting.ledger import ledger_summary
    ledger = ledger_summary(sport=sport)
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
