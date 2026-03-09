"""
Discord webhook integration for picks and grading results.

Sends structured embeds to sport-specific channels via webhook URLs
resolved dynamically from environment variables:
    DISCORD_WEBHOOK_{SPORT}_{PICKS|RESULTS}

Follows the Ollama pattern: check if configured, try/except, log warnings,
never crash the pipeline.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date
from typing import Any

import requests

from betting_agent.intelligence.picks import BetCandidate, _pick_label, _confidence_stars

logger = logging.getLogger(__name__)

# Discord embed color constants
COLOR_BLUE = 0x3498DB    # summary headers
COLOR_GREEN = 0x2ECC71   # positive P&L / picks
COLOR_RED = 0xE74C3C     # negative P&L
COLOR_GREY = 0x95A5A6    # no data

# Discord allows max 10 embeds per message
MAX_EMBEDS_PER_MESSAGE = 10

WEBHOOK_TIMEOUT = 10  # seconds

SPORT_EMOJI = {"NFL": "\U0001f3c8", "NBA": "\U0001f3c0", "NHL": "\U0001f3d2", "MLB": "\u26be"}


def _get_webhook_url(sport: str, channel_type: str) -> str | None:
    """
    Resolve webhook URL from environment variable.
    Looks up DISCORD_WEBHOOK_{SPORT}_{CHANNEL_TYPE}.

    Args:
        sport: Sport name (e.g. "NFL", "NBA")
        channel_type: "PICKS" or "RESULTS"

    Returns:
        Webhook URL string, or None if not configured.
    """
    key = f"DISCORD_WEBHOOK_{sport.upper()}_{channel_type.upper()}"
    return os.environ.get(key) or None


def is_discord_configured(sport: str, channel_type: str) -> bool:
    """Check if a Discord webhook is configured for this sport/channel."""
    return _get_webhook_url(sport, channel_type) is not None


def _send_webhook(url: str, payload: dict) -> bool:
    """
    POST JSON payload to a Discord webhook URL.
    Retries once on 429 (rate limit). Returns True on success.
    """
    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=WEBHOOK_TIMEOUT)
            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 1.0)
                logger.warning("Discord rate-limited, retrying after %.1fs", retry_after)
                time.sleep(min(retry_after, 5.0))
                continue
            if resp.status_code >= 400:
                logger.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
                return False
            return True
        except requests.RequestException as exc:
            logger.warning("Discord webhook request failed: %s", exc)
            return False
    return False


def _build_pick_embed(
    pick: BetCandidate,
    rank: int,
    star_thresholds: tuple[float, float, float, float] = (0.04, 0.07, 0.12, 0.20),
    analysis: dict | None = None,
) -> dict:
    """Build a Discord embed dict for a single pick."""
    label = _pick_label(pick)
    stars = _confidence_stars(pick.edge, star_thresholds)
    matchup = f"{pick.away_team} @ {pick.home_team}"

    desc = (
        f"{matchup}\n\n"
        f"**Odds:** `{pick.odds:+d}`  |  **Edge:** `{pick.edge:+.1%}`  |  {stars}\n"
        f"**Model:** `{pick.model_prob:.1%}`  vs  **Market:** `{pick.implied_prob:.1%}`\n"
        f"**Kelly:** `{pick.kelly_fraction:.2%}`  \u2192  **Bet:** `${pick.recommended_bet:.2f}`"
    )

    if analysis and analysis.get("key_factors"):
        factors = "\n".join(f"- {f}" for f in analysis["key_factors"])
        desc += f"\n\n**Key Factors:**\n{factors}"

    return {
        "title": f"#{rank}  {label}",
        "description": desc,
        "color": COLOR_GREEN,
    }


def _build_summary_embed(
    candidates: list[BetCandidate], bankroll: float, sport: str
) -> dict:
    """Build a header embed summarizing today's picks."""
    total_action = sum(c.recommended_bet for c in candidates)
    pct_bankroll = (total_action / bankroll * 100) if bankroll > 0 else 0
    pick_date = candidates[0].game_date if candidates else date.today()
    emoji = SPORT_EMOJI.get(sport.upper(), "")

    desc = (
        f"{pick_date}\n\n"
        f"**Bankroll:** ${bankroll:,.2f}  |  "
        f"**{len(candidates)} Picks**  |  "
        f"**Action:** ${total_action:.2f} ({pct_bankroll:.1f}%)"
    )

    return {
        "title": f"{emoji} Picks of the Day \u2014 {sport}",
        "description": desc,
        "color": COLOR_BLUE,
    }


def send_picks_to_discord(
    candidates: list[BetCandidate], bankroll: float, sport: str
) -> bool:
    """
    Send today's picks to the Discord picks channel for the given sport.

    Builds a summary embed + one embed per pick, splitting into multiple
    messages if there are more than 10 embeds (Discord limit).

    Returns True if all messages sent successfully, False otherwise.
    """
    url = _get_webhook_url(sport, "PICKS")
    if not url:
        logger.debug("Discord not configured for %s picks, skipping", sport)
        return False

    if not candidates:
        logger.debug("No picks to send to Discord")
        return True

    # Build all embeds: summary header + one per pick
    from betting_agent.sports.registry import get_sport_config
    star_thresholds = get_sport_config(sport).star_thresholds

    embeds = [_build_summary_embed(candidates, bankroll, sport)]
    ranked = sorted(candidates, key=lambda c: c.edge, reverse=True)
    for rank, pick in enumerate(ranked, 1):
        analysis = pick.extra.get("analysis") if pick.extra else None
        embeds.append(_build_pick_embed(pick, rank, star_thresholds, analysis=analysis))

    # Split into chunks of MAX_EMBEDS_PER_MESSAGE
    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        chunk = embeds[i : i + MAX_EMBEDS_PER_MESSAGE]
        payload: dict[str, Any] = {"embeds": chunk}
        if not _send_webhook(url, payload):
            all_ok = False

    return all_ok


def _build_results_embed(
    summary: dict[str, Any],
    sport: str,
    graded_date: date | None = None,
    pick_details: list[dict] | None = None,
) -> dict:
    """Build a Discord embed for grading results."""
    if "message" in summary:
        return {
            "title": f"Results \u2014 {sport}",
            "description": summary["message"],
            "color": COLOR_GREY,
        }

    pnl = summary.get("total_pnl", 0)
    if pnl > 0:
        color = COLOR_GREEN
    elif pnl < 0:
        color = COLOR_RED
    else:
        color = COLOR_GREY

    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    pushes = summary.get("pushes", 0)
    win_rate = summary.get("win_rate_pct", 0)
    roi = summary.get("roi_pct", 0)
    avg_edge = summary.get("avg_edge_pct", 0)

    lines = [
        f"**Record:** {wins}-{losses}-{pushes} ({win_rate:.1f}%)  |  "
        f"**P&L:** {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}  |  **ROI:** {roi:+.2f}%",
        f"**Avg Edge:** {avg_edge:+.2f}%",
    ]

    if summary.get("avg_clv_pct") is not None:
        lines[-1] += f"  |  **Avg CLV:** {summary['avg_clv_pct']:+.2f}%"

    if pick_details:
        lines.append("")
        lines.append("**Picks:**")
        for d in pick_details:
            result_tag = d["result"].upper()
            pnl_val = d["pnl"]
            pnl_str = f"+${pnl_val:,.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):,.2f}"
            matchup = f"{d['away_team']} @ {d['home_team']}"
            odds_str = f"{d['odds']:+d}" if d["odds"] else ""
            lines.append(
                f"`{result_tag}`  {d['pick_side']} {d['bet_type'].title()} ({odds_str}) "
                f"\u2014 {matchup} \u2014 {pnl_str}"
            )

    date_str = str(graded_date) if graded_date else ""
    desc = f"{date_str}\n\n" + "\n".join(lines) if date_str else "\n".join(lines)

    return {
        "title": f"Results \u2014 {sport}",
        "description": desc,
        "color": color,
    }


def _build_breakdown_embed(breakdown: list[dict], sport: str) -> dict:
    """Build an embed showing per-bet-type breakdown."""
    lines = []
    for row in breakdown:
        bt = row.get("bet_type", "").upper()
        wins = row.get("wins", 0)
        losses = row.get("losses", 0)
        wr = row.get("win_rate_pct", 0)
        roi = row.get("roi_pct", 0)
        lines.append(f"**{bt}** — {wins}-{losses} | WR={wr:.1f}% | ROI={roi:+.1f}%")

    return {
        "title": f"Breakdown by Bet Type — {sport}",
        "description": "\n".join(lines) if lines else "No data",
        "color": COLOR_BLUE,
    }


def send_results_to_discord(
    summary: dict[str, Any],
    sport: str,
    breakdown: list[dict] | None = None,
    graded_date: date | None = None,
    pick_details: list[dict] | None = None,
) -> bool:
    """
    Send grading results to the Discord results channel for the given sport.

    Returns True if sent successfully, False otherwise.
    """
    url = _get_webhook_url(sport, "RESULTS")
    if not url:
        logger.debug("Discord not configured for %s results, skipping", sport)
        return False

    if "total_bets" not in summary:
        logger.debug("No graded picks for %s, skipping Discord", sport)
        return True

    embeds = [_build_results_embed(summary, sport, graded_date, pick_details=pick_details)]
    if breakdown:
        embeds.append(_build_breakdown_embed(breakdown, sport))

    payload: dict[str, Any] = {"embeds": embeds}
    return _send_webhook(url, payload)


def _build_alltime_sport_embed(
    summary: dict[str, Any], sport: str, starting_bankroll: float
) -> dict:
    """Build an embed for one sport's all-time record + current bankroll."""
    if "message" in summary:
        return {
            "title": sport,
            "description": "No graded picks yet",
            "color": COLOR_GREY,
        }

    pnl = summary.get("total_pnl", 0)
    current_bankroll = starting_bankroll + pnl
    if pnl > 0:
        color = COLOR_GREEN
    elif pnl < 0:
        color = COLOR_RED
    else:
        color = COLOR_GREY

    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    pushes = summary.get("pushes", 0)
    win_rate = summary.get("win_rate_pct", 0)
    roi = summary.get("roi_pct", 0)

    lines = [
        f"**Record:** {wins}-{losses}-{pushes} ({win_rate:.1f}%)  |  **ROI:** {roi:+.2f}%",
        f"**Bankroll:** ${starting_bankroll:,.2f} \u2192 ${current_bankroll:,.2f}  |  "
        f"**P&L:** {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}",
    ]

    extra_parts = []
    if summary.get("avg_edge_pct") is not None:
        extra_parts.append(f"**Avg Edge:** {summary['avg_edge_pct']:+.2f}%")
    if summary.get("avg_clv_pct") is not None:
        extra_parts.append(f"**Avg CLV:** {summary['avg_clv_pct']:+.2f}%")
    if extra_parts:
        lines.append("  |  ".join(extra_parts))

    return {
        "title": sport,
        "description": "\n".join(lines),
        "color": color,
    }


def _aggregate_sport_summaries(
    summaries: dict[str, dict[str, Any]],
    total_starting_bankroll: float,
) -> dict[str, Any]:
    """Combine multiple sport summaries into a single aggregate."""
    total_bets = sum(s.get("total_bets", 0) for s in summaries.values())
    if total_bets == 0:
        return {"message": "No graded picks found"}

    wins = sum(s.get("wins", 0) for s in summaries.values())
    losses = sum(s.get("losses", 0) for s in summaries.values())
    pushes = sum(s.get("pushes", 0) for s in summaries.values())
    total_pnl = sum(s.get("total_pnl", 0) for s in summaries.values())
    total_wagered = sum(s.get("total_wagered", 0) for s in summaries.values())

    # Weighted average of edge across all picks
    weighted_edge = sum(
        s.get("avg_edge_pct", 0) * s.get("total_bets", 0) for s in summaries.values()
    )
    avg_edge = weighted_edge / total_bets

    # Weighted average CLV (only from sports that have it)
    clv_bets = sum(
        s.get("total_bets", 0)
        for s in summaries.values()
        if s.get("avg_clv_pct") is not None
    )
    avg_clv = None
    if clv_bets > 0:
        weighted_clv = sum(
            s["avg_clv_pct"] * s.get("total_bets", 0)
            for s in summaries.values()
            if s.get("avg_clv_pct") is not None
        )
        avg_clv = weighted_clv / clv_bets

    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate_pct": (wins / (wins + losses) * 100.0) if (wins + losses) else 0.0,
        "total_pnl": round(total_pnl, 2),
        "total_wagered": round(total_wagered, 2),
        "roi_pct": round(
            (total_pnl / total_starting_bankroll * 100.0)
            if total_starting_bankroll
            else 0.0,
            2,
        ),
        "avg_edge_pct": round(avg_edge, 2),
        "avg_clv_pct": round(avg_clv, 2) if avg_clv is not None else None,
    }


def send_alltime_to_discord(
    sport_summaries: dict[str, dict[str, Any]],
    starting_bankroll: float,
    as_of_date: date | None = None,
) -> bool:
    """
    Send all-time results for all sports to the alltime-results channel.

    Args:
        sport_summaries: mapping of sport name → get_summary() result (no date filter).
        starting_bankroll: the original bankroll each sport started with.
        as_of_date: the date to display; defaults to today.

    Uses DISCORD_WEBHOOK_ALLTIME_RESULTS env var.
    Returns True if sent successfully, False otherwise.
    """
    url = _get_webhook_url("ALLTIME", "RESULTS")
    if not url:
        logger.debug("Discord not configured for alltime results, skipping")
        return False

    if not sport_summaries:
        return True

    header: dict[str, Any] = {
        "title": "All-Time Results",
        "description": str(as_of_date or date.today()),
        "color": COLOR_BLUE,
    }

    # Only include sports that have actual graded data
    active_summaries = {
        sport: summary
        for sport, summary in sport_summaries.items()
        if "total_bets" in summary
    }
    if not active_summaries:
        logger.debug("No sports have graded picks, skipping alltime Discord")
        return True

    embeds = [header]

    # Combined "All Sports" embed above individual sports. The aggregate bankroll
    # should reflect one bankroll allocation per sport with graded picks.
    combined_starting_bankroll = starting_bankroll * len(active_summaries)
    combined = _aggregate_sport_summaries(active_summaries, combined_starting_bankroll)
    embeds.append(
        _build_alltime_sport_embed(
            combined,
            "🏆 All Sports",
            combined_starting_bankroll,
        )
    )

    for sport, summary in sorted(active_summaries.items()):
        emoji = SPORT_EMOJI.get(sport.upper(), "")
        embeds.append(_build_alltime_sport_embed(summary, f"{emoji} {sport}" if emoji else sport, starting_bankroll))

    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        chunk = embeds[i : i + MAX_EMBEDS_PER_MESSAGE]
        payload: dict[str, Any] = {"embeds": chunk}
        if not _send_webhook(url, payload):
            all_ok = False

    return all_ok
