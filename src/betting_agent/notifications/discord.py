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


def _build_pick_embed(pick: BetCandidate, rank: int) -> dict:
    """Build a Discord embed dict for a single pick."""
    label = _pick_label(pick)
    stars = _confidence_stars(pick.edge)
    matchup = f"{pick.away_team} @ {pick.home_team}"

    fields = [
        {"name": "Odds", "value": f"`{pick.odds:+d}`", "inline": True},
        {"name": "Edge", "value": f"`{pick.edge:+.1%}`", "inline": True},
        {"name": "Confidence", "value": stars, "inline": True},
        {"name": "Model Prob", "value": f"`{pick.model_prob:.1%}`", "inline": True},
        {"name": "Implied Prob", "value": f"`{pick.implied_prob:.1%}`", "inline": True},
        {"name": "Kelly", "value": f"`{pick.kelly_fraction:.2%}`", "inline": True},
        {"name": "Bet Size", "value": f"`${pick.recommended_bet:.2f}`", "inline": True},
    ]

    return {
        "title": f"#{rank}  {label}",
        "description": matchup,
        "color": COLOR_GREEN,
        "fields": fields,
    }


def _build_summary_embed(
    candidates: list[BetCandidate], bankroll: float, sport: str
) -> dict:
    """Build a header embed summarizing today's picks."""
    total_action = sum(c.recommended_bet for c in candidates)
    pct_bankroll = (total_action / bankroll * 100) if bankroll > 0 else 0
    pick_date = candidates[0].game_date if candidates else date.today()

    return {
        "title": f"Picks of the Day — {sport}",
        "description": str(pick_date),
        "color": COLOR_BLUE,
        "fields": [
            {"name": "Bankroll", "value": f"`${bankroll:,.2f}`", "inline": True},
            {"name": "Picks", "value": f"`{len(candidates)}`", "inline": True},
            {
                "name": "Total Action",
                "value": f"`${total_action:.2f} ({pct_bankroll:.1f}%)`",
                "inline": True,
            },
        ],
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
    embeds = [_build_summary_embed(candidates, bankroll, sport)]
    ranked = sorted(candidates, key=lambda c: c.edge, reverse=True)
    for rank, pick in enumerate(ranked, 1):
        embeds.append(_build_pick_embed(pick, rank))

    # Split into chunks of MAX_EMBEDS_PER_MESSAGE
    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        chunk = embeds[i : i + MAX_EMBEDS_PER_MESSAGE]
        payload: dict[str, Any] = {"embeds": chunk}
        if not _send_webhook(url, payload):
            all_ok = False

    return all_ok


def _build_results_embed(
    summary: dict[str, Any], sport: str, graded_date: date | None = None
) -> dict:
    """Build a Discord embed for grading results."""
    if "message" in summary:
        return {
            "title": f"Results — {sport}",
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

    fields = [
        {"name": "Record", "value": f"`{wins}-{losses}-{pushes}`", "inline": True},
        {"name": "Win Rate", "value": f"`{summary.get('win_rate_pct', 0):.1f}%`", "inline": True},
        {"name": "P&L", "value": f"`${pnl:+,.2f}`", "inline": True},
        {"name": "ROI", "value": f"`{summary.get('roi_pct', 0):+.2f}%`", "inline": True},
        {"name": "Avg Edge", "value": f"`{summary.get('avg_edge_pct', 0):+.2f}%`", "inline": True},
    ]

    if summary.get("avg_clv_pct") is not None:
        fields.append(
            {"name": "Avg CLV", "value": f"`{summary['avg_clv_pct']:+.2f}%`", "inline": True}
        )

    desc = str(graded_date) if graded_date else ""

    return {
        "title": f"Results — {sport}",
        "description": desc,
        "color": color,
        "fields": fields,
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

    embeds = [_build_results_embed(summary, sport, graded_date)]
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

    fields = [
        {"name": "Record", "value": f"`{wins}-{losses}-{pushes}`", "inline": True},
        {"name": "Win Rate", "value": f"`{summary.get('win_rate_pct', 0):.1f}%`", "inline": True},
        {"name": "ROI", "value": f"`{summary.get('roi_pct', 0):+.2f}%`", "inline": True},
        {"name": "Starting Bankroll", "value": f"`${starting_bankroll:,.2f}`", "inline": True},
        {"name": "Current Bankroll", "value": f"`${current_bankroll:,.2f}`", "inline": True},
        {"name": "All-Time P&L", "value": f"`${pnl:+,.2f}`", "inline": True},
    ]

    if summary.get("avg_edge_pct") is not None:
        fields.append(
            {"name": "Avg Edge", "value": f"`{summary['avg_edge_pct']:+.2f}%`", "inline": True}
        )
    if summary.get("avg_clv_pct") is not None:
        fields.append(
            {"name": "Avg CLV", "value": f"`{summary['avg_clv_pct']:+.2f}%`", "inline": True}
        )

    return {
        "title": sport,
        "color": color,
        "fields": fields,
    }


def send_alltime_to_discord(
    sport_summaries: dict[str, dict[str, Any]],
    starting_bankroll: float,
) -> bool:
    """
    Send all-time results for all sports to the alltime-results channel.

    Args:
        sport_summaries: mapping of sport name → get_summary() result (no date filter).
        starting_bankroll: the original bankroll each sport started with.

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
        "description": str(date.today()),
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
    for sport, summary in sorted(active_summaries.items()):
        embeds.append(_build_alltime_sport_embed(summary, sport, starting_bankroll))

    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        chunk = embeds[i : i + MAX_EMBEDS_PER_MESSAGE]
        payload: dict[str, Any] = {"embeds": chunk}
        if not _send_webhook(url, payload):
            all_ok = False

    return all_ok
