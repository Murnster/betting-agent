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

from betting_agent.intelligence.picks import (
    BetCandidate,
    _confidence_stars,
    _market_label,
    _pick_label,
)
from betting_agent.sports.nfl.props import ladder_label, over_label
from betting_agent.sports.nfl.td_props import TD_MARKET

LADDER_STRATEGY = "ladder"
OVERS_STRATEGY = "overs"
PARLAY_STRATEGIES = ("parlay", "parlay_leg")

logger = logging.getLogger(__name__)

# Discord embed color constants
COLOR_BLUE = 0x3498DB    # summary headers
COLOR_GREEN = 0x2ECC71   # positive P&L / picks
COLOR_RED = 0xE74C3C     # negative P&L
COLOR_GREY = 0x95A5A6    # no data
COLOR_ORANGE = 0xE67E22  # straight-overs section
COLOR_PURPLE = 0x9B59B6  # parlays — their own channel, never the card's green

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
        channel_type: "PICKS", "RESULTS", "EXTRAS" or "EXTRAS_RESULTS"

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

    edge_text = f"{pick.edge:+.1%}"
    if pick.original_edge is not None and abs(pick.original_edge - pick.edge) > 1e-9:
        edge_text = f"{pick.original_edge:+.1%} -> {pick.edge:+.1%}"

    desc = (
        f"{matchup}\n\n"
        f"**Odds:** `{pick.odds:+d}`  |  **Edge:** `{edge_text}`  |  {stars}\n"
        f"**Model:** `{pick.model_prob:.1%}`  vs  **Market:** `{pick.implied_prob:.1%}`\n"
        f"**Kelly:** `{pick.kelly_fraction:.2%}`  \u2192  **Bet:** `${pick.recommended_bet:.2f}`"
    )

    if pick.agent_verdict and pick.agent_verdict != "SKIPPED":
        shadow = bool((pick.extra or {}).get("agent", {}).get("shadow"))
        desc += f"\n**Verdict:** `{pick.agent_verdict}`" + (" (shadow)" if shadow else "")

    reasons = pick.agent_reasons or []
    if reasons:
        desc += "\n**Why:** " + "; ".join(reasons[:2])

    if analysis and analysis.get("key_factors"):
        factors = "\n".join(f"- {f}" for f in analysis["key_factors"])
        desc += f"\n\n**Key Factors:**\n{factors}"

    return {
        "title": f"#{rank}  {label}",
        "description": desc,
        "color": COLOR_GREEN,
    }


def _build_summary_embed(
    candidates: list[BetCandidate], bankroll: float, sport: str, agent_summary: dict | None = None
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
    if agent_summary and (
        agent_summary.get("validated_games")
        or agent_summary.get("skipped_games")
        or agent_summary.get("total_cost_usd")
    ):
        label = "Validator (SHADOW)" if agent_summary.get("shadow") else "Validator"
        desc += (
            "\n"
            f"**{label}:** {agent_summary.get('validated_games', 0)} games  |  "
            f"**Skipped:** {agent_summary.get('skipped_games', 0)}  |  "
            f"**Cost:** ${agent_summary.get('total_cost_usd', 0.0):.4f}"
        )

    return {
        "title": f"{emoji} Picks of the Day \u2014 {sport}",
        "description": desc,
        "color": COLOR_BLUE,
    }


def send_picks_to_discord(
    candidates: list[BetCandidate], bankroll: float, sport: str, agent_summary: dict | None = None
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

    embeds = [_build_summary_embed(candidates, bankroll, sport, agent_summary=agent_summary)]
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


def _build_lean_embed(pick: BetCandidate, rank: int) -> dict:
    """A game-market lean: labelled as such, never as a pick."""
    label = _pick_label(pick)
    matchup = f"{pick.away_team} @ {pick.home_team}"
    ref = (pick.extra or {}).get("reference", "reference")
    book = (pick.extra or {}).get("bookmaker", "")
    desc = (
        f"{matchup}\n\n"
        f"**Odds:** `{pick.odds:+d}` at {book}  |  **Edge vs {ref}:** `{pick.edge:+.1%}`\n"
        f"**Fair ({ref}):** `{pick.model_prob:.1%}`  vs  **{book}:** `{pick.implied_prob:.1%}`\n"
        f"_Market lean, not a pick — no model beats the NFL close. Saved on paper "
        f"to measure whether early numbers beat closing ones (CLV)._"
    )
    return {"title": f"LEAN #{rank}  {label}", "description": desc, "color": COLOR_GREY}


def send_slate_to_discord(
    title: str, leans: list[BetCandidate], props: list[BetCandidate], bankroll: float,
    sport: str = "NFL", agent_summary: dict | None = None, extra_saved: int = 0,
    td_scorers: list[BetCandidate] | None = None, td_saved: int = 0,
    td_bankroll: float | None = None,
    ladder: list[BetCandidate] | None = None, ladder_saved: int = 0,
    ladder_bankroll: float | None = None,
    overs: list[BetCandidate] | None = None, overs_saved: int = 0,
    overs_bankroll: float | None = None,
) -> bool:
    """
    One card per slate: header, the game lean(s), the prop picks, then the
    three side books, each behind its own divider and on its own paper
    bankroll — the anytime-TD scorers, the straight overs and the ladder hits.
    Only the props are the tracked picks; nothing below the first divider
    touches the main record or the main bankroll, which is why nothing below
    it is green. `extra_saved` / `td_saved` / `overs_saved` / `ladder_saved` =
    picks that cleared the floors but did not make the card. `td_scorers` are
    labelled PICK or LEAN by edge.
    """
    url = _get_webhook_url(sport, "PICKS")
    if not url:
        logger.debug("Discord not configured for %s picks, skipping", sport)
        return False
    if not leans and not props and not td_scorers and not ladder and not overs:
        return True

    from betting_agent.sports.registry import get_sport_config
    star_thresholds = get_sport_config(sport).star_thresholds

    header = _build_summary_embed(props, bankroll, sport, agent_summary=agent_summary) if props else {
        "description": f"{date.today()}\n\nNo prop clears the floors on this slate.", "color": COLOR_BLUE,
    }
    header["title"] = f"{SPORT_EMOJI.get(sport.upper(), '')} {title}"
    if extra_saved:
        header["description"] += f"\n_{extra_saved} more paper pick(s) saved off-card._"
    embeds = [header]
    for rank, lean in enumerate(sorted(leans, key=lambda c: c.edge, reverse=True), 1):
        embeds.append(_build_lean_embed(lean, rank))
    for rank, pick in enumerate(sorted(props, key=lambda c: c.edge, reverse=True), 1):
        embeds.append(_build_pick_embed(pick, rank, star_thresholds,
                                        analysis=(pick.extra or {}).get("analysis")))
    if td_scorers:
        embeds.append(_build_td_header_embed(td_scorers, td_bankroll, td_saved))
        for rank, td in enumerate(sorted(td_scorers, key=lambda c: c.edge, reverse=True), 1):
            embeds.append(_build_td_embed(td, rank))
    if overs:
        embeds.append(_build_overs_header_embed(overs, overs_bankroll, overs_saved))
        for rank, pick in enumerate(sorted(overs, key=lambda c: c.edge, reverse=True), 1):
            embeds.append(_build_over_embed(pick, rank))
    if ladder:
        embeds.append(_build_ladder_header_embed(ladder, ladder_bankroll, ladder_saved))
        for rank, pick in enumerate(sorted(ladder, key=lambda c: c.edge, reverse=True), 1):
            embeds.append(_build_ladder_embed(pick, rank))

    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        if not _send_webhook(url, {"embeds": embeds[i: i + MAX_EMBEDS_PER_MESSAGE]}):
            all_ok = False
    return all_ok


def _build_td_header_embed(td_scorers: list[BetCandidate], bankroll: float | None,
                           saved_off_card: int) -> dict:
    """Sub-section divider for the anytime-TD scorers: their own paper book."""
    total = sum(c.recommended_bet for c in td_scorers)
    desc = ("The best edge on the book's Yes board, de-vigged to the touchdowns the "
            "market expects from the game. Shadow board with its own paper bankroll — "
            "never mixed with the picks above.")
    if bankroll is not None:
        desc += f"\n**TD bankroll:** `${bankroll:,.2f}`  |  **Action:** `${total:,.2f}`"
    if saved_off_card:
        desc += f"\n_{saved_off_card} more TD pick(s) saved off-card._"
    return {"title": f"TD SCORERS \u2014 {len(td_scorers)} on card", "description": desc,
            "color": COLOR_BLUE}


def _build_td_embed(pick: BetCandidate, rank: int) -> dict:
    """
    The slate's anytime-TD scorer: the best edge on the book's Yes board once
    it is de-vigged to the market's expected TDs. A PICK (paper stake from the
    TD bankroll) when the edge sits inside the window, otherwise a LEAN at
    stake 0 — shown either way, like the game lean, and tracked for hit rate
    vs fair. Never green: green is the main card's picks and a TD scorer has
    never been one of those.
    """
    extra = pick.extra or {}
    is_pick = bool(extra.get("td_pick"))
    matchup = f"{pick.away_team} @ {pick.home_team}"
    book = extra.get("bookmaker", "")
    hold = extra.get("board_hold")
    desc = (
        f"{matchup}\n\n"
        f"**Odds:** `{pick.odds:+d}` at {book}  |  **Edge:** `{pick.edge:+.1%}`\n"
        f"**Model:** `{pick.model_prob:.1%}` to score  vs  **Fair:** `{pick.implied_prob:.1%}`"
        + (f" (board hold {hold:.0%})" if hold is not None else "") + "\n"
    )
    if is_pick:
        desc += (f"**Kelly:** `{pick.kelly_fraction:.2%}`  \u2192  **Bet:** `${pick.recommended_bet:.2f}` "
                 "_(paper — TD bankroll, anytime TD is in shadow)_")
    else:
        desc += "_Lean, not a pick: edge below the TD floor. Stake 0, tracked for hit rate and CLV._"
    for flag in extra.get("flags", []):
        desc += f"\n\u26a0 {flag.get('detail', '')}"
    if pick.agent_verdict and pick.agent_verdict != "SKIPPED":
        shadow = bool(extra.get("agent", {}).get("shadow"))
        desc += f"\n**Verdict:** `{pick.agent_verdict}`" + (" (shadow)" if shadow else "")
    if pick.agent_reasons:
        desc += "\n**Why:** " + "; ".join(pick.agent_reasons[:2])
    tag = "PICK" if is_pick else "LEAN"
    return {"title": f"TD SCORER #{rank}  {pick.player} anytime TD ({tag})",
            "description": desc[:4000], "color": COLOR_BLUE if is_pick else COLOR_GREY}


def _build_ladder_header_embed(ladder: list[BetCandidate], bankroll: float | None,
                               saved_off_card: int) -> dict:
    """Sub-section divider for the ladder hits: its own paper book."""
    total = sum(c.recommended_bet for c in ladder)
    desc = ("Best overs — the player reaching a milestone, priced on the alternate boards. "
            "Experimental pick pool: every entry is staked from its own paper bankroll, "
            "never mixed with the picks above.")
    if bankroll is not None:
        desc += f"\n**Ladder bankroll:** `${bankroll:,.2f}`  |  **Action:** `${total:,.2f}`"
    if saved_off_card:
        desc += f"\n_{saved_off_card} more ladder pick(s) saved off-card._"
    return {"title": f"LADDER HITS \u2014 {len(ladder)} on card", "description": desc,
            "color": COLOR_BLUE}


def _build_ladder_embed(pick: BetCandidate, rank: int) -> dict:
    """One ladder rung: 'A.J. Brown 60+ receiving yds' with its own stake."""
    extra = pick.extra or {}
    matchup = f"{pick.away_team} @ {pick.home_team}"
    book = extra.get("bookmaker", "")
    desc = (
        f"{matchup}\n\n"
        f"**Odds:** `{pick.odds:+d}` at {book}  |  **Edge:** `{pick.edge:+.1%}`\n"
        f"**Model:** `{pick.model_prob:.1%}` to hit  vs  **Fair:** `{pick.implied_prob:.1%}`"
    )
    mean = extra.get("projection_mean")
    if mean is not None:
        desc += f"  |  **Proj:** `{mean:g}`"
    desc += (f"\n**Kelly:** `{pick.kelly_fraction:.2%}`  \u2192  **Bet:** `${pick.recommended_bet:.2f}` "
             "_(paper — ladder bankroll)_")
    for flag in extra.get("flags", []):
        desc += f"\n\u26a0 {flag.get('detail', '')}"
    if pick.agent_verdict and pick.agent_verdict != "SKIPPED":
        shadow = bool(extra.get("agent", {}).get("shadow"))
        desc += f"\n**Verdict:** `{pick.agent_verdict}`" + (" (shadow)" if shadow else "")
    if pick.agent_reasons:
        desc += "\n**Why:** " + "; ".join(pick.agent_reasons[:2])
    return {"title": f"LADDER #{rank}  {ladder_label(pick.player, pick.market, pick.line)}",
            "description": desc[:4000], "color": COLOR_BLUE}


def _build_overs_header_embed(overs: list[BetCandidate], bankroll: float | None,
                              saved_off_card: int) -> dict:
    """Sub-section divider for the straight overs: its own paper book."""
    total = sum(c.recommended_bet for c in overs)
    n_flat = sum(1 for c in overs if (c.extra or {}).get("flat_stake"))
    desc = ("Straight overs — the book's main-line Over, priced by the ladder projection. "
            "Experimental pick pool: the game's best over is always here, staked from its "
            "own paper bankroll, never mixed with the picks above.")
    if n_flat:
        desc += (f"\n{n_flat} at a flat 1% stake: the model has no edge on any over in that "
                 "game, the selection is tracked anyway.")
    if bankroll is not None:
        desc += f"\n**Overs bankroll:** `${bankroll:,.2f}`  |  **Action:** `${total:,.2f}`"
    if saved_off_card:
        desc += f"\n_{saved_off_card} more over(s) saved off-card._"
    return {"title": f"STRAIGHT OVERS \u2014 {len(overs)} on card", "description": desc,
            "color": COLOR_ORANGE}


def _build_over_embed(pick: BetCandidate, rank: int) -> dict:
    """One straight over: 'Romeo Doubs over 36.5 receiving yds' with its own stake."""
    extra = pick.extra or {}
    matchup = f"{pick.away_team} @ {pick.home_team}"
    book = extra.get("bookmaker", "")
    desc = (
        f"{matchup}\n\n"
        f"**Odds:** `{pick.odds:+d}` at {book}  |  **Edge:** `{pick.edge:+.1%}`\n"
        f"**Model:** `{pick.model_prob:.1%}` over  vs  **Fair:** `{pick.implied_prob:.1%}`"
    )
    mean = extra.get("projection_mean")
    if mean is not None:
        desc += f"  |  **Proj:** `{mean:g}`"
    desc += (f"\n**Kelly:** `{pick.kelly_fraction:.2%}`  \u2192  **Bet:** `${pick.recommended_bet:.2f}` "
             "_(paper — overs bankroll)_")
    if extra.get("flat_stake"):
        desc += "\n_Best over in the game by edge, but the model has none: flat 1% stake, tracked._"
    for flag in extra.get("flags", []):
        desc += f"\n\u26a0 {flag.get('detail', '')}"
    if pick.agent_verdict and pick.agent_verdict != "SKIPPED":
        shadow = bool(extra.get("agent", {}).get("shadow"))
        desc += f"\n**Verdict:** `{pick.agent_verdict}`" + (" (shadow)" if shadow else "")
    if pick.agent_reasons:
        desc += "\n**Why:** " + "; ".join(pick.agent_reasons[:2])
    return {"title": f"OVER #{rank}  {over_label(pick.player, pick.market, pick.line)}",
            "description": desc[:4000], "color": COLOR_ORANGE}


def _ladder_line(summary: dict[str, Any] | None, bankroll: float | None = None,
                 label: str = "Ladder hits") -> str | None:
    """One line for the ladder book: record, P&L, ROI and (all-time) its bankroll."""
    if not summary or "total_bets" not in summary:
        return None
    pnl = summary.get("total_pnl", 0.0)
    text = (f"**{label} (own bankroll):** {summary.get('wins', 0)}-{summary.get('losses', 0)}"
            f"-{summary.get('pushes', 0)}  |  **P&L:** {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}"
            f"  |  **ROI:** {summary.get('roi_pct', 0):+.2f}%")
    if summary.get("avg_fair_pct") is not None:
        text += (f"  |  **Hit:** {summary.get('win_rate_pct', 0):.1f}% vs fair "
                 f"{summary['avg_fair_pct']:.1f}%")
    if bankroll is not None:
        text += f"  |  **Bankroll:** ${bankroll:,.2f} \u2192 ${bankroll + pnl:,.2f}"
    return text


LEAN_BET_TYPES = ("moneyline", "spread", "total")


def _lean_line(lean_summary: dict[str, Any] | None, label: str = "Leans") -> str | None:
    """One line for the market leans: record, CLV, line moves — never P&L as a pick."""
    if not lean_summary or "total_bets" not in lean_summary:
        return None
    text = (f"**{label} (paper, not bets):** {lean_summary.get('wins', 0)}-"
            f"{lean_summary.get('losses', 0)}-{lean_summary.get('pushes', 0)}")
    if lean_summary.get("avg_clv_pct") is not None:
        text += (f"  |  **CLV:** {lean_summary['avg_clv_pct']:+.2f}% "
                 f"on {lean_summary.get('clv_sample', 0)}")
    mf, ma = lean_summary.get("line_moves_for", 0), lean_summary.get("line_moves_against", 0)
    if mf or ma:
        text += f"  |  **Line moves:** {mf} for / {ma} against"
    return text


def _td_line(td_summary: dict[str, Any] | None, label: str = "TD scorers",
             bankroll: float | None = None) -> str | None:
    """Anytime-TD scorers: record and hit rate against the FAIR probability
    (a 30% hit rate on +250 shots is a win; never compare with 50%), plus its
    own paper bankroll — the board stakes settings.td_bankroll, never the
    main one."""
    if not td_summary or "total_bets" not in td_summary:
        return None
    pnl = td_summary.get("total_pnl", 0.0)
    text = (f"**{label} (own bankroll):** {td_summary.get('wins', 0)}-"
            f"{td_summary.get('losses', 0)}-{td_summary.get('pushes', 0)}")
    if td_summary.get("avg_fair_pct") is not None:
        text += (f"  |  **Hit:** {td_summary.get('win_rate_pct', 0):.1f}% vs fair "
                 f"{td_summary['avg_fair_pct']:.1f}%")
    if td_summary.get("avg_clv_pct") is not None:
        text += f"  |  **CLV:** {td_summary['avg_clv_pct']:+.2f}% on {td_summary.get('clv_sample', 0)}"
    if bankroll is not None:
        text += f"  |  **Bankroll:** ${bankroll:,.2f} \u2192 ${bankroll + pnl:,.2f}"
    return text


def _result_line(d: dict[str, Any], leans_labelled: bool = False) -> str:
    """One graded pick as a results line, tagged by the section it came from."""
    result_tag = d["result"].upper()
    pnl_val = d["pnl"]
    pnl_str = f"+${pnl_val:,.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):,.2f}"
    matchup = f"{d['away_team']} @ {d['home_team']}"
    odds_str = f"{d['odds']:+d}" if d["odds"] else ""
    if d.get("bet_type") == "prop" and d.get("market") == TD_MARKET:
        bet_desc = f"TD {d['player']} anytime TD"
    elif d.get("bet_type") == "prop" and d.get("strategy") == LADDER_STRATEGY:
        bet_desc = f"LADDER {ladder_label(d.get('player'), d.get('market'), d.get('line'))}"
    elif d.get("bet_type") == "prop" and d.get("strategy") == OVERS_STRATEGY:
        bet_desc = f"OVER {over_label(d.get('player'), d.get('market'), d.get('line'))}"
    elif d.get("bet_type") == "prop" and d.get("player"):
        line_val = d.get("line")
        line_str = f" {line_val:g}" if line_val is not None else ""
        bet_desc = f"{d['player']} {_market_label(d.get('market'))} {d['pick_side']}{line_str}"
    elif leans_labelled and d.get("bet_type") in LEAN_BET_TYPES:
        bet_desc = f"LEAN {d['pick_side']} {d['bet_type'].title()}"
    else:
        bet_desc = f"{d['pick_side']} {d['bet_type'].title()}"
    return f"`{result_tag}`  {bet_desc} ({odds_str}) \u2014 {matchup} \u2014 {pnl_str}"


def _result_section(d: dict[str, Any], leans_labelled: bool) -> str:
    """Which headline line a graded pick belongs under."""
    if d.get("bet_type") == "prop":
        if d.get("market") == TD_MARKET:
            return "td"
        if d.get("strategy") == LADDER_STRATEGY:
            return "ladder"
        if d.get("strategy") == OVERS_STRATEGY:
            return "overs"
    elif leans_labelled and d.get("bet_type") in LEAN_BET_TYPES:
        return "lean"
    return "picks"


# Headline order, so the pick list reads in the same order as the records above it.
_RESULT_SECTIONS = (
    ("picks", "**Picks:**"),
    ("lean", "**Leans:**"),
    ("td", "**TD scorers:**"),
    ("overs", "**Straight overs:**"),
    ("ladder", "**Ladder hits:**"),
)


def _result_lines(pick_details: list[dict[str, Any]], leans_labelled: bool) -> list[str]:
    """
    The graded picks as sectioned lines, one section per headline record.

    Tagging each line (LEAN/TD/OVER/LADDER) made the sections *distinguishable*
    but left the main card's props interleaved with three paper books that have
    their own bankrolls and their own headline records. Group them instead.
    """
    grouped: dict[str, list[str]] = {}
    for d in pick_details:
        if d.get("bet_type") == "parlay" or d.get("strategy") in PARLAY_STRATEGIES:
            continue  # own channel (send_parlay_results_to_discord)
        section = _result_section(d, leans_labelled)
        grouped.setdefault(section, []).append(_result_line(d, leans_labelled=leans_labelled))

    lines: list[str] = []
    for key, header in _RESULT_SECTIONS:
        if grouped.get(key):
            lines += ["", header, *grouped[key]]
    return lines


def _build_results_embed(
    summary: dict[str, Any],
    sport: str,
    graded_date: date | None = None,
    pick_details: list[dict] | None = None,
    lean_summary: dict[str, Any] | None = None,
    td_summary: dict[str, Any] | None = None,
    ladder_summary: dict[str, Any] | None = None,
    overs_summary: dict[str, Any] | None = None,
) -> dict:
    """Build a Discord embed for grading results."""
    if "message" in summary:
        extra_lines = [ln for ln in (_lean_line(lean_summary), _td_line(td_summary),
                                     _ladder_line(overs_summary, label="Straight overs"),
                                     _ladder_line(ladder_summary)) if ln]
        return {
            "title": f"Results \u2014 {sport}",
            "description": "\n".join([summary["message"], *extra_lines]),
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
    lean = _lean_line(lean_summary)
    if lean:
        lines.append(lean)
    td = _td_line(td_summary)
    if td:
        lines.append(td)
    overs = _ladder_line(overs_summary, label="Straight overs")
    if overs:
        lines.append(overs)
    ladder = _ladder_line(ladder_summary)
    if ladder:
        lines.append(ladder)

    if pick_details:
        lines += _result_lines(pick_details, leans_labelled=lean_summary is not None)

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
    lean_summary: dict[str, Any] | None = None,
    alltime_summary: dict[str, Any] | None = None,
    alltime_lean_summary: dict[str, Any] | None = None,
    starting_bankroll: float | None = None,
    td_summary: dict[str, Any] | None = None,
    alltime_td_summary: dict[str, Any] | None = None,
    td_bankroll: float | None = None,
    ladder_summary: dict[str, Any] | None = None,
    alltime_ladder_summary: dict[str, Any] | None = None,
    ladder_bankroll: float | None = None,
    overs_summary: dict[str, Any] | None = None,
    alltime_overs_summary: dict[str, Any] | None = None,
    overs_bankroll: float | None = None,
) -> bool:
    """
    Send grading results to the Discord results channel for the given sport.

    lean_summary adds a paper-leans line (record, CLV) under the picks;
    alltime_summary appends an all-time recap embed so the results channel
    carries the running record without a separate all-time channel.

    Returns True if sent successfully, False otherwise.
    """
    url = _get_webhook_url(sport, "RESULTS")
    if not url:
        logger.debug("Discord not configured for %s results, skipping", sport)
        return False

    if "total_bets" not in summary:
        logger.debug("No graded picks for %s, skipping Discord", sport)
        return True

    embeds = [_build_results_embed(summary, sport, graded_date, pick_details=pick_details,
                                   lean_summary=lean_summary, td_summary=td_summary,
                                   ladder_summary=ladder_summary, overs_summary=overs_summary)]
    if breakdown:
        embeds.append(_build_breakdown_embed(breakdown, sport))
    if alltime_summary is not None and starting_bankroll is not None:
        recap = _build_alltime_sport_embed(alltime_summary, f"All-time \u2014 {sport}", starting_bankroll)
        lean = _lean_line(alltime_lean_summary, label="Leans all-time")
        if lean:
            recap["description"] += f"\n{lean}"
        td = _td_line(alltime_td_summary, label="TD scorers all-time", bankroll=td_bankroll)
        if td:
            recap["description"] += f"\n{td}"
        overs = _ladder_line(alltime_overs_summary, bankroll=overs_bankroll,
                             label="Straight overs all-time")
        if overs:
            recap["description"] += f"\n{overs}"
        ladder = _ladder_line(alltime_ladder_summary, bankroll=ladder_bankroll,
                              label="Ladder hits all-time")
        if ladder:
            recap["description"] += f"\n{ladder}"
        embeds.append(recap)

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
            (total_pnl / total_wagered * 100.0)
            if total_wagered
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


# ---------------------------------------------------------------------------
# Extras channel: the props that cleared the floors but missed the card.
#
# They are the same model, markets and book as the main card — only the slate
# cap separated them — so they are not a Pick.strategy, just on_card=False.
# They get their own channel and their own paper bankroll so their record can
# be read next to the card's without ever entering it. The walk-forward eval
# finds no reliable ordering inside the top ten picks of a week, so whether
# the card should be wider is a real open question; this channel is the A/B
# that answers it with live prices instead of a proxy book.
# ---------------------------------------------------------------------------

#: Extras are listed compactly (there can be dozens); Discord caps an embed
#: description at 4096 characters, so the list is chunked well inside that.
EXTRAS_LINES_PER_EMBED = 20


def _extras_line(pick: BetCandidate, rank: int) -> str:
    """One off-card pick, compact: no per-pick embed, there are too many."""
    book = (pick.extra or {}).get("bookmaker", "")
    return (
        f"`{rank:>2}` **{_pick_label(pick)}** `{pick.odds:+d}`"
        + (f" @ {book}" if book else "")
        + f"  — {pick.away_team} @ {pick.home_team}\n"
        f"     edge `{pick.edge:+.1%}`  model `{pick.model_prob:.1%}` vs `{pick.implied_prob:.1%}`"
        f"  → `${pick.recommended_bet:.2f}`"
    )


def send_extras_to_discord(
    title: str,
    extras: list[BetCandidate],
    bankroll: float,
    sport: str = "NFL",
) -> bool:
    """Post the off-card props for one slate to the extras picks channel."""
    url = _get_webhook_url(sport, "EXTRAS")
    if not url:
        logger.debug("Discord not configured for %s extras, skipping", sport)
        return False
    if not extras:
        return True

    ranked = sorted(extras, key=lambda c: c.edge, reverse=True)
    action = sum(c.recommended_bet for c in ranked)
    header = {
        "title": f"{SPORT_EMOJI.get(sport.upper(), '')} Extras — {title}",
        "description": (
            f"{date.today()}\n\n"
            f"**Bankroll:** ${bankroll:,.2f}  |  **{len(ranked)} off-card**  |  "
            f"**Action:** ${action:.2f}\n"
            "_Cleared the floors but missed the card. Paper only, own bankroll — "
            "never counted in the main record._"
        ),
        "color": COLOR_GREY,
    }

    lines = [_extras_line(p, i) for i, p in enumerate(ranked, 1)]
    embeds = [header]
    for i in range(0, len(lines), EXTRAS_LINES_PER_EMBED):
        embeds.append({
            "description": "\n".join(lines[i: i + EXTRAS_LINES_PER_EMBED]),
            "color": COLOR_GREY,
        })

    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        if not _send_webhook(url, {"embeds": embeds[i: i + MAX_EMBEDS_PER_MESSAGE]}):
            all_ok = False
    return all_ok


def send_extras_results_to_discord(
    summary: dict[str, Any],
    sport: str,
    graded_date: date | None = None,
    pick_details: list[dict] | None = None,
    alltime_summary: dict[str, Any] | None = None,
    starting_bankroll: float | None = None,
) -> bool:
    """
    Grading results for the off-card props.

    Its own channel (DISCORD_WEBHOOK_<SPORT>_EXTRAS_RESULTS), separate from
    both the extras PICKS channel and the main results channel — picks and
    results split the same way for the card, and the main results channel
    still reports only what was offered.
    """
    url = _get_webhook_url(sport, "EXTRAS_RESULTS")
    if not url:
        return False
    if "total_bets" not in summary:
        logger.debug("No graded %s extras, skipping Discord", sport)
        return True

    pnl = summary.get("total_pnl", 0)
    lines = [
        f"**Record:** {summary.get('wins', 0)}-{summary.get('losses', 0)}"
        f"-{summary.get('pushes', 0)} ({summary.get('win_rate_pct', 0):.1f}%)  |  "
        f"**P&L:** {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}  |  "
        f"**ROI:** {summary.get('roi_pct', 0):+.2f}%",
        f"**Avg Edge:** {summary.get('avg_edge_pct', 0):+.2f}%",
    ]
    if summary.get("avg_clv_pct") is not None:
        lines[-1] += f"  |  **Avg CLV:** {summary['avg_clv_pct']:+.2f}%"
    if pick_details:
        lines.append("")
        lines.append("**Off-card picks:**")
        lines.extend(_result_line(d) for d in pick_details)

    date_str = str(graded_date) if graded_date else ""
    embeds = [{
        "title": f"Extras Results — {sport}",
        "description": (f"{date_str}\n\n" if date_str else "") + "\n".join(lines),
        "color": COLOR_GREEN if pnl > 0 else COLOR_RED if pnl < 0 else COLOR_GREY,
    }]

    if alltime_summary and "total_bets" in alltime_summary:
        at_pnl = alltime_summary.get("total_pnl", 0)
        start = starting_bankroll if starting_bankroll is not None else 0.0
        at = [
            f"**Record:** {alltime_summary.get('wins', 0)}-{alltime_summary.get('losses', 0)}"
            f"-{alltime_summary.get('pushes', 0)} "
            f"({alltime_summary.get('win_rate_pct', 0):.1f}%)  |  "
            f"**ROI:** {alltime_summary.get('roi_pct', 0):+.2f}%",
            f"**Bankroll:** ${start:,.2f} → ${start + at_pnl:,.2f}  |  "
            f"**P&L:** {'+' if at_pnl >= 0 else '-'}${abs(at_pnl):,.2f}",
            "_The card's counterfactual: what the picks that missed the cut would have "
            "returned. Compare with the main all-time line to see whether the card is "
            "wide enough._",
        ]
        embeds.append({"title": f"Extras All-time — {sport}",
                       "description": "\n".join(at), "color": COLOR_BLUE})

    return _send_webhook(url, {"embeds": embeds})


# ---------------------------------------------------------------------------
# Long-shot parlays — their own channel and their own paper bankroll
# (intelligence/parlay.py). Never on the card, never in the results post.
# ---------------------------------------------------------------------------

def _leg_label(leg) -> str:
    """A leg on one line: the pick, its price, the game it is in."""
    if isinstance(leg, dict):
        d = leg
        if d.get("bet_type") == "prop":
            line = f" {d['line']:g}" if d.get("line") is not None else ""
            market = (d.get("market") or "").removeprefix("player_").replace("_", " ")
            side = "anytime TD" if d.get("market") == TD_MARKET else f"{market} {d['pick_side']}{line}"
            desc = f"{d.get('player') or '?'} {side}"
        elif d.get("bet_type") == "moneyline":
            desc = f"{d['pick_side']} ML"
        else:
            desc = str(d["pick_side"])
        return f"{desc} `{d['odds']:+d}` — {d.get('away_team', '')} @ {d.get('home_team', '')}"
    return f"{_pick_label(leg)} `{leg.odds:+d}` — {leg.away_team} @ {leg.home_team}"


def _parlay_title(parent, legs_n: int) -> str:
    kind = {"sgp": "SGP", "parlay": "PARLAY"}.get(parent.market, "PARLAY")
    return f"{legs_n}-LEG {kind} · {parent.odds:+d}"


def send_parlays_to_discord(title: str, parlays: list, bankroll: float,
                            sport: str = "NFL") -> bool:
    """Post the run's parlays to the parlays channel, one embed per ticket."""
    url = _get_webhook_url(sport, "PARLAYS")
    if not url:
        logger.debug("Discord not configured for %s parlays, skipping", sport)
        return False
    if not parlays:
        return True

    header = {
        "title": f"{SPORT_EMOJI.get(sport.upper(), '')} Parlays — {title}",
        "description": (
            f"{date.today()}\n\n"
            f"**Bankroll:** ${bankroll:,.2f}  |  **{len(parlays)} ticket(s)**  |  "
            f"**Action:** ${sum(p.parent.recommended_bet for p in parlays):.2f}\n"
            "_Lottery tickets, flat stake, own bankroll — never counted in the main "
            "record. Legs are recombined from the lines already fetched._"
        ),
        "color": COLOR_PURPLE,
    }
    embeds = [header]
    for p in parlays:
        parent = p.parent
        label = {"sgp": "Same-game parlay", "window": "Cross-game parlay",
                 "leans": "Lean parlay"}.get(p.kind, "Parlay")
        lines = [f"**{label}** — {parent.away_team} @ {parent.home_team}" if p.same_game
                 else f"**{label}**", ""]
        lines += [f"`{i}` {_leg_label(leg)}  · model `{leg.model_prob:.0%}`"
                  for i, leg in enumerate(p.legs, 1)]
        lines += ["",
                  f"**Stake:** `${parent.recommended_bet:.2f}` flat  |  "
                  f"**Pays:** `${parent.recommended_bet * _decimal(parent.odds):.2f}`  |  "
                  f"**Claimed P(hit):** `{parent.model_prob:.1%}`  vs  fair `{parent.implied_prob:.1%}`"]
        if p.same_game:
            lines.append("_Same game: the price is the product of the leg prices. The book's "
                         "real SGP price will be lower — check it before staking._")
        embeds.append({"title": _parlay_title(parent, len(p.legs)),
                       "description": "\n".join(lines), "color": COLOR_PURPLE})

    all_ok = True
    for i in range(0, len(embeds), MAX_EMBEDS_PER_MESSAGE):
        if not _send_webhook(url, {"embeds": embeds[i: i + MAX_EMBEDS_PER_MESSAGE]}):
            all_ok = False
    return all_ok


def _decimal(odds: int) -> float:
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def send_parlay_results_to_discord(
    summary: dict[str, Any],
    sport: str,
    graded_date: date | None = None,
    parlays: list[dict] | None = None,
    alltime_summary: dict[str, Any] | None = None,
    starting_bankroll: float | None = None,
) -> bool:
    """
    Grading results for the parlay book, in its own channel
    (DISCORD_WEBHOOK_<SPORT>_PARLAYS_RESULTS): the day's tickets leg by leg,
    the record, and the all-time bankroll line. Hit rate is read against the
    mean claimed probability, never against 50%.
    """
    url = _get_webhook_url(sport, "PARLAYS_RESULTS")
    if not url:
        return False
    if "total_bets" not in summary:
        logger.debug("No graded %s parlays, skipping Discord", sport)
        return True

    pnl = summary.get("total_pnl", 0)
    lines = [
        f"**Record:** {summary.get('wins', 0)}-{summary.get('losses', 0)}"
        f"-{summary.get('pushes', 0)}  |  "
        f"**P&L:** {'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}  |  "
        f"**ROI:** {summary.get('roi_pct', 0):+.2f}%",
    ]
    if parlays:
        claimed = sum(p["model_prob"] for p in parlays) / len(parlays)
        hits = sum(1 for p in parlays if p["result"] == "win")
        lines.append(f"**Hit:** {hits}/{len(parlays)} vs claimed `{claimed:.1%}` per ticket")
        for p in parlays:
            tag = p["result"].upper()
            pnl_str = f"+${p['pnl']:,.2f}" if p["pnl"] >= 0 else f"-${abs(p['pnl']):,.2f}"
            lines += ["", f"`{tag}`  **{p['label']}** `{p['odds']:+d}` — {pnl_str}"]
            for leg in p["legs"]:
                mark = {"win": "\u2705", "loss": "\u274c", "push": "\u27a1\ufe0f",
                        "void": "\u2796"}.get(leg["result"] or "", "\u2753")
                lines.append(f"  {mark} {_leg_label(leg)}")

    date_str = str(graded_date) if graded_date else ""
    embeds = [{
        "title": f"Parlay Results — {sport}",
        "description": (f"{date_str}\n\n" if date_str else "") + "\n".join(lines),
        "color": COLOR_GREEN if pnl > 0 else COLOR_RED if pnl < 0 else COLOR_GREY,
    }]

    if alltime_summary and "total_bets" in alltime_summary:
        at_pnl = alltime_summary.get("total_pnl", 0)
        start = starting_bankroll if starting_bankroll is not None else 0.0
        at = [
            f"**Record:** {alltime_summary.get('wins', 0)}-{alltime_summary.get('losses', 0)}"
            f"-{alltime_summary.get('pushes', 0)} "
            f"({alltime_summary.get('win_rate_pct', 0):.1f}%)  |  "
            f"**ROI:** {alltime_summary.get('roi_pct', 0):+.2f}%",
            f"**Bankroll:** ${start:,.2f} → ${start + at_pnl:,.2f}  |  "
            f"**P&L:** {'+' if at_pnl >= 0 else '-'}${abs(at_pnl):,.2f}",
            "_Lottery tickets: expect a low hit rate and a bankroll that bleeds between "
            "hits. The question is whether the hits pay for the misses._",
        ]
        embeds.append({"title": f"Parlays All-time — {sport}",
                       "description": "\n".join(at), "color": COLOR_PURPLE})

    return _send_webhook(url, {"embeds": embeds})
