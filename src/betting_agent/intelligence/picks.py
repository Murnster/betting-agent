"""
POTD (Picks of the Day) generation + CLI formatting.
Generates best pick per bet type per game, filtered by edge threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

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
from betting_agent.sports.registry import get_sport_config
from betting_agent.sports.teams import canonical_team

logger = logging.getLogger(__name__)


def _passes_guardrails(
    edge: float,
    odds: int,
    model_prob: float,
    bet_type: str,
) -> bool:
    """Reject structurally suspect picks. Returns True if pick passes."""
    if edge > settings.max_edge_pct:
        logger.warning(
            "Guardrail: rejecting pick — edge %.1f%% exceeds max %.1f%%",
            edge * 100, settings.max_edge_pct * 100,
        )
        return False
    if bet_type == "moneyline":
        if odds > 0 and odds > settings.max_underdog_odds:
            logger.warning(
                "Guardrail: rejecting ML pick — odds +%d exceed max +%d",
                odds, settings.max_underdog_odds,
            )
            return False
        if model_prob < settings.min_model_prob:
            logger.warning(
                "Guardrail: rejecting ML pick — model prob %.1f%% below min %.1f%%",
                model_prob * 100, settings.min_model_prob * 100,
            )
            return False
    return True


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
    scheduled_game_date: date | None = None
    kelly_fraction: float = 0.0
    recommended_bet: float = 0.0
    bankroll_at_pick: float = 0.0
    extra: dict = field(default_factory=dict)  # spread_line, total_line, etc.
    external_id: str | None = None  # Odds API game ID for DB upsert
    original_edge: float | None = None
    agent_verdict: str | None = None
    agent_reasons: list[str] = field(default_factory=list)
    agent_adjusted_bet: float | None = None
    agent_adjusted_kelly: float | None = None
    agent_cost_usd: float | None = None

    @property
    def event_date(self) -> date:
        return self.scheduled_game_date or self.game_date


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
    max_picks: int | None = None,
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

    sport_config = get_sport_config(sport)

    def _ml_min_edge(implied_prob: float) -> float:
        tiers = sport_config.ml_edge_tiers
        if not tiers:
            return settings.min_edge_pct
        for prob_max, min_edge in sorted(tiers, key=lambda x: x[0]):
            if implied_prob < prob_max:
                return max(min_edge, settings.min_edge_pct)
        return settings.min_edge_pct

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
        external_id = _normalize_external_id(meta.get("external_id"))
        home = str(meta.get("home_team", ""))
        away = str(meta.get("away_team", ""))
        scheduled_game_date = _coerce_game_date(
            meta.get("game_date") or meta.get("gameday"),
            fallback=pick_date,
        )

        # Find best odds for this game (use Odds API name if available)
        home_odds_name = str(meta.get("home_team_odds", home))
        away_odds_name = str(meta.get("away_team_odds", away))
        best_odds = _find_best_odds(odds_data, home_odds_name, sport=sport)

        win_prob = float(pred_row["win_prob"])
        pred_margin = float(pred_row["pred_margin"])
        pred_total = float(pred_row["pred_total"])

        # Sentiment adjustment helper
        def _sent_adj(team: str) -> float:
            if sentiment_scores is None:
                return 0.0
            return sentiment_scores.get(team, 0.0) * settings.sentiment_weight

        # ---- Moneyline ----
        # Each side is priced at its best book and de-vigged against that same
        # book's other side. Mixing books produces a market nobody offers and
        # inflates the edge.
        ml_home = best_odds.get("home_price")
        ml_away = best_odds.get("away_price")
        ml_home_pair = best_odds.get("home_pair_away_price")
        ml_away_pair = best_odds.get("away_pair_home_price")

        if ml_home:
            if ml_home_pair:
                edge = calculate_edge_fair(win_prob, ml_home, ml_home_pair, pick_home=True)
            else:
                edge = calculate_edge(win_prob, ml_home)
            edge += _sent_adj(home)
            implied = american_to_implied_prob(ml_home)
            min_edge = _ml_min_edge(implied)
            if edge >= min_edge and _passes_guardrails(edge, ml_home, win_prob, "moneyline"):
                kf, bet = recommended_bet(win_prob, ml_home, edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    scheduled_game_date=scheduled_game_date,
                    bet_type="moneyline", pick_side=home_odds_name,
                    model_prob=win_prob, implied_prob=implied, edge=edge,
                    odds=ml_home, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll, external_id=external_id,
                ))

        away_win_prob = 1.0 - win_prob
        if ml_away:
            if ml_away_pair:
                edge = calculate_edge_fair(away_win_prob, ml_away_pair, ml_away, pick_home=False)
            else:
                edge = calculate_edge(away_win_prob, ml_away)
            edge += _sent_adj(away)
            implied = american_to_implied_prob(ml_away)
            min_edge = _ml_min_edge(implied)
            if edge >= min_edge and _passes_guardrails(edge, ml_away, away_win_prob, "moneyline"):
                kf, bet = recommended_bet(away_win_prob, ml_away, edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    scheduled_game_date=scheduled_game_date,
                    bet_type="moneyline", pick_side=away_odds_name,
                    model_prob=away_win_prob, implied_prob=implied, edge=edge,
                    odds=ml_away, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll, external_id=external_id,
                ))

        # ---- Spread ----
        spread_line = best_odds.get("spread_home")
        spread_home_price = best_odds.get("spread_home_price")
        # The opposing price must come from the book quoting this line.
        spread_away_price = best_odds.get("spread_home_pair_away_price")
        if spread_line is not None and spread_home_price:
            cover_prob, s_edge = calculate_spread_edge(
                pred_margin, -spread_line, spread_home_price,
                away_odds=spread_away_price, sigma=margin_sigma,
            )
            s_edge += _sent_adj(home)
            if s_edge >= settings.min_edge_pct and _passes_guardrails(s_edge, spread_home_price, cover_prob, "spread"):
                implied = american_to_implied_prob(spread_home_price)
                kf, bet = recommended_bet(cover_prob, spread_home_price, s_edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    scheduled_game_date=scheduled_game_date,
                    bet_type="spread", pick_side=f"{home_odds_name} {spread_line:+.1f}",
                    model_prob=cover_prob, implied_prob=implied, edge=s_edge,
                    odds=spread_home_price, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll, external_id=external_id,
                    extra={"spread_line": spread_line},
                ))

        # ---- Total (Over/Under) ----
        # Books hang different numbers, so each side carries its own line and
        # its own book's opposing price.
        total_line = best_odds.get("total_line")
        over_price = best_odds.get("over_price")
        over_pair_under = best_odds.get("over_pair_under_price")
        under_line = best_odds.get("total_under_line")
        under_price = best_odds.get("under_price")
        under_pair_over = best_odds.get("under_pair_over_price")

        if total_line and over_price:
            over_prob, o_edge = calculate_total_edge(pred_total, total_line, over_price, "over", other_odds=over_pair_under, sigma=total_sigma)
            o_edge += (_sent_adj(home) + _sent_adj(away)) / 2
            if o_edge >= settings.min_edge_pct and _passes_guardrails(o_edge, over_price, over_prob, "total"):
                implied = american_to_implied_prob(over_price)
                kf, bet = recommended_bet(over_prob, over_price, o_edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    scheduled_game_date=scheduled_game_date,
                    bet_type="total", pick_side=f"over {total_line}",
                    model_prob=over_prob, implied_prob=implied, edge=o_edge,
                    odds=over_price, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll, external_id=external_id,
                    extra={"total_line": total_line},
                ))

        if under_line and under_price:
            under_prob, u_edge = calculate_total_edge(pred_total, under_line, under_price, "under", other_odds=under_pair_over, sigma=total_sigma)
            u_edge += (_sent_adj(home) + _sent_adj(away)) / 2
            if u_edge >= settings.min_edge_pct and _passes_guardrails(u_edge, under_price, under_prob, "total"):
                implied = american_to_implied_prob(under_price)
                kf, bet = recommended_bet(under_prob, under_price, u_edge, bankroll)
                candidates.append(BetCandidate(
                    game_id=game_id, home_team=home, away_team=away,
                    game_date=pick_date, sport=sport,
                    scheduled_game_date=scheduled_game_date,
                    bet_type="total", pick_side=f"under {under_line}",
                    model_prob=under_prob, implied_prob=implied, edge=u_edge,
                    odds=under_price, kelly_fraction=kf, recommended_bet=bet,
                    bankroll_at_pick=bankroll, external_id=external_id,
                    extra={"total_line": under_line},
                ))

    # Dedup per actual game, not the temporary integer game_id placeholder used
    # before Odds API external IDs are resolved into DB ids.
    seen: dict[tuple[str, str], BetCandidate] = {}
    for c in candidates:
        key = (_candidate_game_key(c), c.bet_type)
        if key not in seen or c.edge > seen[key].edge:
            seen[key] = c
    candidates = list(seen.values())

    # Dedup correlated ML/spread: when both exist for the same team
    # in the same game, keep only the higher-edge pick.
    team_bets: dict[tuple[str, str], BetCandidate] = {}
    uncorrelated: list[BetCandidate] = []
    for c in candidates:
        team = _extract_team_from_pick(c)
        if team is None:
            uncorrelated.append(c)
            continue
        key = (_candidate_game_key(c), team)
        if key not in team_bets or c.edge > team_bets[key].edge:
            team_bets[key] = c
    candidates = uncorrelated + list(team_bets.values())

    # Sort by edge descending
    candidates.sort(key=lambda c: c.edge, reverse=True)
    if max_picks is not None and max_picks > 0:
        candidates = candidates[:max_picks]

    _apply_same_game_correlation_adjustment(candidates)
    return candidates


def _extract_team_from_pick(candidate: BetCandidate) -> str | None:
    """Extract the team name from a pick for correlated-bet dedup.

    Returns None for totals (not team-directional).
    """
    if candidate.bet_type == "moneyline":
        return candidate.pick_side
    if candidate.bet_type == "spread":
        # Strip the spread number, e.g. "Buffalo Bills +1.5" → "Buffalo Bills"
        return candidate.pick_side.rsplit(" ", 1)[0]
    return None


def _candidate_game_key(candidate: BetCandidate) -> str:
    external_id = _normalize_external_id(candidate.external_id)
    if external_id:
        return external_id
    if candidate.game_id:
        return str(candidate.game_id)
    return f"{candidate.away_team}@{candidate.home_team}:{candidate.event_date.isoformat()}"


def _normalize_external_id(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"<na>", "nan", "nat", "none", "null"}:
        return None
    return text


def _apply_same_game_correlation_adjustment(candidates: list[BetCandidate]) -> None:
    # Scale Kelly down only across the final returned slate.
    from collections import Counter
    import math

    game_counts = Counter(_candidate_game_key(candidate) for candidate in candidates)
    for candidate in candidates:
        count = game_counts[_candidate_game_key(candidate)]
        if count <= 1:
            continue
        corr_factor = 1.0 / math.sqrt(count)
        candidate.kelly_fraction *= corr_factor
        candidate.recommended_bet *= corr_factor


def _coerce_game_date(value, fallback: date) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return fallback
    return parsed.date()


def _find_best_odds(odds_data: list[dict], home_team: str, sport: str = "") -> dict:
    """Find best available odds for the home_team matchup in raw Odds API data."""
    from betting_agent.api.odds import OddsAPIClient
    client = OddsAPIClient()
    return client.get_best_odds(odds_data, home_team, sport=sport)


def _resolve_game_id(session, candidate: BetCandidate) -> int:
    """Ensure a Game row exists for this candidate and return its DB id."""
    from betting_agent.db.queries import upsert_game, get_game_by_external_id

    # If the candidate already has a valid game_id (from historical data), use it
    if candidate.game_id and candidate.game_id != 0:
        existing = session.query(Game).get(candidate.game_id)
        if existing:
            return candidate.game_id

    # Look up or create by external_id (Odds API game ID)
    external_id = _normalize_external_id(candidate.external_id)
    if external_id:
        candidate.external_id = external_id
        existing = get_game_by_external_id(session, external_id)
        if existing:
            return existing.id

        game = upsert_game(session, {
            "external_id": external_id,
            "sport": candidate.sport,
            "season": candidate.event_date.year,
            "game_date": candidate.event_date,
            # Canonical abbreviation, matching loader-seeded rows — grading
            # compares the pick's team against these.
            "home_team": canonical_team(candidate.sport, candidate.home_team),
            "away_team": canonical_team(candidate.sport, candidate.away_team),
            "status": "scheduled",
        })
        return game.id

    raise ValueError(
        f"Cannot resolve game_id for {candidate.away_team}@{candidate.home_team}: "
        "no valid game_id or external_id"
    )


def save_picks_to_db(candidates: list[BetCandidate]) -> None:
    """
    Persist POTD candidates to the picks table.
    Ensures Game rows exist (via external_id upsert) before inserting picks.
    Prevents duplicates by checking if (game_id, bet_type, pick_date) already exists.
    """
    if not candidates:
        return

    with get_session() as session:
        # Resolve real DB game IDs for all candidates
        resolved_ids: dict[int, int] = {}  # index → db game_id
        for i, c in enumerate(candidates):
            try:
                resolved_ids[i] = _resolve_game_id(session, c)
            except ValueError as exc:
                logger.warning("Skipping pick: %s", exc)

        # Pre-fetch existing picks for these games. Existing rows may have been
        # written under either the historical run date or a scheduled game date,
        # so dedupe at the game/bet-type level.
        game_ids = set(resolved_ids.values())

        existing_rows = (
            session.query(Pick)
            .filter(
                Pick.game_id.in_(game_ids),
            )
            .all()
        )

        # Create set of (game_id, bet_type) that already exist.
        existing_keys = {(r.game_id, r.bet_type) for r in existing_rows}

        added = 0
        skipped = 0
        for i, c in enumerate(candidates):
            if i not in resolved_ids:
                continue
            db_game_id = resolved_ids[i]

            key = (db_game_id, c.bet_type)
            if key in existing_keys:
                skipped += 1
                continue

            pick = Pick(
                game_id=db_game_id,
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
            existing_keys.add(key)
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


def _confidence_stars(
    edge: float,
    thresholds: tuple[float, float, float, float] = (0.04, 0.07, 0.12, 0.20),
) -> str:
    """Return 1-5 star rating based on edge magnitude and sport-specific thresholds."""
    t2, t3, t4, t5 = thresholds
    if edge >= t5:
        return "\u2605\u2605\u2605\u2605\u2605"
    if edge >= t4:
        return "\u2605\u2605\u2605\u2605\u2606"
    if edge >= t3:
        return "\u2605\u2605\u2605\u2606\u2606"
    if edge >= t2:
        return "\u2605\u2605\u2606\u2606\u2606"
    return "\u2605\u2606\u2606\u2606\u2606"


def format_picks_cli(
    candidates: list[BetCandidate],
    bankroll: float,
    sport: str = "NFL",
    agent_summary: dict | None = None,
) -> str:
    """Format picks for terminal output in card-style ranked by confidence."""
    if not candidates:
        return "\nNo +EV picks found for today.\n"

    from betting_agent.sports.registry import get_sport_config
    star_thresholds = get_sport_config(sport).star_thresholds

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
    if agent_summary and (
        agent_summary.get("validated_games")
        or agent_summary.get("skipped_games")
        or agent_summary.get("total_cost_usd")
    ):
        lines.append(
            "  Validator: "
            f"{agent_summary.get('validated_games', 0)} games  |  "
            f"Skipped: {agent_summary.get('skipped_games', 0)}  |  "
            f"Cost: ${agent_summary.get('total_cost_usd', 0.0):.4f}"
        )
        lines.append("=" * 58)

    # Collect LLM analyses by game_id (show after cards)
    analyses: list[str] = []

    W = 54  # inner card width
    for rank, pick in enumerate(ranked, 1):
        matchup = f"{pick.away_team} @ {pick.home_team}"
        label = _pick_label(pick)
        stars = _confidence_stars(pick.edge, star_thresholds)

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
        edge_text = f"{pick.edge:+.1%}"
        if pick.original_edge is not None and abs(pick.original_edge - pick.edge) > 1e-9:
            edge_text = f"{pick.original_edge:+.1%} -> {pick.edge:+.1%}"
        lines.append(_card_row(
            f"       Edge: {edge_text}",
            f"Kelly: {pick.kelly_fraction:.2%}  ",
        ))
        lines.append(_card_row(
            f"       Model: {pick.model_prob:.1%}",
            f"Implied: {pick.implied_prob:.1%}  ",
        ))
        if pick.agent_verdict and pick.agent_verdict != "SKIPPED":
            lines.append(_card_row(
                f"       Verdict: {pick.agent_verdict}",
                "Validator  ",
            ))
        if pick.agent_reasons:
            reason = ", ".join(pick.agent_reasons[:2])[:48]
            lines.append(_card_row(
                f"       Why: {reason}",
                "  ",
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
