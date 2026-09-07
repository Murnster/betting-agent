#!/usr/bin/env python
"""
NFL player prop picks (Phase 3 — PAPER TRADING).

Fetches per-event prop odds (bet365, falling back to DraftKings/FanDuel — the
Odds API has no bet365 prop feed), projects each player's stat
distribution from nflreadpy weekly stats, and surfaces overs/unders where the
model probability beats the de-vigged market probability by the edge
threshold. Picks save as bet_type="prop" with recommended (paper) stakes —
this phase logs and grades picks, it does not tell you to bet money.

Usage:
    uv run python scripts/props.py                     # print picks
    uv run python scripts/props.py --save              # log to picks table
    uv run python scripts/props.py --max-events 5      # cap API spend
    uv run python scripts/props.py --pick-games        # choose games first
    uv run python scripts/props.py --suggest 3         # auto-pick 3 hottest games
    uv run python scripts/props.py --today --save      # daily cron: today's slate only,
                                                       # off-days exit free. Combine with
                                                       # --suggest N to cap credits.
    uv run python scripts/props.py --closing           # pre-kickoff: snapshot closing
                                                       # price/line on held picks (CLV)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

import pandas as pd

from betting_agent.config import settings
from betting_agent.intelligence.ev import american_to_implied_prob, remove_vig
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.intelligence.picks import BetCandidate, save_picks_to_db
from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    PROP_EDGE_FLOORS,
    ReceivingPropsModel,
    active_player_keys,
    books_in_preference,
    approximate_nfl_week,
    book_proxy_line,
    build_receiving_history,
    edge_floor,
    fetch_prop_odds,
    load_player_stats,
    nfl_week_for,
    normalize_player,
    pair_outcomes,
    prop_bookmaker_order,
    schedule_row_for,
)
from betting_agent.sports.registry import get_sport_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RECENT_GAMES_FOR_PAYLOAD = 8


def _current_nfl_season(today: date) -> int:
    return get_sport_config("NFL").season_for_date(today)


# Kept under the old private name for callers; the implementation is shared
# with the closing-line capture in accounting/prop_clv.py.
_pair_outcomes = pair_outcomes
_nfl_week = approximate_nfl_week


def _event_date(event: dict) -> date:
    try:
        return datetime.fromisoformat(
            event.get("commence_time", "").replace("Z", "+00:00")
        ).date()
    except ValueError:
        return date.today()


def _active_players(
    hist: pd.DataFrame,
    asof_t: int,
    teams: tuple[str, str],
    current_teams: dict[str, str] | None,
    last_team: pd.Series | None = None,
) -> dict[str, str]:
    """
    {player_key: team} for players a book would hang a line on in this
    game: seen in the last three slates, on one of the two teams. The
    published roster wins over the last stats row, so offseason movers land
    on their new side.
    """
    if last_team is None:
        last_team = hist.sort_values("t").groupby("player_key")["team"].last()
    current_teams = current_teams or {}
    out: dict[str, str] = {}
    for pk in active_player_keys(hist, asof_t):
        team = current_teams.get(pk) or last_team.get(pk)
        if team in teams:
            out[pk] = team
    return out


def rank_events_by_model_heat(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    history,
    season: int,
    current_teams: dict[str, str] | None = None,
    schedule: pd.DataFrame | None = None,
) -> list[tuple[dict, int, float]]:
    """
    Free pre-screen: score each upcoming game by the prop edges the models
    expect at trailing-median proxy lines, before any paid odds call.

    Returns [(event, n_players_clearing_a_floor, summed_best_edges)] sorted
    hottest first. The proxy line is our own trailing median, not the book's
    line, so heat is a where-to-look signal — the real edges are computed
    against real prices after the per-event fetch.
    """
    from betting_agent.sports.teams import canonical_team

    hist = history.sort_values("t")
    last_team = hist.groupby("player_key")["team"].last()
    tails = {
        model.stat_col: hist.groupby("player_key")[model.stat_col].apply(
            lambda s: [max(0.0, v) for v in s.tail(8).tolist()]
        )
        for model in models.values()
    }

    ranked: list[tuple[dict, int, float]] = []
    for event in events:
        home_ab = canonical_team("NFL", event.get("home_team", ""))
        away_ab = canonical_team("NFL", event.get("away_team", ""))
        week = nfl_week_for(_event_date(event), season, schedule, home_ab, away_ab)
        asof_t = season * 100 + week
        players = _active_players(hist, asof_t, (home_ab, away_ab), current_teams, last_team)

        best: dict[str, float] = {}
        for market, model in models.items():
            for pk, team in players.items():
                opponent = away_ab if team == home_ab else home_ab
                proj = model.project(pk, season, week, opponent=opponent)
                if proj is None:
                    continue
                line = book_proxy_line(tails[model.stat_col].get(pk, []), market)
                if line is None:
                    continue
                p_over = proj.prob_over(line)
                for p in (p_over, proj.prob_under(line)):
                    if not 0.15 <= p <= 0.85:
                        continue
                    edge = p - 0.5
                    if edge >= edge_floor(market, None) and edge > best.get(pk, 0.0):
                        best[pk] = edge
        ranked.append((event, len(best), sum(best.values())))
    ranked.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return ranked


def _choose_events(ranked: list[tuple[dict, int, float]]) -> list[dict]:
    """
    Interactive event picker, hottest games first. Listing events is a free
    API call; the paid per-event odds calls only happen for what gets chosen.
    """
    print("\nUpcoming games, ranked by expected prop edges (free pre-screen):")
    for i, (e, n_edges, heat) in enumerate(ranked, 1):
        when = e.get("commence_time", "")[:16].replace("T", " ")
        print(f"  {i:>2}. {e.get('away_team', '?')} @ {e.get('home_team', '?')}"
              f"  ({when} UTC)  edges={n_edges}  heat={heat:+.2f}")
    events = [e for e, _, _ in ranked]
    if not sys.stdin.isatty():
        return events
    raw = input("\nSelect games (e.g. 1,3,5 — or 'all'): ").strip().lower()
    if raw in ("", "all", "a"):
        return events
    picked = []
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit() and 1 <= int(tok) <= len(events):
            picked.append(events[int(tok) - 1])
    return picked or events


def _events_commencing_today(events: list[dict], now: datetime | None = None) -> list[dict]:
    """
    Games whose kickoff falls on the LOCAL calendar day and has not started.

    commence_time is UTC, and Sunday Night Football at 8:20pm ET is already
    Monday in UTC — filtering on the raw string would misfile every primetime
    game. This is what makes a dumb daily cron job schedule-aware: Thursday,
    Saturday, Sunday, and Monday slates all match on their own day, and
    off-days return nothing (no credits spent). Games already kicked off are
    dropped so a second run never refreshes saved picks with in-play prices.
    """
    now = (now or datetime.now()).astimezone()
    today = now.date()
    out = []
    for e in events:
        try:
            kickoff = datetime.fromisoformat(
                e.get("commence_time", "").replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if kickoff.astimezone().date() == today and kickoff > now:
            out.append(e)
    return out


def _deduplicate_by_player(candidates: list[BetCandidate]) -> list[BetCandidate]:
    """
    One pick per player per slate — keep the highest-edge market.

    Receptions and receiving yards on the same player are close to the same
    bet: in the walk-forward diagnostic 72.8% of picks were a doubled player,
    88% of those took the same side, and doubled players lost BOTH legs 27.7%
    of the time against 17.1% if they were independent. Keeping only the best
    leg also raised the realized hit rate (59.7% → 62.2%).
    """
    best: dict[str, BetCandidate] = {}
    for c in sorted(candidates, key=lambda x: x.edge, reverse=True):
        best.setdefault(normalize_player(c.player), c)
    return list(best.values())


def _recent_values(model, player_key: str) -> list[float]:
    """Last games of the modeled stat for the validator payload ([] if unknown)."""
    col = getattr(model, "stat_col", None)
    hist = getattr(model, "history", None)
    if col is None or hist is None or col not in hist.columns:
        return []
    rows = hist[hist["player_key"] == player_key]
    if "t" in rows.columns:
        rows = rows.sort_values("t")
    return [round(max(0.0, float(v)), 1) for v in rows[col].tail(RECENT_GAMES_FOR_PAYLOAD)]


def player_teams(candidates: list[BetCandidate]) -> dict[str, str]:
    """player_key → team abbreviation, from what generate_prop_candidates recorded."""
    return {
        normalize_player(c.player): c.extra["team"]
        for c in candidates
        if c.player and c.extra.get("team")
    }


def generate_prop_candidates(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    bankroll: float,
    min_edge: float | None,
    season: int,
    current_teams: dict[str, str] | None = None,
    schedule: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    qb1: dict | None = None,
    book_order: list[str] | None = None,
) -> list[BetCandidate]:
    from betting_agent.intelligence.picks import (
        _apply_same_game_correlation_adjustment,
    )
    from betting_agent.sports.nfl.injuries import apply_injury_policy, prop_injury_flags
    from betting_agent.sports.teams import canonical_team

    current_teams = current_teams or {}
    candidates: list[BetCandidate] = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        home_abbrev = canonical_team("NFL", home)
        away_abbrev = canonical_team("NFL", away)
        event_date = _event_date(event)
        week = nfl_week_for(event_date, season, schedule, home_abbrev, away_abbrev)

        books = books_in_preference(event, book_order)
        if book_order and books and books[0].get("key") != book_order[0]:
            logger.warning("%s posted no props for %s @ %s — priced against %s",
                           book_order[0], away, home, books[0].get("key"))
        for book in books:
            for market in book.get("markets", []):
                key = market.get("key")
                model = models.get(key)
                if model is None:
                    continue
                for (player, line), pair in pair_outcomes(market).items():
                    over = pair.get("Over")
                    under = pair.get("Under")
                    if not over or not under:
                        continue
                    player_key = normalize_player(player)
                    player_team = current_teams.get(player_key)
                    if player_team is None:
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
                    floor = edge_floor(key, min_edge)
                    for side, price, model_p, fair_p in (
                        ("over", over["price"], p_over, fair_over),
                        ("under", under["price"], proj.prob_under(line), fair_under),
                    ):
                        edge = model_p - fair_p
                        if edge < floor:
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
                                   "bookmaker": book.get("key"),
                                   "team": player_team if opponent else None,
                                   "recent_values": _recent_values(model, player_key)},
                        ))
    candidates = _deduplicate_by_player(candidates)
    # Official injury report: Out/Doubtful players are dropped, Questionable
    # and a QB1 who is out are attached as flags. No-op when the feeds are
    # unavailable (pre-season gate, download failure).
    if injuries is not None or qb1:
        flags = prop_injury_flags(candidates, injuries, qb1, player_teams(candidates))
        candidates = apply_injury_policy(candidates, flags)
    # Props in one game share a quarterback and a game script, so their
    # outcomes move together — scale the stakes down accordingly.
    _apply_same_game_correlation_adjustment(candidates)
    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates


def game_lines_from_schedule(
    events: list[dict], schedule: pd.DataFrame | None
) -> dict[str, tuple[float | None, float | None]]:
    """external_id → (spread_line, total_line) from the published schedule."""
    from betting_agent.sports.teams import canonical_team

    if schedule is None or schedule.empty:
        return {}
    out: dict[str, tuple[float | None, float | None]] = {}
    for e in events:
        row = schedule_row_for(
            schedule, _event_date(e),
            canonical_team("NFL", e.get("home_team", "")),
            canonical_team("NFL", e.get("away_team", "")),
        )
        if row is None or not e.get("id"):
            continue

        def _num(v):
            return None if v is None or pd.isna(v) else float(v)

        out[str(e["id"])] = (_num(row.get("spread_line")), _num(row.get("total_line")))
    return out


def _print_candidates(candidates: list[BetCandidate], shadow: bool) -> None:
    print(f"\n{'player':<24} {'market':<22} {'side':<6} {'line':>6} {'odds':>6} "
          f"{'model':>7} {'fair':>7} {'edge':>7} {'paper $':>8}  {'book':<11} verdict")
    for c in candidates:
        verdict = c.agent_verdict or ""
        if verdict and verdict != "SKIPPED" and shadow:
            verdict += " (SHADOW)"
        print(f"{c.player:<24} {c.market:<22} {c.pick_side:<6} {c.line:>6.1f} "
              f"{c.odds:>+6} {c.model_prob:>6.1%} {c.implied_prob:>6.1%} "
              f"{c.edge:>+6.1%} {c.recommended_bet:>8.2f}  "
              f"{(c.extra.get('bookmaker') or ''):<11} {verdict}")
        for flag in c.extra.get("flags", []):
            print(f"{'':<24} ! {flag.get('detail', '')}")
        agent = c.extra.get("agent") or {}
        for reason in agent.get("reasons", [])[:2]:
            print(f"{'':<24} > {reason}")


def run_closing_capture(window_minutes: int) -> None:
    from betting_agent.accounting.prop_clv import capture_closing_lines_for_upcoming

    books = prop_bookmaker_order()
    n = capture_closing_lines_for_upcoming(window_minutes=window_minutes, bookmakers=books)
    if n:
        print(f"Stored closing lines for {n} prop picks.")
    else:
        print(f"No held prop picks kick off in the next {window_minutes} minutes — "
              "nothing captured, no credits spent.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NFL prop picks (paper trading)")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--save", action="store_true", help="Persist picks to DB")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Cap per-event odds calls (API credit control)")
    parser.add_argument("--min-edge", type=float, default=None,
                        help="Override the per-market edge floors "
                             f"(defaults: {PROP_EDGE_FLOORS})")
    parser.add_argument("--max-picks", type=int, default=10)
    parser.add_argument("--pick-games", action="store_true",
                        help="List upcoming games (free call) and choose which "
                             "to fetch prop odds for — saves API credits")
    parser.add_argument("--suggest", type=int, default=None, metavar="N",
                        help="Rank upcoming games by expected prop edges (free "
                             "pre-screen) and fetch odds for only the top N "
                             "(~2 credits per game)")
    parser.add_argument("--today", action="store_true",
                        help="Only games kicking off today (local time). Made "
                             "for a daily cron job: off-days exit immediately "
                             "with no credits spent, Saturday slates and "
                             "primetime games are caught on their own day")
    parser.add_argument("--closing", action="store_true",
                        help="Capture closing price/line for held prop picks in "
                             "games kicking off within --window-minutes (2 credits "
                             "per game we hold picks in; zero when none)")
    parser.add_argument("--window-minutes", type=int, default=90,
                        help="Kickoff window for --closing (default 90)")
    parser.add_argument("--agent-mode", type=str,
                        choices=["off", "top", "all"],
                        default=settings.agent_mode if settings.agent_enabled else "off",
                        help="LLM validator: off, top (hottest games), or all. "
                             "Runs in shadow mode unless AGENT_SHADOW=false")
    args = parser.parse_args()

    if args.closing:
        run_closing_capture(args.window_minutes)
        return

    bankroll = args.bankroll or settings.starting_bankroll
    min_edge = args.min_edge
    season = _current_nfl_season(date.today())

    # The free events call comes BEFORE model fitting: on an off-day a
    # --today cron run exits here in seconds, without downloading stats.
    upcoming = None
    if args.pick_games or args.suggest or args.today:
        from betting_agent.api.odds import OddsAPIClient
        upcoming = OddsAPIClient().fetch_events("americanfootball_nfl")
        if not upcoming:
            raise SystemExit("No upcoming NFL events — check ODDS_API_KEY / season timing.")
        if args.today:
            upcoming = _events_commencing_today(upcoming)
            if not upcoming:
                print("No NFL games today — nothing fetched, no credits spent.")
                return

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

    # Season context, all free: roster (offseason movers), schedule (exact
    # week, spread/total), injury report + depth charts (QB1).
    from betting_agent.sports.nfl.injuries import load_injury_report, qb1_by_team
    from betting_agent.sports.nfl.props import current_teams, load_season_schedule

    roster = current_teams(season)
    if roster:
        last_team = history.sort_values("t").groupby("player_key")["team"].last()
        moved = sum(1 for pk, t in last_team.items() if roster.get(pk) not in (None, t))
        logger.info("Roster overlay: %d players, %d re-teamed vs their last stats row",
                    len(roster), moved)
    schedule = load_season_schedule(season)
    first_event = (upcoming or [None])[0]
    week_hint = nfl_week_for(_event_date(first_event) if first_event else date.today(),
                             season, schedule)
    injuries = load_injury_report(season, week_hint)
    qb1 = qb1_by_team(season)
    if injuries.empty:
        logger.info("No injury report for %s week %d — injury checks skipped", season, week_hint)

    books = prop_bookmaker_order()
    chosen = None
    if upcoming is not None:
        ranked = rank_events_by_model_heat(upcoming, models, history, season, roster, schedule)
        if args.suggest:
            chosen = [e for e, _, _ in ranked[: args.suggest]]
            print(f"\nSuggested games (top {len(chosen)} by expected prop edges):")
            for e, n_edges, heat in ranked[: args.suggest]:
                print(f"  {e.get('away_team', '?')} @ {e.get('home_team', '?')}"
                      f"  edges={n_edges}  heat={heat:+.2f}")
        elif args.pick_games:
            chosen = _choose_events(ranked)
        else:
            chosen = upcoming  # --today alone: every game on today's slate
    logger.info("Fetching prop odds (books: %s)...", ",".join(books))
    events = fetch_prop_odds(bookmakers=books, max_events=args.max_events, events=chosen)
    if not events:
        raise SystemExit("No prop odds returned — check ODDS_API_KEY / season timing.")

    candidates = generate_prop_candidates(
        events, models, bankroll, min_edge, season,
        current_teams=roster, schedule=schedule, injuries=injuries, qb1=qb1,
        book_order=books,
    )
    candidates = candidates[:args.max_picks]

    if not candidates:
        print("\nNo prop edges clear the threshold today.")
        return

    # LLM validator — shadow by default: verdicts are recorded and shown but
    # do not touch edge, sizing, or the slate.
    agent_summary: dict | None = None
    validation_records = []
    if args.agent_mode != "off":
        try:
            from betting_agent.intelligence.validator import validate_picks

            candidates, validation = validate_picks(
                candidates, sport="NFL", mode=args.agent_mode,
                injuries=injuries, qb1=qb1, player_teams=player_teams(candidates),
                game_lines=game_lines_from_schedule(events, schedule),
            )
            validation_records = validation.records
            agent_summary = validation.as_dict()
        except Exception as exc:
            logger.warning("Validator failed, continuing without it: %s", exc)

    shadow = bool(agent_summary and agent_summary.get("shadow"))
    _print_candidates(candidates, shadow)
    if agent_summary:
        print(f"\nValidator{' (SHADOW — verdicts recorded, stakes untouched)' if shadow else ''}: "
              f"{agent_summary['validated_games']} games, "
              f"{agent_summary['skipped_games']} skipped, "
              f"${agent_summary['total_cost_usd']:.4f}")
    print(f"\n{len(candidates)} paper picks. These are NOT bets — Phase 3 "
          "validates the projections first.")

    if args.save:
        save_picks_to_db(candidates)
        print("Saved to picks table (bet_type='prop').")
        if validation_records:
            try:
                from betting_agent.intelligence.validator import save_agent_validations_to_db

                save_agent_validations_to_db(validation_records)
            except Exception as exc:
                logger.warning("Could not save agent validations: %s", exc)

    # On an unattended box (cron), Discord is the only way the picks get seen.
    if settings.discord_enabled:
        try:
            from betting_agent.notifications.discord import (
                is_discord_configured,
                send_picks_to_discord,
            )
            if is_discord_configured("NFL", "PICKS"):
                logger.info("Sending prop picks to Discord...")
                send_picks_to_discord(candidates, bankroll, "NFL", agent_summary=agent_summary)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
