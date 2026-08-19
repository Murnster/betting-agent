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
    uv run python scripts/props.py --pick-games        # choose games first
    uv run python scripts/props.py --suggest 3         # auto-pick 3 hottest games
    uv run python scripts/props.py --today --save      # daily cron: today's slate only,
                                                       # off-days exit free. Combine with
                                                       # --suggest N to cap credits.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from betting_agent.config import settings
from betting_agent.intelligence.ev import american_to_implied_prob, remove_vig
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.intelligence.picks import BetCandidate, save_picks_to_db
from betting_agent.sports.nfl.props import (
    MODELED_MARKETS,
    PROP_EDGE_FLOORS,
    edge_floor,
    ReceivingPropsModel,
    book_proxy_line,
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


def rank_events_by_model_heat(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    history,
    season: int,
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
    recent_t = hist.groupby("player_key")["t"].max()
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
        try:
            event_date = datetime.fromisoformat(
                event.get("commence_time", "").replace("Z", "+00:00")
            ).date()
        except ValueError:
            event_date = date.today()
        week = _nfl_week(event_date, season)
        asof_t = season * 100 + week
        # Only players a book would hang a line on: seen in the last 3 weeks.
        active = recent_t[recent_t >= asof_t - 3].index
        players = [pk for pk in active if last_team.get(pk) in (home_ab, away_ab)]

        best: dict[str, float] = {}
        for market, model in models.items():
            for pk in players:
                opponent = away_ab if last_team[pk] == home_ab else home_ab
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


def _nfl_week(event_date: date, season: int) -> int:
    """Approximate NFL week from the date — only used to time-split history."""
    season_start = date(season, 9, 1)
    days = (event_date - season_start).days
    return max(1, min(22, days // 7 + 1))


def _events_commencing_today(events: list[dict]) -> list[dict]:
    """
    Games whose kickoff falls on the LOCAL calendar day.

    commence_time is UTC, and Sunday Night Football at 8:20pm ET is already
    Monday in UTC — filtering on the raw string would misfile every primetime
    game. This is what makes a dumb daily cron job schedule-aware: Thursday,
    Saturday, Sunday, and Monday slates all match on their own day, and
    off-days return nothing (no credits spent).
    """
    today = datetime.now().astimezone().date()
    out = []
    for e in events:
        try:
            kickoff = datetime.fromisoformat(
                e.get("commence_time", "").replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if kickoff.astimezone().date() == today:
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


def generate_prop_candidates(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    bankroll: float,
    min_edge: float | None,
    season: int,
) -> list[BetCandidate]:
    from betting_agent.intelligence.picks import (
        _apply_same_game_correlation_adjustment,
    )
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
                                   "bookmaker": book.get("key")},
                        ))
    candidates = _deduplicate_by_player(candidates)
    # Props in one game share a quarterback and a game script, so their
    # outcomes move together — scale the stakes down accordingly.
    _apply_same_game_correlation_adjustment(candidates)
    candidates.sort(key=lambda c: c.edge, reverse=True)
    return candidates


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
    args = parser.parse_args()

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

    books = settings.preferred_bookmaker_list or ["bet365"]
    chosen = None
    if upcoming is not None:
        ranked = rank_events_by_model_heat(upcoming, models, history, season)
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

    # On an unattended box (cron), Discord is the only way the picks get seen.
    if settings.discord_enabled:
        try:
            from betting_agent.notifications.discord import (
                is_discord_configured,
                send_picks_to_discord,
            )
            if is_discord_configured("NFL", "PICKS"):
                logger.info("Sending prop picks to Discord...")
                send_picks_to_discord(candidates, bankroll, "NFL")
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
