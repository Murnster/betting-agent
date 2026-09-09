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
    uv run python scripts/props.py --no-td             # skip the anytime-TD board
                                                       # (saves 1 credit per game)
    uv run python scripts/props.py --no-ladder         # skip the alternate boards
                                                       # (saves 1 credit per game per market)

Anytime-TD scorers (sports/nfl/td_props.py) are fetched alongside the
receiving markets and priced against the Yes board de-vigged to the market's
expected TDs. Like the game lean, the best scorer per game is always on the
card: a PICK (paper stake) when its edge sits inside the 8-15% window, else a
LEAN at stake 0 — both saved and tracked for hit rate vs fair and CLV.

Ladder hits (generate_ladder_candidates) are the card's "best overs"
sub-section: the player reaching a milestone — 60+ receiving yards, 6+
receptions, 40+ rushing — priced on the books' Over-only alternate boards.
They run as their own paper book (Pick.strategy = "ladder", own bankroll),
never deduped against or sized with the picks above them. Every entry is a
staked pick: the pool is an experiment (user's call) in whether the model
can pick overs the way the main card picks unders, not a validated edge.

Straight overs (generate_over_candidates) are the third section: the book's
main-line Over on the receiving markets, at least one per game (the best by
edge, flat-staked when the model has none), own paper book
(Pick.strategy = "overs"). No extra credits — the main markets are fetched
for the unders card anyway.
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
from betting_agent.intelligence.game_lean import REFERENCE_BOOK, fetch_game_lines, game_leans
from betting_agent.intelligence.picks import BetCandidate, save_picks_to_db
from betting_agent.intelligence.slate import (
    Slate,
    candidates_in_slate,
    cap_per_slate,
    group_events_by_slate,
    select_card,
)
from betting_agent.sports.nfl.props import (
    ALTERNATE_MARKETS,
    DEFAULT_LADDER_HOLD,
    DISTRIBUTION_MARKETS,
    LADDER_MAX_FAIR_PROB,
    LADDER_MIN_FAIR_PROB,
    MODELED_MARKETS,
    OVERS_EDGE_FLOOR,
    OVERS_MIN_STAKE_PCT,
    PROP_EDGE_FLOORS,
    RECEIVING_POSITIONS,
    ReceivingPropsModel,
    active_player_keys,
    books_in_preference,
    approximate_nfl_week,
    book_proxy_line,
    build_receiving_history,
    build_rushing_history,
    edge_cap,
    edge_floor,
    fetch_prop_odds,
    ladder_edge_cap,
    ladder_edge_floor,
    ladder_label,
    ladder_min_rung,
    load_player_stats,
    nfl_week_for,
    normalize_player,
    over_label,
    pair_outcomes,
    prop_bookmaker_order,
    schedule_row_for,
)
from betting_agent.sports.nfl.td_props import (
    BOARD_COVERAGE,
    MIN_FAIR_PROB,
    TD_MARKET,
    TD_POSITIONS,
    TouchdownPropsModel,
    build_td_history,
    expected_game_tds,
    fair_yes_probabilities,
    yes_outcomes,
)
from betting_agent.sports.registry import get_sport_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RECENT_GAMES_FOR_PAYLOAD = 8
LADDER_STRATEGY = "ladder"
OVERS_STRATEGY = "overs"


def ladder_markets() -> list[str]:
    """Base markets whose ladders the run prices (settings.ladder_markets)."""
    wanted = [m.strip() for m in settings.ladder_markets.split(",") if m.strip()]
    return [m for m in wanted if m in DISTRIBUTION_MARKETS]


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


def _td_recent(model: TouchdownPropsModel, player_key: str) -> list[float]:
    entry = model._players.get(player_key)
    if entry is None:
        return []
    return [float(v) for v in entry[1][-RECENT_GAMES_FOR_PAYLOAD:]]


def generate_td_candidates(
    events: list[dict],
    model: TouchdownPropsModel,
    bankroll: float,
    season: int,
    current_teams: dict[str, str] | None = None,
    schedule: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    qb1: dict | None = None,
    book_order: list[str] | None = None,
    per_game: int | None = None,
    min_edge: float | None = None,
) -> list[BetCandidate]:
    """
    Anytime-TD "yes" scorers, best edge first. The board (every Yes price the
    book posted for the game) is de-vigged as a whole against the touchdowns
    the market expects from the game's spread/total (schedule lines, free),
    and the model's P(any TD) is compared with that fair price.

    Per game: the best-edge player is ALWAYS returned (the card's TD scorer,
    like the game lean) — as a PICK with a paper stake when the edge is
    inside [floor, cap), otherwise as a LEAN at stake 0 (`extra["td_pick"]`
    tells them apart). Further players, up to `per_game`, only when they are
    picks. Players at or above the cap are dropped (the diagnostic shows the
    model, not the market, is wrong there), and so are players the market
    prices under MIN_FAIR_PROB (long shots: favourite-longshot bias plus
    depth-chart facts the model cannot see). Kept separate from the
    receiving candidates: never deduped against them, never displacing them
    from the card, same-game Kelly scaling counts TD picks only.
    """
    from betting_agent.intelligence.picks import _apply_same_game_correlation_adjustment
    from betting_agent.sports.nfl.injuries import apply_injury_policy, prop_injury_flags
    from betting_agent.sports.teams import canonical_team

    current_teams = current_teams or {}
    per_game = per_game or settings.td_props_per_game
    game_lines = game_lines_from_schedule(events, schedule)
    floor = edge_floor(TD_MARKET, min_edge)
    cap = edge_cap(TD_MARKET)
    out: list[BetCandidate] = []
    for event in events:
        home, away = event.get("home_team", ""), event.get("away_team", "")
        home_ab, away_ab = canonical_team("NFL", home), canonical_team("NFL", away)
        event_date = _event_date(event)
        week = nfl_week_for(event_date, season, schedule, home_ab, away_ab)
        spread, total = game_lines.get(str(event.get("id")), (None, None))
        exp_home, exp_away = expected_game_tds(spread, total)
        expected = {home_ab: exp_home, away_ab: exp_away}

        books = books_in_preference(event, book_order)
        market = next((m for b in books for m in b.get("markets", [])
                       if m.get("key") == TD_MARKET and m.get("outcomes")), None)
        if market is None:
            continue
        book_key = next(b.get("key") for b in books
                        if any(m is market for m in b.get("markets", [])))
        prices = {pl: int(o["price"]) for pl, o in yes_outcomes(market).items()}
        fair, hold = fair_yes_probabilities(prices, BOARD_COVERAGE * (exp_home + exp_away))
        logger.info("%s @ %s anytime-TD board: %d players, %.2f expected TDs%s, hold %.0f%% (%s)",
                    away, home, len(prices), exp_home + exp_away,
                    "" if total is not None else " (no market total — league average)",
                    hold * 100, book_key)

        game_cands: list[BetCandidate] = []
        for player, price in prices.items():
            pk = normalize_player(player)
            team = current_teams.get(pk)
            if team is None:
                entry = model._players.get(pk)
                team = entry[3][-1] if entry else None
            if team not in expected:
                continue
            opponent = away_ab if team == home_ab else home_ab
            proj = model.project(pk, season, week, opponent=opponent, team=team,
                                 team_expected_tds=expected[team])
            if proj is None:
                continue
            edge = proj.prob - fair[player]
            if fair[player] < MIN_FAIR_PROB or (cap is not None and edge >= cap):
                continue
            is_pick = edge >= floor
            kelly, bet = recommended_bet(proj.prob, price, edge, bankroll) if is_pick else (0.0, 0.0)
            game_cands.append(BetCandidate(
                game_id=0, external_id=event.get("id"), home_team=home, away_team=away,
                game_date=date.today(), scheduled_game_date=event_date, sport="NFL",
                bet_type="prop", pick_side="yes", player=player, market=TD_MARKET, line=0.5,
                model_prob=proj.prob, implied_prob=fair[player], edge=edge, odds=price,
                kelly_fraction=kelly, recommended_bet=bet, bankroll_at_pick=bankroll,
                extra={"projection_mean": round(proj.rate, 3), "projection_games": proj.games,
                       "bookmaker": book_key, "team": team, "board_hold": round(hold, 3),
                       "expected_game_tds": round(exp_home + exp_away, 2),
                       "recent_values": _td_recent(model, pk),
                       "td_pick": is_pick},
            ))
        game_cands.sort(key=lambda c: c.edge, reverse=True)
        # The best scorer always survives (the card's TD lean); the rest
        # only as picks, up to per_game.
        out.extend(game_cands[:1] + [c for c in game_cands[1:per_game] if c.extra["td_pick"]])

    if injuries is not None or qb1:
        flags = prop_injury_flags(out, injuries, qb1, player_teams(out))
        out = apply_injury_policy(out, flags)
    _apply_same_game_correlation_adjustment(out)
    out.sort(key=lambda c: c.edge, reverse=True)
    return out


def _player_team(player_key: str, current_teams: dict[str, str], model) -> str | None:
    team = current_teams.get(player_key)
    if team is None and getattr(model, "history", None) is not None:
        rows = model.history[model.history["player_key"] == player_key]
        team = rows["team"].iloc[-1] if len(rows) else None
    return team


def generate_ladder_candidates(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    bankroll: float,
    season: int,
    current_teams: dict[str, str] | None = None,
    schedule: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    qb1: dict | None = None,
    book_order: list[str] | None = None,
    exclude_players: set[str] | None = None,
    markets: list[str] | None = None,
    min_edge: float | None = None,
) -> list[BetCandidate]:
    """
    Ladder hits — the card's "best overs": for every player on the book's
    alternate board, the rung (60+ yards, 6+ receptions, 40+ rushing) where
    the model's P(hit) most exceeds the fair price, one rung per player.

    The alternate boards are Over-only, so a rung cannot be de-vigged
    against its own Under: the hold is measured from the player's main-line
    pair on the same book (DEFAULT_LADDER_HOLD when there is none) and
    divided out of every rung. The main-line Over itself is a rung too, fair
    from its pair, and qualifies at any line — it is the book's own number for
    that player. Alternate rungs below LADDER_MIN_RUNG are not milestones
    (10+ yards for a backup) and are skipped; so are rungs the market prices outside
    [LADDER_MIN_FAIR_PROB, LADDER_MAX_FAIR_PROB] (long shots carry the
    favourite-longshot hold a uniform de-vig cannot see; chalk is not a
    milestone), and any rung the model likes by the market's ladder_edge_cap
    or more — it is wrong there, not the book. P(hit) comes from the
    ladder's own projection (`project_ladder`: lighter shrinkage — the main
    projection pulls a WR1 far under his own average, which is fine for the
    unders card and fatal for overs) through the tail calibrator
    (`Projection.prob_hit`), not the main-line one.

    Players in `exclude_players` (the main card's candidates, mostly unders)
    never appear here, so the card never argues with itself. Sized against
    the ladder's own bankroll and tagged strategy="ladder": a separate paper
    book from the receiving picks.

    Every entry is a PICK with a paper stake (user's call, Sep 8 2026: an
    experimental pool, "I don't care if overall long term they lose money").
    The game's best over is always returned whenever the model has any edge
    on it, so the section is never empty for a game that has a board;
    further players need to clear the (low) ladder floor.
    """
    from betting_agent.intelligence.picks import _apply_same_game_correlation_adjustment
    from betting_agent.sports.nfl.injuries import apply_injury_policy, prop_injury_flags
    from betting_agent.sports.teams import canonical_team

    current_teams = current_teams or {}
    exclude = set(exclude_players or ())
    markets = [m for m in (markets or ladder_markets()) if m in models]
    out: list[BetCandidate] = []
    for event in events:
        home, away = event.get("home_team", ""), event.get("away_team", "")
        home_ab, away_ab = canonical_team("NFL", home), canonical_team("NFL", away)
        event_date = _event_date(event)
        week = nfl_week_for(event_date, season, schedule, home_ab, away_ab)
        books = books_in_preference(event, book_order)
        if not books:
            continue
        book = books[0]
        by_key = {m.get("key"): m for m in book.get("markets", []) if m.get("outcomes")}

        game_cands: list[BetCandidate] = []
        for base in markets:
            model = models[base]
            floor, cap = ladder_edge_floor(base, min_edge), ladder_edge_cap(base)
            min_rung = ladder_min_rung(base)
            main_pairs = pair_outcomes(by_key[base]) if base in by_key else {}
            alt_pairs = pair_outcomes(by_key[ALTERNATE_MARKETS[base]]) \
                if ALTERNATE_MARKETS[base] in by_key else {}
            if not main_pairs and not alt_pairs:
                continue

            # (player, rung) → (price, fair P(over), market key it came from)
            rungs: dict[tuple[str, float], tuple[int, float, str]] = {}
            holds: dict[str, float] = {}
            for (player, line), pair in main_pairs.items():
                over, under = pair.get("Over"), pair.get("Under")
                if not over or not under:
                    continue
                p_o, p_u = (american_to_implied_prob(over["price"]),
                            american_to_implied_prob(under["price"]))
                holds[normalize_player(player)] = p_o + p_u - 1.0
                rungs[(player, line)] = (int(over["price"]), remove_vig(p_o, p_u)[0], base)
            for (player, line), pair in alt_pairs.items():
                over = pair.get("Over")
                if not over or (player, line) in rungs:
                    continue
                hold = holds.get(normalize_player(player), DEFAULT_LADDER_HOLD)
                fair = american_to_implied_prob(over["price"]) / (1.0 + hold)
                rungs[(player, line)] = (int(over["price"]), fair, ALTERNATE_MARKETS[base])

            projections: dict[str, object] = {}
            for (player, line), (price, fair, mkey) in rungs.items():
                pk = normalize_player(player)
                # The milestone floor is for the alternate rungs (10+ yards for a
                # backup is depth-chart noise); the book's own main-line Over is a
                # real over at any number.
                if pk in exclude or (mkey != base and line < min_rung) \
                        or not LADDER_MIN_FAIR_PROB <= fair <= LADDER_MAX_FAIR_PROB:
                    continue
                if pk not in projections:
                    team = _player_team(pk, current_teams, model)
                    opponent = (away_ab if team == home_ab else home_ab
                                if team == away_ab else None)
                    projections[pk] = (model.project_ladder(pk, season, week, opponent=opponent),
                                       team if opponent else None)
                proj, team = projections[pk]
                if proj is None:
                    continue
                p_hit = proj.prob_hit(line)
                edge = p_hit - fair
                if edge <= 0 or edge >= cap:
                    continue
                kelly, bet = recommended_bet(p_hit, price, edge, bankroll)
                game_cands.append(BetCandidate(
                    game_id=0, external_id=event.get("id"), home_team=home, away_team=away,
                    game_date=date.today(), scheduled_game_date=event_date, sport="NFL",
                    bet_type="prop", pick_side="over", player=player, market=mkey, line=line,
                    model_prob=p_hit, implied_prob=fair, edge=edge, odds=price,
                    kelly_fraction=kelly, recommended_bet=bet, bankroll_at_pick=bankroll,
                    strategy=LADDER_STRATEGY,
                    extra={"projection_mean": round(proj.mean, 2),
                           "projection_games": proj.games,
                           "bookmaker": book.get("key"), "team": team,
                           "base_market": base,
                           "book_hold": round(holds.get(pk, DEFAULT_LADDER_HOLD), 3),
                           "recent_values": _recent_values(model, pk),
                           "edge_floor": floor},
                ))
        # One rung per player (best edge across rungs and markets); the game's
        # best over is always kept, the rest must clear their market's floor.
        game_cands = sorted(_deduplicate_by_player(game_cands), key=lambda c: c.edge, reverse=True)
        out.extend(game_cands[:1] + [c for c in game_cands[1:] if c.edge >= c.extra["edge_floor"]])

    if injuries is not None or qb1:
        flags = prop_injury_flags(out, injuries, qb1, player_teams(out))
        out = apply_injury_policy(out, flags)
    _apply_same_game_correlation_adjustment(out)
    out.sort(key=lambda c: c.edge, reverse=True)
    return out


def generate_over_candidates(
    events: list[dict],
    models: dict[str, ReceivingPropsModel],
    bankroll: float,
    season: int,
    current_teams: dict[str, str] | None = None,
    schedule: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    qb1: dict | None = None,
    book_order: list[str] | None = None,
    exclude_players: set[str] | None = None,
    min_edge: float | None = None,
) -> list[BetCandidate]:
    """
    Straight overs — the book's main-line Over on each receiving market,
    fair from its own Over/Under pair, P(over) from the ladder projection
    (`project_ladder` + `prob_hit`: the main projection is built to find
    unders and sits under the book on most starters). One over per player;
    players in `exclude_players` (the main card) never appear.

    User's call (Sep 8 2026): at least one straight over per game. The best
    over by edge is ALWAYS returned — Kelly-staked when the model has an
    edge, otherwise flat at OVERS_MIN_STAKE_PCT of the overs bankroll
    (`extra["flat_stake"]`) so the selection is still tracked. Further overs
    in the game need `min_edge` (OVERS_EDGE_FLOOR). Rungs the model likes by
    the market's ladder cap or more are dropped (it is wrong there). Own paper
    book: strategy="overs".
    """
    from betting_agent.intelligence.picks import _apply_same_game_correlation_adjustment
    from betting_agent.sports.nfl.injuries import apply_injury_policy, prop_injury_flags
    from betting_agent.sports.teams import canonical_team

    current_teams = current_teams or {}
    exclude = set(exclude_players or ())
    floor = OVERS_EDGE_FLOOR if min_edge is None else min_edge
    out: list[BetCandidate] = []
    for event in events:
        home, away = event.get("home_team", ""), event.get("away_team", "")
        home_ab, away_ab = canonical_team("NFL", home), canonical_team("NFL", away)
        event_date = _event_date(event)
        week = nfl_week_for(event_date, season, schedule, home_ab, away_ab)
        books = books_in_preference(event, book_order)
        if not books:
            continue
        book = books[0]
        by_key = {m.get("key"): m for m in book.get("markets", []) if m.get("outcomes")}

        game_cands: list[BetCandidate] = []
        for market in MODELED_MARKETS:
            if market not in models or market not in by_key:
                continue
            model = models[market]
            cap = ladder_edge_cap(market)
            projections: dict[str, object] = {}
            for (player, line), pair in pair_outcomes(by_key[market]).items():
                over, under = pair.get("Over"), pair.get("Under")
                if not over or not under:
                    continue
                pk = normalize_player(player)
                if pk in exclude:
                    continue
                p_o, p_u = (american_to_implied_prob(over["price"]),
                            american_to_implied_prob(under["price"]))
                fair = remove_vig(p_o, p_u)[0]
                if pk not in projections:
                    team = _player_team(pk, current_teams, model)
                    opponent = (away_ab if team == home_ab else home_ab
                                if team == away_ab else None)
                    projections[pk] = (model.project_ladder(pk, season, week, opponent=opponent),
                                       team if opponent else None)
                proj, team = projections[pk]
                if proj is None:
                    continue
                p_hit = proj.prob_hit(line)
                edge = p_hit - fair
                if edge >= cap:
                    continue
                price = int(over["price"])
                kelly, bet = recommended_bet(p_hit, price, edge, bankroll) if edge > 0 else (0.0, 0.0)
                game_cands.append(BetCandidate(
                    game_id=0, external_id=event.get("id"), home_team=home, away_team=away,
                    game_date=date.today(), scheduled_game_date=event_date, sport="NFL",
                    bet_type="prop", pick_side="over", player=player, market=market, line=line,
                    model_prob=p_hit, implied_prob=fair, edge=edge, odds=price,
                    kelly_fraction=kelly, recommended_bet=bet, bankroll_at_pick=bankroll,
                    strategy=OVERS_STRATEGY,
                    extra={"projection_mean": round(proj.mean, 2),
                           "projection_games": proj.games,
                           "bookmaker": book.get("key"), "team": team,
                           "book_hold": round(p_o + p_u - 1.0, 3),
                           "recent_values": _recent_values(model, pk),
                           "edge_floor": floor, "flat_stake": False},
                ))
        game_cands = sorted(_deduplicate_by_player(game_cands), key=lambda c: c.edge, reverse=True)
        if not game_cands:
            continue
        best = game_cands[0]
        if best.recommended_bet <= 0:
            best.kelly_fraction = OVERS_MIN_STAKE_PCT
            best.recommended_bet = round(bankroll * OVERS_MIN_STAKE_PCT, 2)
            best.extra["flat_stake"] = True
        out.append(best)
        out.extend(c for c in game_cands[1:] if c.edge >= floor)

    if injuries is not None or qb1:
        flags = prop_injury_flags(out, injuries, qb1, player_teams(out))
        out = apply_injury_policy(out, flags)
    _apply_same_game_correlation_adjustment(out)
    out.sort(key=lambda c: c.edge, reverse=True)
    return out


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


def _print_td_scorers(td: list[BetCandidate], shadow: bool) -> None:
    for c in td:
        tag = "PICK (paper)" if c.extra.get("td_pick") else "LEAN (below floor, stake 0)"
        verdict = c.agent_verdict or ""
        if verdict and verdict != "SKIPPED" and shadow:
            verdict += " (SHADOW)"
        print(f"    {c.player:<24} anytime TD {c.odds:>+6} at {c.extra.get('bookmaker', '?'):<10} "
              f"model {c.model_prob:5.1%} vs fair {c.implied_prob:5.1%}  edge {c.edge:+.1%}  "
              f"{tag}  {verdict}")
        for flag in c.extra.get("flags", []):
            print(f"{'':<28} ! {flag.get('detail', '')}")
        for reason in (c.extra.get("agent") or {}).get("reasons", [])[:2]:
            print(f"{'':<28} > {reason}")


def _print_overs(overs: list[BetCandidate], shadow: bool) -> None:
    print(f"    {'straight over':<40} {'odds':>6} {'model':>7} {'fair':>7} {'edge':>7} "
          f"{'paper $':>8}  {'book':<11} tag / verdict")
    for c in overs:
        verdict = c.agent_verdict or ""
        if verdict and verdict != "SKIPPED" and shadow:
            verdict += " (SHADOW)"
        tag = ("PICK (flat 1%, no model edge)" if c.extra.get("flat_stake")
               else "PICK (paper, overs bankroll)")
        print(f"    {over_label(c.player, c.market, c.line):<40} {c.odds:>+6} "
              f"{c.model_prob:>6.1%} {c.implied_prob:>6.1%} {c.edge:>+6.1%} "
              f"{c.recommended_bet:>8.2f}  {(c.extra.get('bookmaker') or ''):<11} {tag}  {verdict}")
        for flag in c.extra.get("flags", []):
            print(f"{'':<28} ! {flag.get('detail', '')}")
        for reason in (c.extra.get("agent") or {}).get("reasons", [])[:2]:
            print(f"{'':<28} > {reason}")


def _print_ladder(ladder: list[BetCandidate], shadow: bool) -> None:
    print(f"    {'ladder hit':<40} {'odds':>6} {'model':>7} {'fair':>7} {'edge':>7} "
          f"{'paper $':>8}  {'book':<11} tag / verdict")
    for c in ladder:
        verdict = c.agent_verdict or ""
        if verdict and verdict != "SKIPPED" and shadow:
            verdict += " (SHADOW)"
        tag = "PICK (paper, ladder bankroll)"
        print(f"    {ladder_label(c.player, c.market, c.line):<40} {c.odds:>+6} "
              f"{c.model_prob:>6.1%} {c.implied_prob:>6.1%} {c.edge:>+6.1%} "
              f"{c.recommended_bet:>8.2f}  {(c.extra.get('bookmaker') or ''):<11} {tag}  {verdict}")
        for flag in c.extra.get("flags", []):
            print(f"{'':<28} ! {flag.get('detail', '')}")
        for reason in (c.extra.get("agent") or {}).get("reasons", [])[:2]:
            print(f"{'':<28} > {reason}")


def _print_slate(slate: Slate, leans: list[BetCandidate], props: list[BetCandidate],
                 off_card: int, shadow: bool, td: list[BetCandidate] | None = None,
                 td_off_card: int = 0, ladder: list[BetCandidate] | None = None,
                 ladder_off_card: int = 0, ladder_bankroll: float | None = None,
                 overs: list[BetCandidate] | None = None, overs_off_card: int = 0,
                 overs_bankroll: float | None = None) -> None:
    print(f"\n{'=' * 78}\n  {slate.title()}  ({slate.date})\n{'=' * 78}")
    if leans:
        print(f"  Game lean{'s' if len(leans) > 1 else ''} (market view vs {REFERENCE_BOOK}, "
              "NOT a pick — paper only, tracked for CLV):")
        for c in leans:
            print(f"    {c.pick_side:<32} {c.odds:>+5} at {c.extra.get('bookmaker', '?'):<10} "
                  f"fair {c.model_prob:5.1%} vs {c.implied_prob:5.1%}  edge {c.edge:+.1%}")
    else:
        print(f"  Game lean: none ({REFERENCE_BOOK} or a bettable book did not quote)")
    if td:
        print(f"  TD scorer{'s' if len(td) > 1 else ''} (best edge on the de-vigged Yes board; "
              f"PICK inside the {edge_floor(TD_MARKET):.0%}-{edge_cap(TD_MARKET):.0%} window, "
              f"else LEAN{f'; {td_off_card} more saved off-card' if td_off_card else ''}):")
        _print_td_scorers(td, shadow)
    if props:
        print(f"  Props ({len(props)} on card, {off_card} more saved off-card):")
        _print_candidates(props, shadow)
    else:
        print(f"  Props: none clear the floors ({off_card} saved off-card)" if off_card
              else "  Props: none clear the floors")
    if overs:
        bank = f", own bankroll ${overs_bankroll:,.2f}" if overs_bankroll is not None else ""
        print(f"  Straight overs — the book's main-line Over, best per game always a pick "
              f"({len(overs)} on card, {overs_off_card} more saved off-card{bank}):")
        _print_overs(overs, shadow)
    elif overs is not None:
        print("  Straight overs: no main-line Over the model could price")
    if ladder:
        bank = f", own bankroll ${ladder_bankroll:,.2f}" if ladder_bankroll is not None else ""
        print(f"  Ladder hits — overs / milestones on the alternate boards, experimental pick "
              f"pool ({len(ladder)} on card, {ladder_off_card} more saved off-card{bank}):")
        _print_ladder(ladder, shadow)
    elif ladder is not None:
        print("  Ladder hits: no board to price (no alternate markets posted)")


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
    parser.add_argument("--max-picks", type=int, default=10,
                        help="Main-card prop candidates to keep PER SLATE. The "
                             "cut is per kickoff window, not per day: a global "
                             "cut starves the later windows (see "
                             "intelligence/slate.py cap_per_slate)")
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
    parser.add_argument("--no-td", action="store_true",
                        help="Skip the anytime-TD board (saves 1 credit per game)")
    parser.add_argument("--no-ladder", action="store_true",
                        help="Skip the ladder hits (alternate boards; saves 1 credit "
                             "per game per ladder market)")
    parser.add_argument("--ladder-bankroll", type=float, default=None,
                        help=f"Ladder section's paper bankroll (default {settings.ladder_bankroll})")
    parser.add_argument("--no-overs", action="store_true",
                        help="Skip the straight-overs section (no credits involved)")
    parser.add_argument("--overs-bankroll", type=float, default=None,
                        help=f"Straight-overs paper bankroll (default {settings.overs_bankroll})")
    parser.add_argument("--no-leans", action="store_true",
                        help="Skip the game-market leans (saves the 3-credit "
                             "sport-level odds call)")
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
    fetch_td = settings.td_props_enabled and not args.no_td
    fetch_ladder = settings.ladder_enabled and not args.no_ladder
    ladder_bankroll = args.ladder_bankroll or settings.ladder_bankroll
    run_overs = settings.overs_enabled and not args.no_overs
    overs_bankroll = args.overs_bankroll or settings.overs_bankroll
    ladder_bases = ladder_markets() if fetch_ladder else []
    # The ladder shares the receiving models (their tail calibrators) and
    # adds a rushing-yards model of its own — ladder only, no main-card use.
    ladder_models = {m: models[m] for m in ladder_bases if m in models}
    if "player_rush_yds" in ladder_bases:
        rushing = build_rushing_history(stats)
        if not rushing.empty:
            rush_model = ReceivingPropsModel("player_rush_yds").fit(rushing)
            rush_model.tune_dispersion([season - 1])
            ladder_models["player_rush_yds"] = rush_model

    # Season context, all free: roster (offseason movers), schedule (exact
    # week, spread/total), injury report + depth charts (QB1).
    from betting_agent.sports.nfl.injuries import load_injury_report, qb1_by_team
    from betting_agent.sports.nfl.props import current_teams, load_season_schedule

    roster = current_teams(season, TD_POSITIONS if (fetch_td or fetch_ladder) else RECEIVING_POSITIONS)
    if roster:
        last_team = history.sort_values("t").groupby("player_key")["team"].last()
        moved = sum(1 for pk, t in last_team.items() if roster.get(pk) not in (None, t))
        logger.info("Roster overlay: %d players, %d re-teamed vs their last stats row",
                    len(roster), moved)
    schedule = load_season_schedule(season)
    td_model: TouchdownPropsModel | None = None
    if fetch_td:
        # TDs are rare: two more seasons (free, cached) for the position
        # priors and the isotonic layer, mirroring td_props_diagnostic.py
        # (fit on four seasons, calibrate on the last two of them).
        td_stats = pd.concat([load_player_stats([season - 4, season - 3]), stats],
                             ignore_index=True)
        td_model = TouchdownPropsModel().fit(build_td_history(td_stats))
        td_model.calibrate([season - 2, season - 1],
                           schedule=pd.concat([load_season_schedule(season - 2),
                                               load_season_schedule(season - 1)],
                                              ignore_index=True))
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
    markets = list(MODELED_MARKETS) + ([TD_MARKET] if fetch_td else [])
    markets += [ALTERNATE_MARKETS[m] for m in ladder_models]
    logger.info("Fetching prop odds (books: %s; markets: %s)...", ",".join(books), ",".join(markets))
    events = fetch_prop_odds(markets=markets, bookmakers=books, max_events=args.max_events,
                             events=chosen)
    if not events:
        raise SystemExit("No prop odds returned — check ODDS_API_KEY / season timing.")

    candidates = generate_prop_candidates(
        events, models, bankroll, min_edge, season,
        current_teams=roster, schedule=schedule, injuries=injuries, qb1=qb1,
        book_order=books,
    )
    # Per slate, not per day — a global cut by edge can spend its whole
    # allowance on the 1pm window and leave Sunday night with nothing to card.
    slates = group_events_by_slate(events)
    candidates = cap_per_slate(candidates, slates, args.max_picks)
    td_picks: list[BetCandidate] = []
    if td_model is not None:
        td_picks = generate_td_candidates(
            events, td_model, bankroll, season, current_teams=roster, schedule=schedule,
            injuries=injuries, qb1=qb1, book_order=books, min_edge=min_edge,
        )
    # Straight overs first (the book's own number), then the ladder on the
    # players neither the main card nor the overs took — the card never
    # carries the same player twice.
    over_picks: list[BetCandidate] = []
    if run_overs:
        over_picks = generate_over_candidates(
            events, models, overs_bankroll, season, current_teams=roster,
            schedule=schedule, injuries=injuries, qb1=qb1, book_order=books,
            exclude_players={normalize_player(c.player) for c in candidates},
        )
    ladder_picks: list[BetCandidate] = []
    if ladder_models:
        ladder_picks = generate_ladder_candidates(
            events, ladder_models, ladder_bankroll, season, current_teams=roster,
            schedule=schedule, injuries=injuries, qb1=qb1, book_order=books,
            exclude_players={normalize_player(c.player) for c in candidates + over_picks},
            markets=list(ladder_models), min_edge=None,
        )
    # The card's TD scorers: like the game lean, up to lean_cap per slate,
    # best edge first; the rest of the TD entries are saved off-card. The
    # overs and ladder sub-sections each take the slate's prop cap.
    td_card: list[BetCandidate] = []
    ladder_card: list[BetCandidate] = []
    overs_card: list[BetCandidate] = []
    for slate in slates:
        td_card += select_card(td_picks, slate, slate.lean_cap)
        overs_card += select_card(over_picks, slate, slate.prop_cap)
        ladder_card += select_card(ladder_picks, slate, slate.prop_cap)

    if not candidates:
        print("\nNo prop edges clear the threshold today.")

    # LLM validator — shadow by default: verdicts are recorded and shown but
    # do not touch edge, sizing, or the slate.
    agent_summary: dict | None = None
    validation_records = []
    if args.agent_mode != "off":
        try:
            from betting_agent.intelligence.validator import validate_picks

            on_cards = candidates + td_card + overs_card + ladder_card
            validated, validation = validate_picks(
                on_cards, sport="NFL", mode=args.agent_mode,
                injuries=injuries, qb1=qb1, player_teams=player_teams(on_cards),
                game_lines=game_lines_from_schedule(events, schedule),
            )
            candidates = [c for c in validated
                          if c.market != TD_MARKET and c.strategy is None]
            td_card = [c for c in validated if c.market == TD_MARKET]
            overs_card = [c for c in validated if c.strategy == OVERS_STRATEGY]
            ladder_card = [c for c in validated if c.strategy == LADDER_STRATEGY]
            validation_records = validation.records
            agent_summary = validation.as_dict()
        except Exception as exc:
            logger.warning("Validator failed, continuing without it: %s", exc)

    shadow = bool(agent_summary and agent_summary.get("shadow"))

    # ---- Cards: one per slate. Leans = the market's sharpest price vs the
    # bettable book (paper only, tracked for CLV); TD scorers; props = the picks. ----
    leans: list[BetCandidate] = []
    if not args.no_leans:
        try:
            lines = fetch_game_lines(events, books + [REFERENCE_BOOK])
            leans = game_leans(lines, books, bankroll)
        except Exception as exc:
            logger.warning("Game leans unavailable: %s", exc)
    cards = []
    for slate in slates:
        card_props = select_card(candidates, slate, slate.prop_cap)
        card_leans = select_card(leans, slate, slate.lean_cap)
        off_card = len(candidates_in_slate(candidates, slate)) - len(card_props)
        slate_td = candidates_in_slate(td_card, slate)
        td_off_card = len(candidates_in_slate(td_picks, slate)) - len(slate_td)
        slate_ladder = candidates_in_slate(ladder_card, slate)
        ladder_off_card = len(candidates_in_slate(ladder_picks, slate)) - len(slate_ladder)
        slate_overs = candidates_in_slate(overs_card, slate)
        overs_off_card = len(candidates_in_slate(over_picks, slate)) - len(slate_overs)
        cards.append((slate, card_leans, card_props, off_card, slate_td, slate_ladder, ladder_off_card,
                      slate_overs, overs_off_card))
        _print_slate(slate, card_leans, card_props, off_card, shadow, slate_td, td_off_card,
                     ladder=slate_ladder if ladder_models else None,
                     ladder_off_card=ladder_off_card, ladder_bankroll=ladder_bankroll,
                     overs=slate_overs if run_overs else None,
                     overs_off_card=overs_off_card, overs_bankroll=overs_bankroll)

    if agent_summary:
        print(f"\nValidator{' (SHADOW — verdicts recorded, stakes untouched)' if shadow else ''}: "
              f"{agent_summary['validated_games']} games, "
              f"{agent_summary['skipped_games']} skipped, "
              f"${agent_summary['total_cost_usd']:.4f}")
    n_td_picks = sum(1 for c in td_picks if c.extra.get("td_pick"))
    print(f"\n{len(candidates)} paper prop picks ({sum(len(c[2]) for c in cards)} on cards), "
          f"{len(td_picks)} TD scorers ({n_td_picks} inside the window), "
          f"{len(over_picks)} straight overs ({sum(len(c[7]) for c in cards)} on cards, own bankroll), "
          f"{len(ladder_picks)} ladder hits ({sum(len(c[5]) for c in cards)} on cards, "
          "own bankroll), "
          f"{len(leans)} game leans. "
          "These are NOT bets — Phase 3 validates the projections first.")

    if args.save:
        save_picks_to_db(candidates + td_picks + over_picks + ladder_picks + leans)
        print("Saved to picks table (props + TD scorers + overs + ladder + leans; "
              "on_card marks the card).")
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
                send_slate_to_discord,
            )
            if is_discord_configured("NFL", "PICKS"):
                for (slate, card_leans, card_props, off_card, slate_td, slate_ladder, ladder_off,
                     slate_overs, overs_off) in cards:
                    if not any((card_leans, card_props, slate_td, slate_ladder, slate_overs)):
                        continue
                    logger.info("Sending %s card to Discord...", slate.label)
                    send_slate_to_discord(f"{slate.title()} — {slate.date}", card_leans, card_props,
                                          bankroll, "NFL", agent_summary=agent_summary,
                                          extra_saved=off_card, td_scorers=slate_td,
                                          ladder=slate_ladder, ladder_saved=ladder_off,
                                          ladder_bankroll=ladder_bankroll,
                                          overs=slate_overs, overs_saved=overs_off,
                                          overs_bankroll=overs_bankroll)
        except Exception as exc:
            logger.warning("Discord notification failed: %s", exc)


if __name__ == "__main__":
    main()
