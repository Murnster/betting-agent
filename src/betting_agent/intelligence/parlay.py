"""
Long-shot parlays: an experiment pool, not a validated edge.

User request (Sep 14 2026): a 3-leg same-game parlay for every primetime
game, a 3-leg cross-game parlay for every Sunday window, and a 3-5 leg
"lean parlay" of moneyline / spread / total sides across the Sunday games —
$1 lottery tickets whose premise is that a few hits over a season pay for
the misses. What the earlier rejection (TODO.md, Sep 10) still says: a
parlay compounds the model's calibration error the way it compounds edge
(4 legs claimed at 70% but truly 60% → 24% claimed vs 13% true), no book
posts a parlay price to the Odds API so the paper price is the product of
the leg prices (a real same-game parlay is priced BELOW that product because
the book discounts correlation), and one ticket a week is a weak measuring
instrument. So the book is scored the way the ladder is: did the legs hit at
the rate they claimed, and did it pay — never as evidence of an edge.

Every leg is recombined from lines the run already fetched: zero credits.

Coherence. A parlay must not argue with itself, so each leg carries a
*narrative sign* per team: an Over, an anytime TD, a moneyline or a cover is
`+team`; an Under on a team's player is `-team`; a game-total Over is plus
both teams and an Under minus both. Two legs with opposite signs on one team
are refused, as are two legs on one player, two moneyline/spread legs
backing different teams, and two legs on the same market of one game. What
remains are the parlays that read as one story: a shootout (all overs), a
grind (all unders), or a team story (T wins, T's receivers over, the
opponent's under).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date

from betting_agent.intelligence.picks import BetCandidate, _candidate_game_key, _pick_label
from betting_agent.intelligence.slate import Slate, candidates_in_slate
from betting_agent.sports.nfl.props import normalize_player
from betting_agent.sports.nfl.td_props import TD_MARKET
from betting_agent.sports.teams import canonical_team

logger = logging.getLogger(__name__)

PARLAY_STRATEGY = "parlay"
PARLAY_LEG_STRATEGY = "parlay_leg"

#: Pick.market of the parent row: "sgp" when every leg is one game (the
#: paper price is the product of the leg prices — optimistic, see module
#: doc), "parlay" for a cross-game ticket, whose product price is honest.
SGP_MARKET = "sgp"
PARLAY_MARKET = "parlay"

#: Policy knobs. Legs per ticket; the lean parlay takes 3-5 (5 when at least
#: five games have a positive-edge side); a ticket shorter than
#: PARLAY_MIN_ODDS is not a long shot and gets its shortest leg swapped for
#: the best plus-money leg that keeps it coherent.
PARLAY_LEGS_SGP = 3
PARLAY_LEGS_WINDOW = 3
PARLAY_LEAN_LEGS = (3, 5)
PARLAY_MIN_ODDS = 300
PARLAY_STAKE = 1.0
#: The main book's props are the picks the user tracks as real bets, so a
#: ticket may carry at most one of them — otherwise every primetime SGP is
#: the card parlayed (user, Sep 14 2026: "at most one main-book leg per
#: parlay ticket and also lean towards player props for the SGPs, and
#: unless it's very likely for a TD, avoid TD scorers").
PARLAY_MAX_MAIN_LEGS = 1
#: An anytime-TD leg needs the model this sure (P(score) at or above it).
PARLAY_TD_MIN_PROB = 0.50

GAME_BET_TYPES = ("moneyline", "spread", "total")


@dataclass
class Parlay:
    kind: str                  # "sgp" | "window" | "leans"
    parent: BetCandidate
    legs: list[BetCandidate] = field(default_factory=list)
    slate_key: str | None = None

    @property
    def same_game(self) -> bool:
        return self.parent.market == SGP_MARKET


# ---- odds arithmetic -------------------------------------------------------

def american_to_decimal(odds: int | float) -> float:
    odds = float(odds)
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def decimal_to_american(dec: float) -> int:
    if dec >= 2:
        return int(round((dec - 1) * 100))
    return int(round(-100 / (dec - 1)))


def combined_odds(legs: list[BetCandidate]) -> int:
    dec = math.prod(american_to_decimal(leg.odds) for leg in legs)
    return decimal_to_american(dec)


# ---- coherence ------------------------------------------------------------

_SPREAD_RE = re.compile(r"^(.*?)\s+[+-]\d+(\.\d+)?$")


def _team_of(leg: BetCandidate) -> str | None:
    """Canonical abbreviation of the team a leg is about (None for totals)."""
    sport = leg.sport or "NFL"
    if leg.bet_type == "prop":
        team = (leg.extra or {}).get("team")
        return canonical_team(sport, team) if team else None
    if leg.bet_type == "moneyline":
        return canonical_team(sport, leg.pick_side)
    if leg.bet_type == "spread":
        m = _SPREAD_RE.match(leg.pick_side.strip())
        return canonical_team(sport, m.group(1) if m else leg.pick_side)
    return None


def _sign(leg: BetCandidate) -> int:
    side = (leg.pick_side or "").lower()
    if leg.bet_type == "prop":
        return 1 if side in ("over", "yes") else -1
    if leg.bet_type == "total":
        return 1 if side.startswith("over") else -1
    return 1  # a moneyline or a cover backs its team


def leg_signs(leg: BetCandidate) -> dict[str, int]:
    """Narrative sign per team (see module doc). Empty when the team is unknown."""
    sport = leg.sport or "NFL"
    if leg.bet_type == "total":
        s = _sign(leg)
        return {t: s for t in (canonical_team(sport, leg.home_team),
                               canonical_team(sport, leg.away_team)) if t}
    team = _team_of(leg)
    return {team: _sign(leg)} if team else {}


def _backed_team(leg: BetCandidate) -> str | None:
    return _team_of(leg) if leg.bet_type in ("moneyline", "spread") else None


def conflicts(a: BetCandidate, b: BetCandidate) -> str | None:
    """Why two legs cannot share a ticket, or None when they can."""
    if a.player and b.player and normalize_player(a.player) == normalize_player(b.player):
        return "same player"
    same_game = _candidate_game_key(a) == _candidate_game_key(b)
    if same_game:
        if a.bet_type == b.bet_type and a.bet_type != "prop" and (a.market or "") == (b.market or ""):
            return f"two {a.bet_type} legs on one game"
        if a.bet_type == "prop" and b.bet_type == "prop" and not a.player and not b.player:
            return "two legs on one market"
        ta, tb = _backed_team(a), _backed_team(b)
        if ta and tb and ta != tb:
            return "backs both teams"
    sa, sb = leg_signs(a), leg_signs(b)
    for team, sign in sa.items():
        if sb.get(team, sign) != sign:
            return f"opposite stories on {team}"
    return None


def coherent(legs: list[BetCandidate], new: BetCandidate) -> bool:
    return all(conflicts(existing, new) is None for existing in legs)


# ---- building -------------------------------------------------------------

def _leg_copy(leg: BetCandidate) -> BetCandidate:
    """A leg row: same line, stake 0, off-card, tagged as a leg. Never the
    section's own candidate — that one keeps its own book and its own stake."""
    extra = dict(leg.extra or {})
    extra.pop("card", None)
    extra["leg_of"] = leg.strategy or ("td" if leg.market == TD_MARKET else
                                       "lean" if leg.bet_type in GAME_BET_TYPES else "props")
    return BetCandidate(
        game_id=leg.game_id, external_id=leg.external_id,
        home_team=leg.home_team, away_team=leg.away_team,
        game_date=leg.game_date, scheduled_game_date=leg.scheduled_game_date,
        sport=leg.sport, bet_type=leg.bet_type, pick_side=leg.pick_side, line=leg.line,
        player=leg.player, market=leg.market,
        model_prob=leg.model_prob, implied_prob=leg.implied_prob, edge=leg.edge, odds=leg.odds,
        kelly_fraction=0.0, recommended_bet=0.0, bankroll_at_pick=0.0,
        strategy=PARLAY_LEG_STRATEGY, extra=extra,
    )


def combine_legs(legs: list[BetCandidate], kind: str, stake: float, bankroll: float,
                 pick_date: date | None = None) -> BetCandidate:
    """The parent row: product price, product probabilities, flat stake."""
    games = {_candidate_game_key(leg) for leg in legs}
    same_game = len(games) == 1
    first = legs[0]
    model_prob = math.prod(leg.model_prob for leg in legs)
    implied_prob = math.prod(leg.implied_prob for leg in legs)
    label = {"sgp": "SGP", "window": "parlay", "leans": "lean parlay"}[kind]
    return BetCandidate(
        game_id=first.game_id, external_id=first.external_id,
        home_team=first.home_team, away_team=first.away_team,
        game_date=pick_date or first.game_date, scheduled_game_date=first.scheduled_game_date,
        sport=first.sport, bet_type="parlay",
        pick_side=f"{len(legs)}-leg {label}",
        market=SGP_MARKET if same_game else PARLAY_MARKET,
        model_prob=model_prob, implied_prob=implied_prob, edge=model_prob - implied_prob,
        odds=combined_odds(legs), kelly_fraction=0.0, recommended_bet=stake,
        bankroll_at_pick=bankroll, strategy=PARLAY_STRATEGY,
        extra={"card": True, "parlay": kind, "same_game": same_game,
               "legs": [_pick_label(leg) for leg in legs]},
    )


def is_main_book(c: BetCandidate) -> bool:
    """A main-book receiving prop: the picks tracked as real bets."""
    return c.bet_type == "prop" and c.strategy is None and c.market != TD_MARKET


def _tier(c: BetCandidate) -> int:
    """Player props first, then a TD favourite, then the game sides — the
    tickets lean towards props; a moneyline or total fills only what the
    props cannot."""
    if c.bet_type == "prop":
        return 1 if c.market == TD_MARKET else 0
    return 2


def _eligible(c: BetCandidate) -> bool:
    if c.edge <= 0:
        return False
    if c.market == TD_MARKET:
        return c.model_prob >= PARLAY_TD_MIN_PROB
    return True


def _positive(pool: list[BetCandidate]) -> list[BetCandidate]:
    return sorted((c for c in pool if _eligible(c)), key=lambda c: (_tier(c), -c.edge))


def _greedy(pool: list[BetCandidate], n: int, one_per_game: bool,
            max_main: int = PARLAY_MAX_MAIN_LEGS) -> list[BetCandidate]:
    legs: list[BetCandidate] = []
    games: set[str] = set()
    main = 0
    for c in pool:
        if len(legs) == n:
            break
        gk = _candidate_game_key(c)
        if one_per_game and gk in games:
            continue
        if is_main_book(c) and main >= max_main:
            continue
        if coherent(legs, c):
            legs.append(c)
            games.add(gk)
            main += is_main_book(c)
    return legs


def pick_legs(pool: list[BetCandidate], n: int, *, one_per_game: bool = False,
              min_odds: int = PARLAY_MIN_ODDS) -> list[BetCandidate]:
    """
    Greedy over the positive-edge pool — player props by edge first, a TD
    favourite next, the game sides last — coherent at every step and
    carrying at most PARLAY_MAX_MAIN_LEGS main-book legs. A ticket short of
    `min_odds` swaps its shortest-priced leg for the best
    plus-money leg that keeps it coherent (a long shot is the brief); when no
    swap gets it there the ticket stands as it is.
    """
    ranked = _positive(pool)
    legs = _greedy(ranked, n, one_per_game)
    if len(legs) < n:
        return []
    if combined_odds(legs) >= min_odds:
        return legs
    shortest = min(legs, key=lambda c: c.odds)
    rest = [c for c in legs if c is not shortest]
    used = {id(c) for c in legs}
    main_left = sum(is_main_book(c) for c in rest)
    for c in ranked:
        if id(c) in used or c.odds <= 0:
            continue
        if one_per_game and _candidate_game_key(c) in {_candidate_game_key(r) for r in rest}:
            continue
        if is_main_book(c) and main_left >= PARLAY_MAX_MAIN_LEGS:
            continue
        if coherent(rest, c):
            swapped = rest + [c]
            if combined_odds(swapped) >= min_odds:
                return swapped
    return legs


def build_parlays(slates: list[Slate], sections: list[list[BetCandidate]],
                  lean_sides: list[BetCandidate], *, bankroll: float,
                  stake: float = PARLAY_STAKE, pick_date: date | None = None) -> list[Parlay]:
    """
    One same-game parlay per single-game slate, one cross-game parlay per
    multi-game slate, and one lean parlay across every multi-game slate's
    games (the 19:00 primetime run has none, so it never rebuilds Sunday's).

    `sections` are the run's candidate lists (props, TD scorers, overs,
    ladder); `lean_sides` every bettable game-market side, not only the
    carded lean.
    """
    prop_pool = [c for section in sections for c in section]
    out: list[Parlay] = []
    for slate in slates:
        if slate.single_game:
            pool = candidates_in_slate(prop_pool + lean_sides, slate)
            legs = pick_legs(pool, PARLAY_LEGS_SGP)
            kind = "sgp"
        else:
            pool = candidates_in_slate(prop_pool, slate)
            legs = pick_legs(pool, PARLAY_LEGS_WINDOW, one_per_game=True)
            kind = "window"
        if legs:
            copies = [_leg_copy(leg) for leg in legs]
            out.append(Parlay(kind, combine_legs(copies, kind, stake, bankroll, pick_date),
                              copies, slate.key))
        else:
            logger.info("No coherent %d-leg %s for %s", PARLAY_LEGS_SGP, kind, slate.label)

    day_slates = [s for s in slates if not s.single_game]
    if day_slates:
        lo, hi = PARLAY_LEAN_LEGS
        pool = [c for s in day_slates for c in candidates_in_slate(lean_sides, s)]
        legs = pick_legs(pool, hi, one_per_game=True)
        if not legs:  # fewer than `hi` games with a positive side: take what there is
            legs = _greedy(_positive(pool), hi, True)
        if len(legs) >= lo:
            copies = [_leg_copy(leg) for leg in legs]
            out.append(Parlay("leans", combine_legs(copies, "leans", stake, bankroll, pick_date),
                              copies, None))
        else:
            logger.info("Lean parlay needs %d positive-edge games, found %d", lo, len(legs))
    return out
