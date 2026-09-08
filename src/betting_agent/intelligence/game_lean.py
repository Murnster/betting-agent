"""
Game-market leans: the market's own sharpest opinion, priced at the book
you can bet.

The NFL game models cannot beat the closing line (from scratch or anchored
on it — see TODO.md). What has not been tested is *timing*: a Tuesday
number is not the closing number. A lean takes Pinnacle's vig-removed
probability as the fair price, converts it to the bettable book's line, and
reports the edge of the book's price against that fair price. One lean per
game (best edge across moneyline / spread / total). Leans go on the card
labelled as leans, not picks, and are saved as zero-risk paper picks so
`props.py --closing` can measure whether they beat the close (CLV). If a
season of CLV runs positive the lean earns promotion; if not, the question
is answered without staking anything.

Spread/total probabilities move between lines through a normal margin/total
model with the sigmas measured by scripts/game_model_gate.py (2017-2025).
"""

from __future__ import annotations

import logging
from datetime import date

from scipy.stats import norm

from betting_agent.intelligence.ev import american_to_implied_prob, remove_vig
from betting_agent.intelligence.kelly import recommended_bet
from betting_agent.intelligence.picks import BetCandidate

logger = logging.getLogger(__name__)

REFERENCE_BOOK = "pinnacle"
LEAN_MARKETS = ("h2h", "spreads", "totals")
MARGIN_SIGMA = 12.8     # sd of (home margin − closing spread), 2017-2025
TOTAL_SIGMA = 13.3      # sd of (total − closing total), 2017-2025


def _devig(p_a: float | None, p_b: float | None) -> tuple[float, float] | None:
    if p_a is None or p_b is None:
        return None
    return remove_vig(american_to_implied_prob(p_a), american_to_implied_prob(p_b))


def fair_from_reference(ref: dict) -> dict:
    """
    Pinnacle's quote → fair parameters: home win probability, expected home
    margin mu_m (positive = home better) and expected total mu_t.
    Spread convention follows the Odds API: the home point is negative when
    home is favoured, and home covers when margin + point > 0.
    """
    out: dict = {}
    ml = _devig(ref.get("ml_home"), ref.get("ml_away"))
    if ml:
        out["home_prob"] = ml[0]
    sp = _devig(ref.get("spread_home_price"), ref.get("spread_away_price"))
    if sp and ref.get("spread_home_line") is not None:
        p_home_cover = min(max(sp[0], 0.02), 0.98)
        out["mu_margin"] = MARGIN_SIGMA * norm.ppf(p_home_cover) - float(ref["spread_home_line"])
    tot = _devig(ref.get("over_price"), ref.get("under_price"))
    if tot and ref.get("total_over_line") is not None:
        p_over = min(max(tot[0], 0.02), 0.98)
        out["mu_total"] = float(ref["total_over_line"]) + TOTAL_SIGMA * norm.ppf(p_over)
    return out


def _sides(event: dict, book: dict, fair: dict) -> list[dict]:
    """Every bettable side at `book` with its fair probability and edge."""
    home, away = event.get("home_team", ""), event.get("away_team", "")
    sides: list[dict] = []

    if "home_prob" in fair and book.get("ml_home") is not None and book.get("ml_away") is not None:
        bh, ba = remove_vig(american_to_implied_prob(book["ml_home"]),
                            american_to_implied_prob(book["ml_away"]))
        sides.append({"bet_type": "moneyline", "pick_side": home, "line": None,
                      "odds": int(book["ml_home"]), "model_prob": fair["home_prob"],
                      "implied_prob": bh})
        sides.append({"bet_type": "moneyline", "pick_side": away, "line": None,
                      "odds": int(book["ml_away"]), "model_prob": 1 - fair["home_prob"],
                      "implied_prob": ba})

    if ("mu_margin" in fair and book.get("spread_home_line") is not None
            and book.get("spread_home_price") is not None and book.get("spread_away_price") is not None):
        pt = float(book["spread_home_line"])
        p_home = float(norm.cdf((fair["mu_margin"] + pt) / MARGIN_SIGMA))
        bh, ba = remove_vig(american_to_implied_prob(book["spread_home_price"]),
                            american_to_implied_prob(book["spread_away_price"]))
        sides.append({"bet_type": "spread", "pick_side": f"{home} {pt:+.1f}", "line": pt,
                      "odds": int(book["spread_home_price"]), "model_prob": p_home,
                      "implied_prob": bh})
        sides.append({"bet_type": "spread", "pick_side": f"{away} {-pt:+.1f}", "line": -pt,
                      "odds": int(book["spread_away_price"]), "model_prob": 1 - p_home,
                      "implied_prob": ba})

    if ("mu_total" in fair and book.get("total_over_line") is not None
            and book.get("over_price") is not None and book.get("under_price") is not None):
        ln = float(book["total_over_line"])
        p_over = float(norm.cdf((fair["mu_total"] - ln) / TOTAL_SIGMA))
        bo, bu = remove_vig(american_to_implied_prob(book["over_price"]),
                            american_to_implied_prob(book["under_price"]))
        sides.append({"bet_type": "total", "pick_side": f"over {ln}", "line": ln,
                      "odds": int(book["over_price"]), "model_prob": p_over, "implied_prob": bo})
        sides.append({"bet_type": "total", "pick_side": f"under {ln}", "line": ln,
                      "odds": int(book["under_price"]), "model_prob": 1 - p_over,
                      "implied_prob": bu})

    for s in sides:
        s["edge"] = s["model_prob"] - s["implied_prob"]
    return sides


def _quotes(event: dict) -> dict[str, dict]:
    from betting_agent.api.odds import OddsAPIClient
    return {q["book_key"]: q for q in OddsAPIClient()._quotes_for_game(event, None)}


def game_leans(odds_events: list[dict], book_order: list[str], bankroll: float,
               reference: str = REFERENCE_BOOK, pick_date: date | None = None,
               ) -> list[BetCandidate]:
    """
    One lean per event: the best-edge side at the first book in `book_order`
    that quotes the game, against `reference`'s fair price. Events without a
    reference quote produce nothing (logged).
    """
    pick_date = pick_date or date.today()
    leans: list[BetCandidate] = []
    for event in odds_events:
        quotes = _quotes(event)
        ref = quotes.get(reference)
        if ref is None:
            logger.info("No %s quote for %s @ %s — no lean", reference,
                        event.get("away_team"), event.get("home_team"))
            continue
        book_key = next((b for b in book_order if b in quotes and b != reference), None)
        if book_key is None:
            logger.info("None of %s quote %s @ %s — no lean", book_order,
                        event.get("away_team"), event.get("home_team"))
            continue
        fair = fair_from_reference(ref)
        sides = _sides(event, quotes[book_key], fair)
        if not sides:
            continue
        best = max(sides, key=lambda s: s["edge"])
        kelly, stake = recommended_bet(best["model_prob"], best["odds"], best["edge"], bankroll)
        ct = event.get("commence_time")
        sched = date.fromisoformat(ct[:10]) if isinstance(ct, str) and len(ct) >= 10 else None
        leans.append(BetCandidate(
            game_id=0, external_id=event.get("id"),
            home_team=event.get("home_team", ""), away_team=event.get("away_team", ""),
            game_date=pick_date, scheduled_game_date=sched, sport="NFL",
            bet_type=best["bet_type"], pick_side=best["pick_side"], line=best["line"],
            model_prob=best["model_prob"], implied_prob=best["implied_prob"],
            edge=best["edge"], odds=best["odds"],
            kelly_fraction=kelly if best["edge"] > 0 else 0.0,
            recommended_bet=stake if best["edge"] > 0 else 0.0,
            bankroll_at_pick=bankroll,
            extra={"lean": True, "reference": reference, "bookmaker": book_key,
                   "reference_home_prob": fair.get("home_prob"),
                   "reference_mu_margin": fair.get("mu_margin"),
                   "reference_mu_total": fair.get("mu_total")},
        ))
    return leans


def fetch_game_lines(events: list[dict], bookmakers: list[str],
                     sport_key: str = "americanfootball_nfl") -> list[dict]:
    """Sport-level odds for these events at these books: 3 credits per call
    (one per market), regardless of the number of games or books (≤10)."""
    from betting_agent.api.odds import OddsAPIClient

    ids = [str(e["id"]) for e in events if e.get("id")]
    if not ids:
        return []
    return OddsAPIClient().fetch_odds(sport_key, markets=list(LEAN_MARKETS),
                                      event_ids=ids, bookmakers=bookmakers)
