"""
Long-shot parlays: an experiment pool with its own paper book and channel.

What these pin: a ticket never argues with itself (one leg per player, no
opposite stories on a team, no backing both sides of a game), the parent's
price and probabilities are the products of its legs, settlement follows the
book convention (any loss loses; push/void legs drop out), and nothing about
the book leaks into the main card, the main results post or the closing
capture.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from betting_agent.accounting import parlays as settle_mod
from betting_agent.intelligence import parlay as pm
from betting_agent.intelligence import picks as picks_mod
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.slate import Slate
from betting_agent.notifications import discord as d

SEA, NE = "Seattle Seahawks", "New England Patriots"
LAR, SF = "Los Angeles Rams", "San Francisco 49ers"


def _prop(player, side, team, *, odds=-110, edge=0.20, market="player_reception_yds",
          line=49.5, home=SEA, away=NE, ext="e1", strategy=None, card=False):
    return BetCandidate(
        game_id=0, external_id=ext, home_team=home, away_team=away,
        game_date=date(2026, 9, 13), scheduled_game_date=date(2026, 9, 13), sport="NFL",
        bet_type="prop", pick_side=side, player=player, market=market, line=line,
        model_prob=0.6 + edge, implied_prob=0.6, edge=edge, odds=odds,
        kelly_fraction=0.02, recommended_bet=1.5, bankroll_at_pick=100.0,
        strategy=strategy, extra={"team": team, "bookmaker": "draftkings", "card": card},
    )


def _game(bet_type, side, *, odds=-110, edge=0.02, line=None, home=SEA, away=NE, ext="e1"):
    return BetCandidate(
        game_id=0, external_id=ext, home_team=home, away_team=away,
        game_date=date(2026, 9, 13), scheduled_game_date=date(2026, 9, 13), sport="NFL",
        bet_type=bet_type, pick_side=side, line=line,
        model_prob=0.5 + edge, implied_prob=0.5, edge=edge, odds=odds,
        extra={"lean": True, "bookmaker": "draftkings"},
    )


def _slate(events, key="2026-09-13-1-early", label="Sunday Early"):
    return Slate(key=key, label=label, date=date(2026, 9, 13), events=events)


def _event(ext, home, away, commence="2026-09-13T17:00:00Z"):
    return {"id": ext, "home_team": home, "away_team": away, "commence_time": commence}


# ---- coherence ------------------------------------------------------------

class TestCoherence:
    def test_same_player_twice_is_refused(self):
        a = _prop("Jaxon Smith-Njigba", "under", "SEA", market="player_receptions", line=5.5)
        b = _prop("Jaxon Smith-Njigba", "over", "SEA")
        assert pm.conflicts(a, b) == "same player"
        c = _prop("Jaxon Smith-Njigba", "under", "SEA")  # same direction is still one bet
        assert pm.conflicts(a, c) == "same player"

    def test_backing_both_teams_is_refused(self):
        ml = _game("moneyline", SEA)
        spread = _game("spread", f"{NE} +3.5", line=3.5)
        assert pm.conflicts(ml, spread) == "backs both teams"
        assert pm.conflicts(_game("total", "over 47.5", line=47.5),
                            _game("total", "under 47.5", line=47.5)) is not None

    def test_total_under_with_a_receiver_over_is_refused(self):
        under = _game("total", "under 44.5", line=44.5)
        over = _prop("Hunter Henry", "over", "NE")
        assert "opposite stories" in pm.conflicts(under, over)
        # ...but an under on a receiver reads as the same grind.
        assert pm.conflicts(under, _prop("Hunter Henry", "under", "NE")) is None

    def test_team_story_is_accepted(self):
        legs = [_game("moneyline", SEA), _prop("Sam Darnold", "over", "SEA",
                                                market="player_pass_yds", line=249.5)]
        assert pm.coherent(legs, _prop("Jaxon Smith-Njigba", "over", "SEA"))
        assert pm.coherent(legs, _prop("Hunter Henry", "under", "NE"))
        assert not pm.coherent(legs, _prop("Cooper Kupp", "under", "SEA"))

    def test_moneyline_and_cover_on_the_same_team_agree(self):
        assert pm.conflicts(_game("moneyline", SEA),
                            _game("spread", f"{SEA} -3.5", line=-3.5)) is None

    def test_anytime_td_is_an_over_on_its_team(self):
        td = _prop("Kenneth Walker", "yes", "SEA", market="player_anytime_td", line=0.5, odds=150)
        assert pm.leg_signs(td) == {"SEA": 1}
        assert pm.conflicts(td, _game("total", "under 40.5", line=40.5)) is not None


# ---- odds math --------------------------------------------------------------

class TestCombine:
    def test_price_and_probabilities_are_products(self):
        legs = [_prop("A", "over", "SEA", odds=-110), _prop("B", "over", "SEA", odds=-110),
                _prop("C", "under", "NE", odds=-110)]
        parent = pm.combine_legs(legs, "sgp", 1.0, 100.0)
        assert parent.odds == 596                      # 1.909^3 = 6.96 → +596
        assert parent.model_prob == pytest.approx(0.8 ** 3)
        assert parent.implied_prob == pytest.approx(0.6 ** 3)
        assert parent.market == pm.SGP_MARKET and parent.extra["same_game"]
        assert parent.strategy == pm.PARLAY_STRATEGY and parent.bet_type == "parlay"
        assert parent.recommended_bet == 1.0 and parent.extra["card"]
        assert parent.pick_side == "3-leg SGP"

    def test_cross_game_ticket_is_not_flagged_as_sgp(self):
        legs = [_prop("A", "over", "SEA"), _prop("B", "over", "LAR", home=LAR, away=SF, ext="e2")]
        parent = pm.combine_legs(legs, "window", 1.0, 100.0)
        assert parent.market == pm.PARLAY_MARKET and not parent.extra["same_game"]

    def test_american_round_trip(self):
        for odds in (-250, -110, 100, 150, 596):
            assert pm.decimal_to_american(pm.american_to_decimal(odds)) == odds


# ---- selection --------------------------------------------------------------

class TestPickLegs:
    def test_greedy_by_edge_stays_coherent(self):
        pool = [
            _prop("A", "under", "SEA", edge=0.40, market="player_receptions", line=4.5),
            _prop("A", "under", "SEA", edge=0.39, strategy="ladder"),  # same player: skipped
            _prop("B", "over", "SEA", edge=0.35, strategy="overs"),    # opposite story on SEA
            _prop("C", "under", "NE", edge=0.30, strategy="ladder"),
            _game("total", "under 44.5", edge=0.05, line=44.5),
        ]
        legs = pm.pick_legs(pool, 3)
        assert [leg.player or leg.pick_side for leg in legs] == ["A", "C", "under 44.5"]

    def test_at_most_one_carded_pick_per_ticket(self):
        pool = [
            _prop("A", "under", "SEA", edge=0.40, card=True),       # on the card
            _prop("B", "under", "SEA", edge=0.39, card=True),       # on the card: refused
            _prop("C", "under", "NE", edge=0.38, card=True),        # on the card: refused
            _prop("D", "under", "NE", edge=0.10, strategy="ladder", odds=200),
            _game("total", "under 44.5", edge=0.02, line=44.5),
        ]
        legs = pm.pick_legs(pool, 3)
        assert [leg.player or leg.pick_side for leg in legs] == ["A", "D", "under 44.5"]
        assert sum(pm.is_card_pick(leg) for leg in legs) == 1

    def test_extras_are_free_legs(self):
        """The off-card main-book props are the same model cut by the slate
        cap, not tracked bets — a ticket may take as many as it likes."""
        pool = [
            _prop("A", "under", "SEA", edge=0.40, card=True),
            _prop("B", "under", "SEA", edge=0.39),                  # extra
            _prop("C", "under", "NE", edge=0.38),                   # extra
            _prop("D", "under", "NE", edge=0.10, strategy="ladder", odds=200),
        ]
        legs = pm.pick_legs(pool, 3)
        assert [leg.player for leg in legs] == ["A", "B", "C"]

    def test_props_before_game_sides_whatever_the_edge(self):
        pool = [
            _game("moneyline", SEA, edge=0.30, odds=150),           # huge but a game side
            _prop("A", "under", "SEA", edge=0.05, strategy="overs"),
            _prop("B", "under", "NE", edge=0.04, strategy="ladder", odds=180),
            _prop("C", "under", "NE", edge=0.03, strategy="ladder", odds=180),
        ]
        legs = pm.pick_legs(pool, 3)
        assert all(leg.bet_type == "prop" for leg in legs)

    def test_td_scorer_only_when_the_model_is_sure(self):
        long_shot = _prop("K", "yes", "SEA", edge=0.12, market="player_anytime_td", line=0.5,
                          odds=300)
        long_shot.model_prob = 0.35
        fav = _prop("W", "yes", "NE", edge=0.10, market="player_anytime_td", line=0.5, odds=-120)
        fav.model_prob = 0.62
        others = [_prop("A", "under", "SEA", edge=0.2, strategy="ladder", odds=150),
                  _prop("B", "under", "SEA", edge=0.15, strategy="overs")]
        legs = pm.pick_legs(others + [long_shot, fav], 3)
        names = {leg.player for leg in legs}
        assert "W" in names and "K" not in names

    def test_short_ticket_swaps_in_a_plus_money_leg(self):
        pool = [
            _prop("A", "under", "SEA", edge=0.40, odds=-300),
            _prop("B", "under", "SEA", edge=0.35, odds=-300, strategy="overs"),
            _prop("C", "under", "NE", edge=0.30, odds=-300, strategy="ladder"),  # +137: short
            _prop("D", "under", "NE", edge=0.10, odds=250, strategy="ladder"),
        ]
        legs = pm.pick_legs(pool, 3, min_odds=300)
        names = {leg.player for leg in legs}
        assert "D" in names and len(legs) == 3
        assert pm.combined_odds(legs) >= 300

    def test_no_ticket_when_the_pool_cannot_fill_it(self):
        assert pm.pick_legs([_prop("A", "under", "SEA"), _prop("B", "over", "SEA")], 3) == []
        assert pm.pick_legs([_prop("A", "under", "SEA", edge=-0.01)] * 3, 3) == []
        # three carded unders alone can never make a ticket
        assert pm.pick_legs([_prop("A", "under", "SEA", card=True),
                             _prop("B", "under", "SEA", card=True),
                             _prop("C", "under", "NE", card=True)], 3) == []

    def test_cross_game_takes_one_leg_per_game(self):
        pool = [_prop("A", "under", "SEA", edge=0.4), _prop("B", "under", "NE", edge=0.39),
                _prop("C", "under", "LAR", edge=0.3, home=LAR, away=SF, ext="e2",
                      strategy="ladder"),
                _prop("D", "under", "SF", edge=0.2, home=LAR, away=SF, ext="e2",
                      strategy="ladder")]
        legs = pm.pick_legs(pool, 2, one_per_game=True)
        assert [leg.player for leg in legs] == ["A", "C"]


class TestBuildParlays:
    def test_single_game_slate_gets_an_sgp_from_every_section(self):
        events = [_event("e1", SEA, NE, "2026-09-11T00:15:00Z")]
        slate = _slate(events, key="2026-09-10-3-night", label="Thursday Night")
        props = [_prop("A", "under", "SEA", edge=0.4)]
        td = [_prop("B", "yes", "NE", edge=0.12, market="player_anytime_td", line=0.5, odds=-110)]
        sides = [_game("moneyline", NE, edge=0.02, odds=140), _game("moneyline", SEA, edge=-0.02)]
        out = pm.build_parlays([slate], [props, td, [], []], sides, bankroll=100.0)
        assert len(out) == 1 and out[0].kind == "sgp" and out[0].same_game
        assert {leg.player or leg.pick_side for leg in out[0].legs} == {"A", "B", NE}
        assert all(leg.strategy == pm.PARLAY_LEG_STRATEGY and leg.recommended_bet == 0
                   for leg in out[0].legs)
        assert out[0].parent.extra["legs"] and out[0].slate_key == slate.key
        # the section's own candidate is untouched — it keeps its own book
        assert props[0].strategy is None and props[0].recommended_bet == 1.5

    def test_sunday_builds_window_parlays_and_one_lean_parlay(self):
        early = [_event("e1", SEA, NE), _event("e2", LAR, SF), _event("e3", "Denver Broncos",
                                                                     "Las Vegas Raiders")]
        s_early = _slate(early)
        props = [_prop("A", "under", "SEA", edge=0.4), _prop("B", "under", "NE", edge=0.39),
                 _prop("C", "under", "LAR", edge=0.3, home=LAR, away=SF, ext="e2",
                       strategy="ladder"),
                 _prop("D", "over", "DEN", edge=0.2, home="Denver Broncos",
                       away="Las Vegas Raiders", ext="e3", strategy="overs")]
        sides = [_game("moneyline", SEA, edge=0.02, ext="e1"),
                 _game("moneyline", NE, edge=-0.02, ext="e1"),
                 _game("spread", f"{SF} +3.5", line=3.5, edge=0.015, home=LAR, away=SF, ext="e2"),
                 _game("total", "under 44.5", line=44.5, edge=0.01, home="Denver Broncos",
                       away="Las Vegas Raiders", ext="e3")]
        out = pm.build_parlays([s_early], [props, [], [], []], sides, bankroll=100.0)
        kinds = {p.kind: p for p in out}
        assert set(kinds) == {"window", "leans"}
        window = kinds["window"]
        assert len({pm._candidate_game_key(leg) for leg in window.legs}) == 3
        assert not window.same_game and window.parent.market == pm.PARLAY_MARKET
        leans = kinds["leans"]
        assert len(leans.legs) == 3 and leans.parent.pick_side == "3-leg lean parlay"
        assert all(leg.bet_type in pm.GAME_BET_TYPES for leg in leans.legs)

    def test_lean_parlay_needs_three_positive_games(self):
        early = [_event("e1", SEA, NE), _event("e2", LAR, SF)]
        sides = [_game("moneyline", SEA, edge=0.02, ext="e1"),
                 _game("spread", f"{SF} +3.5", line=3.5, edge=0.015, home=LAR, away=SF, ext="e2")]
        out = pm.build_parlays([_slate(early)], [[], [], [], []], sides, bankroll=100.0)
        assert out == []

    def test_lean_parlay_takes_up_to_five(self):
        names = [("e%d" % i, f"Home{i}", f"Away{i}") for i in range(1, 8)]
        events = [_event(e, h, a) for e, h, a in names]
        sides = [_game("moneyline", h, edge=0.01 + i / 100, home=h, away=a, ext=e)
                 for i, (e, h, a) in enumerate(names)]
        out = pm.build_parlays([_slate(events)], [[], [], [], []], sides, bankroll=100.0)
        leans = next(p for p in out if p.kind == "leans")
        assert len(leans.legs) == 5


# ---- settlement -------------------------------------------------------------

class TestSettle:
    def test_any_loss_loses_the_ticket(self):
        assert settle_mod.settle(1.0, [("win", -110), ("loss", -110), ("win", 150)]) == ("loss", -1.0)

    def test_all_wins_pay_the_product(self):
        result, pnl = settle_mod.settle(1.0, [("win", -110), ("win", -110), ("win", -110)])
        assert result == "win" and pnl == pytest.approx(5.96, abs=0.01)

    def test_push_and_void_legs_drop_out(self):
        result, pnl = settle_mod.settle(2.0, [("win", 100), ("push", -110), ("void", -110)])
        assert result == "win" and pnl == pytest.approx(2.0)
        assert settle_mod.settle(1.0, [("void", -110), ("push", -110)]) == ("void", 0.0)

    def test_open_while_a_leg_is_ungraded(self):
        assert settle_mod.settle(1.0, [("win", -110), (None, -110)]) is None


# ---- persistence round-trip (SQLite) -----------------------------------------

@pytest.fixture
def db(monkeypatch):
    from betting_agent.db.models import Base

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def fake_session():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(picks_mod, "get_session", fake_session)
    monkeypatch.setattr(settle_mod, "get_session", fake_session)
    return factory


class TestPersistence:
    def _ticket(self):
        legs = [_prop("A", "under", "SEA", edge=0.4), _prop("B", "under", "NE", edge=0.3),
                _game("moneyline", SEA, edge=0.02, odds=-150)]
        copies = [pm._leg_copy(leg) for leg in legs]
        return pm.Parlay("sgp", pm.combine_legs(copies, "sgp", 1.0, 100.0), copies, "k")

    def test_parent_then_legs_then_settle(self, db):
        from betting_agent.db.models import Pick

        picks_mod.save_parlays_to_db([self._ticket()], {"e1"})
        with db() as s:
            parent = s.query(Pick).filter(Pick.bet_type == "parlay").one()
            legs = s.query(Pick).filter(Pick.parlay_id == parent.id).all()
            assert parent.strategy == "parlay" and parent.on_card and parent.recommended_bet == 1.0
            assert len(legs) == 3 and all(leg.strategy == "parlay_leg" for leg in legs)
            assert all(not leg.on_card and leg.recommended_bet == 0 for leg in legs)
            assert legs[0].game_id == parent.game_id
            # nothing settles while a leg is open
            legs[0].result = "win"
            legs[0].pnl = 0.0
            s.commit()
        assert settle_mod.settle_parlays() == 0
        with db() as s:
            for leg in s.query(Pick).filter(Pick.parlay_id.isnot(None)).all():
                leg.result = "win"
            s.commit()
        assert settle_mod.settle_parlays() == 1
        with db() as s:
            parent = s.query(Pick).filter(Pick.bet_type == "parlay").one()
            assert parent.result == "win" and parent.graded_at is not None
            assert parent.pnl == pytest.approx(1.0 * (1.909 ** 2 * (1 + 100 / 150) - 1), abs=0.02)
        out = settle_mod.graded_parlays("NFL")
        assert len(out) == 1 and len(out[0]["legs"]) == 3 and out[0]["result"] == "win"

    def test_rerun_refreshes_the_ticket_and_replaces_its_legs(self, db):
        from betting_agent.db.models import Pick

        picks_mod.save_parlays_to_db([self._ticket()], {"e1"})
        ticket = self._ticket()
        ticket.legs = ticket.legs[:2]
        ticket.parent.odds = 400
        picks_mod.save_parlays_to_db([ticket], {"e1"})
        with db() as s:
            parents = s.query(Pick).filter(Pick.bet_type == "parlay").all()
            assert len(parents) == 1 and parents[0].odds == 400
            assert s.query(Pick).filter(Pick.parlay_id == parents[0].id).count() == 2

    def test_rerun_without_a_ticket_retires_the_old_one(self, db):
        from betting_agent.db.models import Pick

        picks_mod.save_parlays_to_db([self._ticket()], {"e1"})
        picks_mod.save_parlays_to_db([], {"e1"})
        with db() as s:
            parent = s.query(Pick).filter(Pick.bet_type == "parlay").one()
            assert parent.on_card is False

    def test_settled_ticket_is_never_rewritten(self, db):
        from betting_agent.db.models import Pick

        picks_mod.save_parlays_to_db([self._ticket()], {"e1"})
        with db() as s:
            parent = s.query(Pick).filter(Pick.bet_type == "parlay").one()
            parent.result, parent.pnl = "loss", -1.0
            s.commit()
        ticket = self._ticket()
        ticket.parent.odds = 999
        picks_mod.save_parlays_to_db([ticket], {"e1"})
        with db() as s:
            parent = s.query(Pick).filter(Pick.bet_type == "parlay").one()
            assert parent.odds != 999 and parent.result == "loss"


# ---- the book stays out of everything else ------------------------------------

class TestIsolation:
    def test_side_books_and_bankroll(self, monkeypatch):
        from betting_agent.accounting import ledger as ledger_mod

        assert "parlay" in ledger_mod.SIDE_BOOKS and "parlay_leg" in ledger_mod.SIDE_BOOKS
        monkeypatch.setattr(ledger_mod.settings, "parlay_bankroll", 25.0)
        assert ledger_mod.starting_bankroll_for("parlay") == 25.0

    def test_main_results_post_never_lists_a_parlay(self):
        rows = [
            {"result": "win", "pnl": 1.0, "away_team": NE, "home_team": SEA, "odds": -110,
             "bet_type": "prop", "player": "A", "market": "player_receptions", "pick_side": "under",
             "line": 4.5},
            {"result": "win", "pnl": 5.96, "away_team": NE, "home_team": SEA, "odds": 596,
             "bet_type": "parlay", "pick_side": "3-leg SGP", "strategy": "parlay"},
            {"result": "win", "pnl": 0.0, "away_team": NE, "home_team": SEA, "odds": -110,
             "bet_type": "prop", "player": "B", "market": "player_receptions", "pick_side": "under",
             "line": 4.5, "strategy": "parlay_leg"},
        ]
        body = "\n".join(d._result_lines(rows, leans_labelled=True))
        assert "A receptions under" in body
        assert "SGP" not in body and " B " not in body

    def test_closing_capture_and_grader_skip_parlays(self):
        clv = Path("src/betting_agent/accounting/prop_clv.py").read_text()
        assert 'p.bet_type != "parlay" and p.strategy != "parlay_leg"' in clv
        grader = Path("src/betting_agent/accounting/grader.py").read_text()
        assert 'pick.bet_type == "parlay"' in grader
        props = Path("scripts/props.py").read_text()
        assert "save_parlays_to_db(parlays" in props and "send_parlays_to_discord" in props
        # built after the cards are selected, so extra["card"] is set on the legs' sources
        assert props.index("card_props = select_card(candidates") < props.index("build_parlays(")


# ---- Discord --------------------------------------------------------------------

@pytest.fixture
def sent(monkeypatch):
    posts: list[tuple[str, dict]] = []
    monkeypatch.setattr(d, "_send_webhook", lambda url, payload: posts.append((url, payload)) or True)
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_PARLAYS", "https://discord.test/parlays")
    monkeypatch.setenv("DISCORD_WEBHOOK_NFL_PARLAYS_RESULTS", "https://discord.test/parlays-results")
    return posts


class TestDiscord:
    def _ticket(self, kind="sgp"):
        legs = [pm._leg_copy(_prop("A", "under", "SEA", edge=0.4)),
                pm._leg_copy(_prop("B", "under", "NE", edge=0.3)),
                pm._leg_copy(_game("moneyline", SEA, edge=0.02, odds=-150))]
        return pm.Parlay(kind, pm.combine_legs(legs, kind, 1.0, 100.0), legs, "k")

    def test_card_post_lists_legs_and_flags_the_sgp_price(self, sent):
        assert d.send_parlays_to_discord("2026-09-13", [self._ticket()], 100.0)
        (url, payload), = sent
        assert url.endswith("/parlays")
        titles = [e.get("title", "") for e in payload["embeds"]]
        assert titles[0].startswith("🏈 Parlays") and "3-LEG SGP" in titles[1]
        body = payload["embeds"][1]["description"]
        assert "A reception yds UNDER 49.5" in body and f"{SEA} Moneyline" in body
        assert "real SGP price will be lower" in body
        assert all(e["color"] == d.COLOR_PURPLE for e in payload["embeds"])

    def test_silent_without_a_webhook(self, monkeypatch):
        monkeypatch.delenv("DISCORD_WEBHOOK_NFL_PARLAYS", raising=False)
        monkeypatch.delenv("DISCORD_WEBHOOK_NFL_PARLAYS_RESULTS", raising=False)
        assert d.send_parlays_to_discord("x", [self._ticket()], 100.0) is False
        assert d.send_parlay_results_to_discord({"total_bets": 1}, "NFL") is False

    def test_results_post_marks_each_leg(self, sent):
        summary = {"total_bets": 1, "wins": 0, "losses": 1, "pushes": 0, "total_pnl": -1.0,
                   "roi_pct": -100.0}
        parlays = [{"label": "3-leg SGP", "odds": 596, "result": "loss", "pnl": -1.0,
                    "model_prob": 0.18, "legs": [
                        {"bet_type": "prop", "pick_side": "under", "player": "A", "line": 49.5,
                         "market": "player_reception_yds", "odds": -110, "result": "win",
                         "home_team": SEA, "away_team": NE},
                        {"bet_type": "moneyline", "pick_side": SEA, "odds": -150,
                         "result": "loss", "home_team": SEA, "away_team": NE}]}]
        assert d.send_parlay_results_to_discord(
            summary, "NFL", date(2026, 9, 14), parlays=parlays,
            alltime_summary={"total_bets": 4, "wins": 0, "losses": 4, "pushes": 0,
                             "win_rate_pct": 0.0, "roi_pct": -100.0, "total_pnl": -4.0},
            starting_bankroll=100.0)
        (url, payload), = sent
        assert url.endswith("/parlays-results")
        body = payload["embeds"][0]["description"]
        assert "Hit:** 0/1 vs claimed `18.0%`" in body
        assert "✅ A reception yds under 49.5" in body and f"❌ {SEA} ML" in body
        assert "$100.00 → $96.00" in payload["embeds"][1]["description"]
