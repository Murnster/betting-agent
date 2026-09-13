"""Slate windows, per-slate caps, game leans vs a sharp reference, and the
closing capture for game picks."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from betting_agent.accounting.prop_clv import capture_game_closing_lines
from betting_agent.intelligence.game_lean import (
    MARGIN_SIGMA,
    fair_from_reference,
    game_leans,
)
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.slate import (
    PRIMETIME_PROP_CAP,
    WINDOW_PROP_CAP,
    cap_per_slate,
    group_events_by_slate,
    is_night_slate,
    select_card,
    slate_for,
)


def _ev(eid, ct, home="Kansas City Chiefs", away="Buffalo Bills"):
    return {"id": eid, "commence_time": ct, "home_team": home, "away_team": away}


def _candidate(external_id, edge):
    return BetCandidate(game_id=0, external_id=external_id, home_team="", away_team="",
                        game_date=date(2026, 10, 11), sport="NFL", bet_type="prop",
                        pick_side="under", model_prob=.6, implied_prob=.5, edge=edge,
                        odds=-110)


class TestSlates:
    @pytest.mark.parametrize("ct,label", [
        ("2026-09-10T00:20:00Z", "Wednesday"),        # Wed 20:20 ET opener
        ("2026-09-11T00:15:00Z", "Thursday Night"),
        ("2026-09-13T13:30:00Z", "International"),    # 9:30 ET London game
        ("2026-09-13T16:00:00Z", "Sunday Early"),     # 12:00 ET — still the early window
        ("2026-09-13T17:00:00Z", "Sunday Early"),
        ("2026-09-13T20:25:00Z", "Sunday Late"),
        ("2026-09-14T00:20:00Z", "Sunday Night"),     # 20:20 ET Sunday
        ("2026-09-15T00:15:00Z", "Monday Night"),
    ])
    def test_window_labels_follow_eastern_kickoff(self, ct, label):
        assert slate_for(ct)[1] == label

    def test_grouping_orders_by_kickoff_and_sets_caps(self):
        events = [_ev("snf", "2026-09-14T00:20:00Z"), _ev("e1", "2026-09-13T17:00:00Z"),
                  _ev("e2", "2026-09-13T17:00:00Z", "Dallas Cowboys", "New York Giants"),
                  _ev("late", "2026-09-13T20:25:00Z")]
        slates = group_events_by_slate(events)
        assert [s.label for s in slates] == ["Sunday Early", "Sunday Late", "Sunday Night"]
        early, late, snf = slates
        assert early.prop_cap == WINDOW_PROP_CAP and early.lean_cap == 3
        assert snf.single_game and snf.prop_cap == PRIMETIME_PROP_CAP and snf.lean_cap == 1
        assert "Buffalo Bills @ Kansas City Chiefs" in snf.title()

    def test_select_card_takes_top_edges_in_slate_only(self):
        slate = group_events_by_slate([_ev("e1", "2026-09-14T00:20:00Z")])[0]
        def cand(eid, edge):
            return BetCandidate(game_id=0, external_id=eid, home_team="", away_team="",
                                game_date=date(2026, 9, 13), sport="NFL", bet_type="prop",
                                pick_side="under", model_prob=.6, implied_prob=.5, edge=edge,
                                odds=-110)
        cands = [cand("e1", .10), cand("e1", .30), cand("e1", .20), cand("other", .99)]
        card = select_card(cands, slate, 2)
        assert [c.edge for c in card] == [.30, .20]
        assert all(c.extra.get("card") for c in card)
        assert not cands[0].extra.get("card") and not cands[3].extra.get("card")


def _book(key, ml=(-150, 130), spread=(-3.0, -110, -110), total=(45.5, -110, -110)):
    return {"key": key, "title": key, "markets": [
        {"key": "h2h", "outcomes": [{"name": "Kansas City Chiefs", "price": ml[0]},
                                    {"name": "Buffalo Bills", "price": ml[1]}]},
        {"key": "spreads", "outcomes": [
            {"name": "Kansas City Chiefs", "point": spread[0], "price": spread[1]},
            {"name": "Buffalo Bills", "point": -spread[0], "price": spread[2]}]},
        {"key": "totals", "outcomes": [{"name": "Over", "point": total[0], "price": total[1]},
                                       {"name": "Under", "point": total[0], "price": total[2]}]},
    ]}


class TestGameLeans:
    def test_fair_parameters_from_reference(self):
        fair = fair_from_reference({"ml_home": -150.0, "ml_away": 130.0,
                                    "spread_home_line": -3.0, "spread_home_price": -110.0,
                                    "spread_away_price": -110.0, "total_over_line": 45.5,
                                    "over_price": -110.0, "under_price": -110.0})
        assert fair["home_prob"] == pytest.approx(0.578, abs=0.01)
        assert fair["mu_margin"] == pytest.approx(3.0)      # even money at -3 → expect +3
        assert fair["mu_total"] == pytest.approx(45.5)

    def test_off_market_book_line_is_the_lean(self):
        # Pinnacle: KC -3 even. DraftKings hangs KC -4.5 → Bills +4.5 is the value.
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("pinnacle"), _book("draftkings", spread=(-4.5, -110, -110))]}
        leans = game_leans([event], ["bet365", "draftkings"], bankroll=1000.0)
        assert len(leans) == 1
        lean = leans[0]
        assert lean.bet_type == "spread" and lean.pick_side == "Buffalo Bills +4.5"
        assert lean.line == 4.5 and lean.extra["bookmaker"] == "draftkings"
        # P(margin < 4.5 | mu 3, sigma) ≈ 0.547 vs 0.5 fair-vig → ~+4.7%
        expected = 0.5 + (4.5 - 3.0) / MARGIN_SIGMA * 0.3989 - 0.5  # linearised, loose
        assert 0.03 < lean.edge < 0.07 and abs(lean.edge - expected) < 0.02
        assert lean.recommended_bet > 0 and lean.extra["lean"] is True

    def test_book_matching_pinnacle_has_only_vig_negative_edge(self):
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("pinnacle"), _book("draftkings")]}
        lean = game_leans([event], ["draftkings"], bankroll=1000.0)[0]
        assert lean.edge == pytest.approx(0.0, abs=1e-6)
        assert lean.recommended_bet == 0.0        # a zero-edge lean is paper with no stake

    def test_no_reference_quote_means_no_lean(self):
        event = {**_ev("e1", "2026-09-14T00:20:00Z"), "bookmakers": [_book("draftkings")]}
        assert game_leans([event], ["draftkings"], bankroll=1000.0) == []

    def test_reference_is_never_the_bettable_book(self):
        event = {**_ev("e1", "2026-09-14T00:20:00Z"), "bookmakers": [_book("pinnacle")]}
        assert game_leans([event], ["pinnacle"], bankroll=1000.0) == []


class TestGameClosingCapture:
    def _pick(self, bet_type, side, line, odds=-110):
        return SimpleNamespace(bet_type=bet_type, pick_side=side, line=line, odds=odds,
                               sport="NFL", closing_odds=None, closing_line=None, clv=None,
                               game=SimpleNamespace(external_id="e1"))

    def test_spread_held_gets_clv_moved_does_not(self):
        held = self._pick("spread", "Buffalo Bills +3.0", 3.0)
        moved = self._pick("spread", "Kansas City Chiefs -3.0", -3.0)
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("draftkings", spread=(-3.5, -105, -115))]}
        # DK closed KC -3.5: Bills side is +3.5 (moved for the Bills pick, line differs),
        # KC side moved from -3 to -3.5 as well.
        assert capture_game_closing_lines([event], [held, moved], ["bet365", "draftkings"]) == 2
        assert held.closing_line == 3.5 and held.closing_odds == -115 and held.clv is None
        assert moved.closing_line == -3.5 and moved.clv is None

    def test_moneyline_and_total(self):
        ml = self._pick("moneyline", "Kansas City Chiefs", None, odds=-140)
        tot = self._pick("total", "under 45.5", 45.5, odds=-110)
        event = {**_ev("e1", "2026-09-14T00:20:00Z"),
                 "bookmakers": [_book("draftkings", ml=(-160, 140), total=(45.5, -105, -115))]}
        assert capture_game_closing_lines([event], [ml, tot], ["draftkings"]) == 2
        assert ml.closing_odds == -160 and ml.clv is not None and ml.clv > 0   # took -140, closed -160
        assert tot.closing_line == 45.5 and tot.closing_odds == -115 and tot.clv > 0


class TestCapPerSlate:
    """
    The main-card candidate cut used to be global (`candidates[:max_picks]`),
    which could spend its whole allowance on one kickoff window and leave a
    later one with nothing to card at all.
    """

    @staticmethod
    def _slates():
        return group_events_by_slate([
            {"id": "early1", "commence_time": "2026-10-11T17:00:00Z"},   # 1pm ET
            {"id": "early2", "commence_time": "2026-10-11T17:00:00Z"},
            {"id": "late1", "commence_time": "2026-10-11T20:25:00Z"},    # 4:25pm ET
            {"id": "night1", "commence_time": "2026-10-12T00:20:00Z"},   # 8:20pm ET
        ])

    def test_late_and_night_survive_a_cut_the_early_window_would_have_eaten(self):
        # Ten fat edges in the early window, thin ones everywhere else: a global
        # top-10 by edge keeps nothing but early games.
        candidates = [_candidate(external_id="early1", edge=0.90 - i * 0.01) for i in range(10)]
        candidates += [_candidate(external_id="late1", edge=0.20)]
        candidates += [_candidate(external_id="night1", edge=0.19)]

        kept = cap_per_slate(candidates, self._slates(), cap=10)
        surviving = {c.external_id for c in kept}
        assert "late1" in surviving, "the 4pm window was starved by the early window"
        assert "night1" in surviving, "Sunday night was starved by the early window"

        # The old global cut is what this guards against.
        assert {c.external_id for c in sorted(
            candidates, key=lambda c: c.edge, reverse=True)[:10]} == {"early1"}

    def test_each_slate_keeps_its_own_best_cap(self):
        candidates = [_candidate(external_id="early1", edge=0.5 - i * 0.01) for i in range(5)]
        candidates += [_candidate(external_id="late1", edge=0.4 - i * 0.01) for i in range(5)]

        kept = cap_per_slate(candidates, self._slates(), cap=2)
        assert len(kept) == 4
        by_event: dict[str, list[float]] = {}
        for c in kept:
            by_event.setdefault(c.external_id, []).append(c.edge)
        assert sorted(by_event) == ["early1", "late1"]
        assert by_event["early1"] == pytest.approx([0.50, 0.49])
        assert by_event["late1"] == pytest.approx([0.40, 0.39])

    def test_input_order_is_preserved(self):
        candidates = [
            _candidate(external_id="early1", edge=0.10),
            _candidate(external_id="late1", edge=0.90),
            _candidate(external_id="early1", edge=0.50),
        ]
        kept = cap_per_slate(candidates, self._slates(), cap=5)
        assert [c.edge for c in kept] == [0.10, 0.90, 0.50]

    def test_a_candidate_on_no_slate_is_kept_not_silently_dropped(self):
        candidates = [_candidate(external_id="not-an-event-today", edge=0.3)]
        assert len(cap_per_slate(candidates, self._slates(), cap=1)) == 1


class TestInternationalSlate:
    """
    The Sunday 9:30 ET game gets its own card, and its own run.

    It kicks hours before the 12:45-local Sunday cron, which drops games that
    have already started — so on the 2026 schedule six games (Weeks 4, 5, 6,
    7, 9, 10) would have been priced not at all.
    """

    def test_it_is_a_slate_of_its_own_with_primetime_caps(self):
        from betting_agent.intelligence.slate import (
            PRIMETIME_LEAN_CAP,
            PRIMETIME_PROP_CAP,
            group_events_by_slate,
        )
        slates = group_events_by_slate([
            _ev("intl", "2026-10-04T13:30:00Z"),                     # 9:30 ET
            _ev("early", "2026-10-04T17:00:00Z", "Chicago Bears", "Carolina Panthers"),
        ])
        assert [s.label for s in slates] == ["International", "Sunday Early"]
        intl = slates[0]
        assert intl.single_game
        assert intl.prop_cap == PRIMETIME_PROP_CAP and intl.lean_cap == PRIMETIME_LEAN_CAP

    def test_thanksgiving_afternoon_games_are_not_thursday_night(self):
        """
        Same bug on a Thursday: 13:00 and 16:30 ET games that the 19:00-local
        run finds already kicked off, and that slate_for used to label
        "Thursday Night" because the Thursday branch ignored the hour.
        """
        from betting_agent.intelligence.slate import (
            PRIMETIME_PROP_CAP,
            group_events_by_slate,
            slate_for,
        )
        assert slate_for("2026-11-26T18:00:00Z")[1] == "Thursday Early"   # 13:00 ET
        assert slate_for("2026-11-26T21:30:00Z")[1] == "Thursday Late"    # 16:30 ET
        assert slate_for("2026-11-27T01:20:00Z")[1] == "Thursday Night"   # 20:20 ET
        # A normal Thursday still collapses to the one primetime slate.
        assert slate_for("2026-09-11T00:15:00Z")[1] == "Thursday Night"

        slates = group_events_by_slate([
            _ev("a", "2026-11-26T18:00:00Z"),
            _ev("b", "2026-11-26T21:30:00Z", "Dallas Cowboys", "New York Giants"),
            _ev("c", "2026-11-27T01:20:00Z", "Green Bay Packers", "Minnesota Vikings"),
        ])
        assert [s.label for s in slates] == ["Thursday Early", "Thursday Late", "Thursday Night"]
        assert all(s.single_game and s.prop_cap == PRIMETIME_PROP_CAP for s in slates)

    def test_the_regular_windows_are_unchanged(self):
        from betting_agent.intelligence.slate import group_events_by_slate
        slates = group_events_by_slate([
            _ev("e", "2026-10-04T17:00:00Z"),                        # 13:00 ET
            _ev("l", "2026-10-04T20:25:00Z", "Dallas Cowboys", "New York Giants"),
            _ev("n", "2026-10-05T00:20:00Z", "Green Bay Packers", "Minnesota Vikings"),
        ])
        assert [s.label for s in slates] == ["Sunday Early", "Sunday Late", "Sunday Night"]

    def test_the_early_run_prices_only_the_imminent_game(self):
        """--within-hours keeps the 07:45 ET run off the rest of Sunday."""
        import sys
        from datetime import datetime
        from zoneinfo import ZoneInfo

        sys.path.insert(0, "scripts")
        from props import _events_commencing_today

        et = ZoneInfo("America/New_York")
        events = [_ev("intl", "2026-10-04T13:30:00Z"),               # 9:30 ET
                  _ev("early", "2026-10-04T17:00:00Z"),              # 13:00 ET
                  _ev("snf", "2026-10-05T00:20:00Z")]                # 20:20 ET
        now = datetime(2026, 10, 4, 7, 45, tzinfo=et)                # the early cron

        windowed = _events_commencing_today(events, now=now, within_hours=4)
        assert [e["id"] for e in windowed] == ["intl"]
        # Unwindowed, the same run would price the whole day.
        assert len(_events_commencing_today(events, now=now)) == 3

    def test_the_midday_run_skips_the_game_that_already_kicked_off(self):
        import sys
        from datetime import datetime
        from zoneinfo import ZoneInfo

        sys.path.insert(0, "scripts")
        from props import _events_commencing_today

        et = ZoneInfo("America/New_York")
        events = [_ev("intl", "2026-10-04T13:30:00Z"), _ev("early", "2026-10-04T17:00:00Z")]
        now = datetime(2026, 10, 4, 11, 45, tzinfo=et)               # the 12:45-local cron
        assert [e["id"] for e in _events_commencing_today(events, now=now)] == ["early"]


class TestCronReachesEveryKickoff:
    """
    The card schedule and the kickoff times have to agree.

    Twice now a real game got no card because a fixed cron hour sat after an
    unusually early kickoff: the Sunday 9:30 ET international games (six in
    2026) and the Thanksgiving afternoon pair. This reads the schedule
    `nfl_loop.sh crontab` prints and checks it against the awkward kickoffs,
    so changing one without the other fails here rather than in November.
    """

    # (local weekday, local HH:MM, NFL_WITHIN_HOURS, NFL_SKIP_NIGHT) per `card`.
    @staticmethod
    def _card_runs() -> list[tuple[int, int, int, float | None, bool]]:
        import re
        import subprocess

        out = subprocess.run(["scripts/nfl_loop.sh", "crontab"],
                             capture_output=True, text=True, check=True).stdout
        runs = []
        for line in out.splitlines():
            if line.startswith("#") or " card" not in line:
                continue
            mm, hh, _, _, dow = line.split()[:5]
            win = re.search(r"NFL_WITHIN_HOURS=(\d+(?:\.\d+)?)", line)
            skip_night = "NFL_SKIP_NIGHT=1" in line
            for d in dow.split(","):
                # cron Sunday is 0; Python's weekday() has Monday 0, Sunday 6.
                runs.append(((int(d) - 1) % 7, int(hh), int(mm),
                             float(win.group(1)) if win else None, skip_night))
        assert runs, "no card entries found in the printed crontab"
        return runs

    #: Kickoffs that have caught this out, plus the ordinary ones. ET.
    KICKOFFS = [
        ("2026-10-04", "09:30", "Sunday international"),
        ("2026-10-04", "13:00", "Sunday early"),
        ("2026-10-04", "20:20", "Sunday night"),
        ("2026-11-26", "13:00", "Thanksgiving early"),
        ("2026-11-26", "16:30", "Thanksgiving late"),
        ("2026-11-26", "20:20", "Thanksgiving night"),
        ("2026-11-27", "15:00", "Black Friday"),
        ("2026-12-19", "17:00", "December Saturday"),
        ("2026-12-25", "13:00", "Christmas Friday"),
        ("2026-09-10", "20:35", "Wednesday opener"),
        ("2026-09-14", "20:15", "Monday night"),
        ("2026-09-17", "20:15", "Thursday night"),
    ]

    def test_every_awkward_kickoff_has_a_card_run_before_it(self):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        # The crontab is written for a box at ET+1 (America/Halifax).
        et, local = ZoneInfo("America/New_York"), ZoneInfo("America/Halifax")
        runs = self._card_runs()

        uncovered = []
        for day, hhmm, note in self.KICKOFFS:
            kickoff = datetime.combine(
                datetime.strptime(day, "%Y-%m-%d").date(),
                datetime.strptime(hhmm, "%H:%M").time(), tzinfo=et,
            ).astimezone(local)
            night = is_night_slate(kickoff)
            if not any(
                run.weekday() == wd and run < kickoff
                and (win is None or kickoff <= run + timedelta(hours=win))
                and not (skip_night and night)
                for wd, hh, mm, win, skip_night in runs
                for run in [kickoff.replace(hour=hh, minute=mm, second=0, microsecond=0)]
            ):
                uncovered.append(f"{note} ({day} {hhmm} ET)")
        assert not uncovered, "no card run fires before: " + ", ".join(uncovered)

    def test_sunday_night_is_carded_in_the_evening_not_at_midday(self):
        """
        Sunday Night belongs to the 19:00 run, like every other primetime game.

        The midday run used to price it eight hours before a 20:20 ET kickoff
        because one --today pass cards every window of the day at once. Now the
        midday run carries NFL_SKIP_NIGHT and a Sunday 19:00 run picks it up
        (user's call, Sep 13 2026).
        """
        runs = self._card_runs()
        sunday = [r for r in runs if r[0] == 6]  # Python weekday: Sunday is 6
        assert any(hh == 19 and not skip_night and win is None
                   for _, hh, _mm, win, skip_night in sunday), \
            "no unrestricted Sunday 19:00 card run to pick up Sunday Night"
        assert all(skip_night or win is not None
                   for _, hh, _mm, win, skip_night in sunday if hh < 19), \
            "a Sunday run before 19:00 still prices the night game"

    def test_the_early_runs_stay_off_the_days_they_are_not_for(self):
        """A windowed early run must exit free on an ordinary week."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        et, local = ZoneInfo("America/New_York"), ZoneInfo("America/Halifax")
        windowed = [r for r in self._card_runs() if r[3] is not None]
        assert windowed, "expected at least one windowed early run"

        for day, hhmm, note in [("2026-10-04", "13:00", "ordinary Sunday early"),
                                ("2026-09-17", "20:15", "ordinary Thursday night")]:
            kickoff = datetime.combine(
                datetime.strptime(day, "%Y-%m-%d").date(),
                datetime.strptime(hhmm, "%H:%M").time(), tzinfo=et,
            ).astimezone(local)
            for wd, hh, mm, win, _skip in windowed:
                if wd != kickoff.weekday():
                    continue
                run = kickoff.replace(hour=hh, minute=mm, second=0, microsecond=0)
                assert kickoff > run + timedelta(hours=win), (
                    f"the {hh}:{mm:02d} early run would re-price the {note} "
                    "that the midday run prices at fresher odds"
                )
