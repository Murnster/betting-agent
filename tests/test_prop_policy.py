"""
Tests for the prop pick-selection policy: book-proxy lines, per-player
deduplication, per-market edge floors, and NFL game finalization.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import scripts.props as props_script
from betting_agent.intelligence.picks import BetCandidate
from betting_agent.sports.nfl.props import (
    MIN_QUOTABLE_LINE,
    book_proxy_line,
    books_in_preference,
    prop_bookmaker_order,
)


class TestBookProxyLine:
    def test_receptions_quoted_to_half_point_above_median(self):
        # median 4 → 4.5
        assert book_proxy_line([3, 4, 4, 5, 6], "player_receptions") == 4.5

    def test_yards_snapped_to_half_point(self):
        assert book_proxy_line([40.0, 50.0, 60.0], "player_reception_yds") == 50.5

    def test_uses_only_last_eight_games(self):
        old_high = [20] * 5
        recent_low = [2, 2, 2, 2, 2, 2, 2, 2]
        assert book_proxy_line(old_high + recent_low, "player_receptions") == 2.5

    def test_below_quotable_threshold_is_none(self):
        assert book_proxy_line([0, 0, 1], "player_receptions") is None
        assert book_proxy_line([1.0, 2.0, 3.0], "player_reception_yds") is None

    def test_thresholds_are_the_documented_ones(self):
        assert MIN_QUOTABLE_LINE["player_receptions"] == 1.5
        assert MIN_QUOTABLE_LINE["player_reception_yds"] == 10.0

    def test_empty_history(self):
        assert book_proxy_line([], "player_receptions") is None


def _cand(player: str, market: str, edge: float, event: str = "g1") -> BetCandidate:
    return BetCandidate(
        game_id=0, external_id=event, home_team="KC", away_team="BUF",
        game_date=date(2026, 9, 13), sport="NFL", bet_type="prop",
        pick_side="under", player=player, market=market, line=4.5,
        model_prob=0.5 + edge, implied_prob=0.5, edge=edge, odds=-110,
        kelly_fraction=0.04, recommended_bet=40.0, bankroll_at_pick=1000.0,
    )


class TestDeduplicateByPlayer:
    def test_keeps_only_the_highest_edge_market_per_player(self):
        out = props_script._deduplicate_by_player([
            _cand("Travis Kelce", "player_receptions", 0.12),
            _cand("Travis Kelce", "player_reception_yds", 0.18),
            _cand("Rashee Rice", "player_receptions", 0.11),
        ])
        assert len(out) == 2
        kelce = [c for c in out if c.player == "Travis Kelce"]
        assert len(kelce) == 1
        assert kelce[0].market == "player_reception_yds"

    def test_matches_players_across_name_spellings(self):
        out = props_script._deduplicate_by_player([
            _cand("Amon-Ra St. Brown", "player_receptions", 0.20),
            _cand("Amon Ra St Brown", "player_reception_yds", 0.09),
        ])
        assert len(out) == 1
        assert out[0].edge == pytest.approx(0.20)

    def test_distinct_players_all_survive(self):
        out = props_script._deduplicate_by_player([
            _cand("A Receiver", "player_receptions", 0.11),
            _cand("B Receiver", "player_receptions", 0.12),
        ])
        assert len(out) == 2


class TestEdgeFloors:
    def test_yards_floor_is_stricter_than_receptions(self):
        # Receiving yards ran ~5pp worse per the walk-forward diagnostic.
        assert (props_script.PROP_EDGE_FLOORS["player_reception_yds"]
                > props_script.PROP_EDGE_FLOORS["player_receptions"])

    def test_floors_exceed_measured_overconfidence(self):
        # Residual overconfidence after recalibration is ~4.6-6.0pp; a floor
        # at or below that would systematically bet negative-EV picks.
        assert min(props_script.PROP_EDGE_FLOORS.values()) >= 0.06


class TestGeneratePropCandidates:
    """End-to-end selection: floors, per-player dedup, correlation scaling."""

    def _event(self):
        def outcome(name, player, point, price):
            return {"name": name, "description": player, "point": point, "price": price}

        return {
            "id": "evt-1", "home_team": "Kansas City Chiefs",
            "away_team": "Buffalo Bills", "commence_time": "2026-09-13T17:00:00Z",
            "bookmakers": [{"key": "bet365", "markets": [
                {"key": "player_receptions", "outcomes": [
                    outcome("Over", "Star Guy", 4.5, -110),
                    outcome("Under", "Star Guy", 4.5, -110),
                    outcome("Over", "Other Guy", 3.5, -110),
                    outcome("Under", "Other Guy", 3.5, -110),
                ]},
                {"key": "player_reception_yds", "outcomes": [
                    outcome("Over", "Star Guy", 55.5, -110),
                    outcome("Under", "Star Guy", 55.5, -110),
                ]},
            ]}],
        }

    class _Model:
        """Stub returning a fixed, deliberately lopsided projection."""

        def __init__(self, p_under):
            self._p = p_under
            self.history = pd.DataFrame({
                "player_key": ["star guy", "other guy"], "team": ["KC", "KC"],
            })

        def project(self, player_key, season, week, opponent=None):
            model = self
            self.calls = getattr(self, "calls", [])
            self.calls.append((player_key, season, week, opponent))

            class _P:
                mean = 3.0
                games = 8

                def prob_over(self, line):
                    return 1.0 - model._p

                def prob_under(self, line):
                    return model._p

            return _P()

    def _generate(self, models, min_edge=None, **kwargs):
        return props_script.generate_prop_candidates(
            [self._event()], models, bankroll=1000.0,
            min_edge=min_edge, season=2026, **kwargs,
        )


    def test_preferred_book_absent_prices_against_first_fallback_only(self):
        # bet365 is not in the Odds API prop feed. When it posts nothing the
        # game is priced against the first fallback that did — never
        # best-of-N across every book in the response.
        model = self._Model(0.72)
        event = self._event()
        dk = {"key": "draftkings", "markets": event["bookmakers"][0]["markets"]}
        fd_outcomes = [dict(o, price=+150) for o in dk["markets"][0]["outcomes"]]
        fd = {"key": "fanduel", "markets": [{"key": "player_receptions", "outcomes": fd_outcomes}]}
        event["bookmakers"] = [fd, dk]   # response order must not matter
        cands = props_script.generate_prop_candidates(
            [event], {"player_receptions": model}, bankroll=1000.0, min_edge=None,
            season=2026, book_order=["bet365", "draftkings", "fanduel"],
        )
        assert cands and {c.extra["bookmaker"] for c in cands} == {"draftkings"}
        assert all(c.odds == -110 for c in cands)   # FanDuel's +150 never considered

    def test_preferred_book_present_is_used_even_when_fallback_is_juicier(self):
        model = self._Model(0.72)
        event = self._event()
        b365 = event["bookmakers"][0]
        dk_outcomes = [dict(o, price=+150) for o in b365["markets"][0]["outcomes"]]
        event["bookmakers"] = [b365, {"key": "draftkings", "markets": [
            {"key": "player_receptions", "outcomes": dk_outcomes}]}]
        cands = props_script.generate_prop_candidates(
            [event], {"player_receptions": model}, bankroll=1000.0, min_edge=None,
            season=2026, book_order=["bet365", "draftkings"],
        )
        assert cands and {c.extra["bookmaker"] for c in cands} == {"bet365"}

    def test_roster_overlay_resolves_opponent_for_a_mover(self):
        # History says Star Guy plays for KC; the 2026 roster says he moved
        # to BUF. Opponent (defense factor) must follow the roster.
        model = self._Model(0.72)
        self._generate({"player_receptions": model},
                       current_teams={"star guy": "BUF"})
        opp = {pk: o for pk, _, _, o in model.calls}
        assert opp["star guy"] == "KC"
        assert opp["other guy"] == "BUF"   # unchanged: history team KC

    def test_unknown_team_leaves_opponent_unset(self):
        model = self._Model(0.72)
        self._generate({"player_receptions": model}, current_teams={"star guy": "PHI"})
        opp = {pk: o for pk, _, _, o in model.calls}
        assert opp["star guy"] is None

    def test_exact_week_comes_from_the_schedule(self):
        model = self._Model(0.72)
        sched = pd.DataFrame({
            "home_team": ["KC"], "away_team": ["BUF"], "season": [2026],
            "game_date": [pd.Timestamp("2026-09-13")], "week": [1],
        })
        self._generate({"player_receptions": model}, schedule=sched)
        assert {w for _, _, w, _ in model.calls} == {1}

    def test_heuristic_week_without_schedule(self):
        model = self._Model(0.72)
        self._generate({"player_receptions": model})
        assert {w for _, _, w, _ in model.calls} == {2}   # Sep 13 → date heuristic

    def test_out_player_is_dropped_and_questionable_flagged(self):
        models = {"player_receptions": self._Model(0.72)}
        injuries = pd.DataFrame({
            "full_name": ["Star Guy", "Other Guy"], "team": ["KC", "KC"],
            "position": ["WR", "TE"], "report_status": ["Out", "Questionable"],
            "practice_status": [None, None], "gsis_id": ["1", "2"],
        })
        out = self._generate(models, injuries=injuries)
        assert [c.player for c in out] == ["Other Guy"]
        assert out[0].extra["flags"][0]["type"] == "player_injury"
        assert out[0].extra["flags"][0]["severity"] == "medium"
        # Single survivor → no same-game correlation scaling.
        assert out[0].recommended_bet == pytest.approx(
            props_script.recommended_bet(0.72, -110, 0.72 - 0.5, 1000.0)[1]
        )

    def test_qb1_out_flags_every_prop_on_that_team(self):
        models = {"player_receptions": self._Model(0.72)}
        injuries = pd.DataFrame({
            "full_name": ["Franchise QB"], "team": ["KC"], "position": ["QB"],
            "report_status": ["Out"], "practice_status": [None], "gsis_id": ["qb1"],
        })
        out = self._generate(models, injuries=injuries, qb1={"KC": ("Franchise QB", "qb1")})
        assert len(out) == 2
        for c in out:
            assert any(f["type"] == "qb_out" for f in c.extra["flags"])

    def test_one_pick_per_player_and_correlation_scaled(self):
        # 0.72 under vs 0.50 fair = 22% edge, clears both floors.
        models = {"player_receptions": self._Model(0.72),
                  "player_reception_yds": self._Model(0.72)}
        out = self._generate(models)
        assert len(out) == 2, "Star Guy must not be bet in both markets"
        assert {c.player for c in out} == {"Star Guy", "Other Guy"}
        # Two picks in one game → Kelly scaled by 1/sqrt(2).
        for c in out:
            assert c.recommended_bet < c.bankroll_at_pick * 0.07
            assert c.kelly_fraction > 0

    def test_yards_floor_rejects_what_receptions_floor_accepts(self):
        # 12% edge: above the receptions floor (10%), below yards (15%).
        models = {"player_receptions": self._Model(0.62),
                  "player_reception_yds": self._Model(0.62)}
        out = self._generate(models)
        assert out, "receptions picks should survive"
        assert all(c.market == "player_receptions" for c in out)

    def test_below_all_floors_yields_nothing(self):
        models = {"player_receptions": self._Model(0.56),
                  "player_reception_yds": self._Model(0.56)}
        assert self._generate(models) == []

    def test_explicit_min_edge_overrides_market_floors(self):
        models = {"player_receptions": self._Model(0.56),
                  "player_reception_yds": self._Model(0.56)}
        assert self._generate(models, min_edge=0.05)


class TestRankEventsByModelHeat:
    """The free pre-screen that decides which games are worth paid odds calls."""

    class _Model:
        """Stub with a per-player fixed P(under)."""

        stat_col = "receptions"

        def __init__(self, p_under_by_player):
            self._p = p_under_by_player

        def project(self, player_key, season, week, opponent=None):
            p = self._p.get(player_key)
            if p is None:
                return None

            class _P:
                def prob_over(self, line):
                    return 1.0 - p

                def prob_under(self, line):
                    return p

            return _P()

    def _history(self):
        # Two players, five recent games each; t=202601 keeps them "active"
        # for a week-2 event (asof 202602, cutoff 202599).
        rows = []
        for pk, team, rec in (("hot guy", "KC", 4), ("meh guy", "MIA", 4)):
            for t in range(202501, 202506):
                rows.append({"player_key": pk, "team": team, "t": t, "receptions": rec})
            rows.append({"player_key": pk, "team": team, "t": 202601, "receptions": rec})
        return pd.DataFrame(rows)

    def _event(self, home, away):
        return {"home_team": home, "away_team": away,
                "commence_time": "2026-09-13T17:00:00Z"}

    def test_game_with_an_edge_ranks_above_game_without(self):
        models = {"player_receptions": self._Model(
            # 0.75 → 25% edge (clears the 10% floor); 0.52 → 2% (doesn't).
            {"hot guy": 0.75, "meh guy": 0.52}
        )}
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Miami Dolphins", "New York Jets"),
             self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, self._history(), season=2026,
        )
        (top, n_top, heat_top), (bottom, n_bottom, heat_bottom) = ranked
        assert top["home_team"] == "Kansas City Chiefs"
        assert n_top == 1 and heat_top == pytest.approx(0.25)
        assert n_bottom == 0 and heat_bottom == 0.0

    def test_player_counted_once_across_markets(self):
        class _YdsModel(self._Model):
            stat_col = "receiving_yards"

        hist = self._history()
        hist["receiving_yards"] = 40.0
        models = {
            # 25% edge on receptions, 20% on yards — heat keeps the best only.
            "player_receptions": self._Model({"hot guy": 0.75}),
            "player_reception_yds": _YdsModel({"hot guy": 0.70}),
        }
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, hist, season=2026,
        )
        _, n_edges, heat = ranked[0]
        assert n_edges == 1
        assert heat == pytest.approx(0.25)

    def test_inactive_player_is_ignored(self):
        hist = self._history()
        # "hot guy" last seen two slates before the three-slate window that
        # "meh guy" defines (202504, 202505, 202601) — not quotable.
        hist = hist[~((hist["player_key"] == "hot guy") & (hist["t"] >= 202503))]
        models = {"player_receptions": self._Model({"hot guy": 0.75})}
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, hist, season=2026,
        )
        assert ranked[0][1] == 0

    def test_week_one_sees_last_seasons_regulars(self):
        # History ends at 2025 week 22 (Super Bowl); the event is 2026 week 1.
        # Under `recent_t >= asof_t - 3` nobody qualified; by rank of the
        # regular-season slates the board is populated — and the two-team
        # postseason slates must not shrink the window.
        rows = []
        for pk, team in (("hot guy", "KC"), ("meh guy", "MIA")):
            for w in range(10, 19):
                rows.append({"player_key": pk, "team": team, "t": 202500 + w,
                             "receptions": 4, "season_type": "REG"})
        for w in (19, 20, 21, 22):   # only KC in the playoffs
            rows.append({"player_key": "hot guy", "team": "KC", "t": 202500 + w,
                         "receptions": 4, "season_type": "POST"})
        hist = pd.DataFrame(rows)
        models = {"player_receptions": self._Model({"hot guy": 0.75, "meh guy": 0.75})}
        sched = pd.DataFrame({
            "home_team": ["KC", "MIA"], "away_team": ["BUF", "NYJ"], "season": [2026] * 2,
            "game_date": [pd.Timestamp("2026-09-13")] * 2, "week": [1, 1],
        })
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Miami Dolphins", "New York Jets"),
             self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, hist, season=2026, schedule=sched,
        )
        assert [n for _, n, _ in ranked] == [1, 1]

    def test_mid_season_window_is_the_last_three_slates(self):
        hist = self._history()
        # Event in week 6 of 2025 (asof 202506): window = 202503-202505.
        sched = pd.DataFrame({
            "home_team": ["KC"], "away_team": ["BUF"], "season": [2025],
            "game_date": [pd.Timestamp("2025-10-12")], "week": [6],
        })
        stale = hist[~((hist["player_key"] == "hot guy") & (hist["t"] >= 202503))]
        models = {"player_receptions": self._Model({"hot guy": 0.75, "meh guy": 0.75})}
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, stale, season=2025, schedule=sched,
        )
        assert ranked[0][1] == 0   # hot guy stale, meh guy is MIA

    def test_roster_overlay_moves_a_player_to_his_new_game(self):
        # "hot guy" last played for KC but the 2026 roster has him in MIA.
        models = {"player_receptions": self._Model({"hot guy": 0.75})}
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Miami Dolphins", "New York Jets"),
             self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, self._history(), season=2026, current_teams={"hot guy": "MIA"},
        )
        assert ranked[0][0]["home_team"] == "Miami Dolphins"
        assert ranked[0][1] == 1 and ranked[1][1] == 0

    def test_lopsided_probability_is_not_heat(self):
        # A book wouldn't hang -110/-110 on a 90/10 proposition.
        models = {"player_receptions": self._Model({"hot guy": 0.90})}
        ranked = props_script.rank_events_by_model_heat(
            [self._event("Kansas City Chiefs", "Buffalo Bills")],
            models, self._history(), season=2026,
        )
        assert ranked[0][1] == 0


class TestEventsCommencingToday:
    """The --today filter that makes a daily cron job schedule-aware."""

    @staticmethod
    def _event(local_dt):
        from datetime import timezone
        utc = local_dt.astimezone(timezone.utc)
        return {"commence_time": utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_team": "KC", "away_team": "BUF"}

    def test_keeps_only_local_today(self):
        from datetime import datetime, timedelta
        now = datetime.now().astimezone().replace(hour=8, minute=0)
        today_game = self._event(now.replace(hour=13, minute=0))
        thursday_game = self._event(now + timedelta(days=4))
        out = props_script._events_commencing_today([today_game, thursday_game], now=now)
        assert out == [today_game]

    def test_late_local_kickoff_still_counts_as_today(self):
        # SNF at 8:20pm local is already tomorrow in UTC — must still match.
        from datetime import datetime
        now = datetime.now().astimezone().replace(hour=8, minute=0)
        snf = self._event(now.replace(hour=20, minute=20))
        assert props_script._events_commencing_today([snf], now=now) == [snf]

    def test_started_game_is_dropped(self):
        # A second run after kickoff must not refresh saved picks with
        # in-play prices.
        from datetime import datetime
        now = datetime.now().astimezone().replace(hour=15, minute=0)
        early = self._event(now.replace(hour=13, minute=0))
        late = self._event(now.replace(hour=16, minute=25))
        assert props_script._events_commencing_today([early, late], now=now) == [late]

    def test_malformed_commence_time_is_dropped(self):
        assert props_script._events_commencing_today(
            [{"commence_time": "not-a-date"}, {"commence_time": ""}]
        ) == []


class TestPropPickLabels:
    def test_prop_label_names_player_market_side_line(self):
        from betting_agent.intelligence.picks import _pick_label
        c = _cand("Travis Kelce", "player_reception_yds", 0.16)
        c.line = 55.5
        c.pick_side = "under"
        assert _pick_label(c) == "Travis Kelce reception yds UNDER 55.5"

    def test_results_embed_names_the_player(self):
        from betting_agent.notifications.discord import _build_results_embed
        embed = _build_results_embed(
            {"total_bets": 1, "wins": 1, "losses": 0, "pushes": 0,
             "win_rate_pct": 100.0, "total_pnl": 2.27, "roi_pct": 90.9,
             "avg_edge_pct": 12.0},
            "NFL",
            pick_details=[{
                "result": "win", "pnl": 2.27, "odds": -110,
                "pick_side": "under", "bet_type": "prop",
                "player": "Travis Kelce", "market": "player_receptions",
                "line": 4.5, "home_team": "KC", "away_team": "BUF",
            }],
        )
        assert "Travis Kelce receptions under 4.5" in embed["description"]


class TestReplayWeekendHelpers:
    """day_slate / heat_board — the replicable weekend-replay pattern."""

    def test_day_slate_filters_to_the_calendar_day(self, monkeypatch):
        import scripts.replay as replay_mod

        sched = pd.DataFrame({
            "home_team": ["KC", "BUF", "DAL"],
            "away_team": ["LV", "MIA", "PHI"],
            "game_date": [pd.Timestamp("2025-12-28"), pd.Timestamp("2025-12-28"),
                          pd.Timestamp("2025-12-25")],
            "week": [17, 17, 17],
            "home_score": [20.0, 24.0, 30.0],
            "away_score": [10.0, 21.0, 27.0],
        })

        class _FakePolars:
            def to_pandas(self):
                return sched

        class _FakeLoader:
            def load_schedules(self, seasons):
                return _FakePolars()

        import betting_agent.sports.nfl.features as feat_mod
        import betting_agent.sports.nfl.loader as loader_mod
        monkeypatch.setattr(loader_mod, "NFLLoader", _FakeLoader)
        monkeypatch.setattr(feat_mod, "normalise_raw_schedules", lambda df: df)

        day, week, matchups = replay_mod.day_slate(2025, date(2025, 12, 28), False)
        assert day == date(2025, 12, 28)
        assert week == 17
        assert matchups == [("BUF", "MIA"), ("KC", "LV")]

    def test_heat_board_ranks_by_summed_edges(self):
        import scripts.replay as replay_mod

        hot = ("KC", "LV")
        cold = ("BUF", "MIA")
        def edge(e):
            return (e, "someone", "player_receptions", "under", 4.5, 0.5 + e, 5.0)
        board = replay_mod.heat_board(
            [cold, hot],
            {hot: [edge(0.20), edge(0.15)], cold: [edge(0.11)]},
        )
        assert [g for _, _, g in board] == [hot, cold]
        assert board[0][0] == pytest.approx(0.35)
        assert board[0][1] == 2


class TestFinalizeNflGames:
    """props.py writes games as 'scheduled'; grading needs them 'final'."""

    def _patch(self, monkeypatch, games, sched):
        import betting_agent.sports.nfl.results as results_mod

        class _Session:
            def query(self, model):
                return self

            def filter(self, *a, **kw):
                return self

            def all(self):
                return games

        class _Ctx:
            def __enter__(self):
                return _Session()

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(results_mod, "get_session", lambda: _Ctx())

        class _FakePolars:
            def to_pandas(self):
                return sched

        class _FakeLoader:
            def load_schedules(self, seasons):
                return _FakePolars()

        import betting_agent.sports.nfl.loader as loader_mod
        import betting_agent.sports.nfl.features as feat_mod
        monkeypatch.setattr(loader_mod, "NFLLoader", _FakeLoader)
        monkeypatch.setattr(feat_mod, "normalise_raw_schedules", lambda df: df)
        return results_mod

    def _sched(self):
        return pd.DataFrame({
            "home_team": ["KC"], "away_team": ["BUF"],
            "game_date": [pd.Timestamp("2025-09-07")],
            "home_score": [27.0], "away_score": [20.0],
        })

    def _game(self):
        from types import SimpleNamespace
        return SimpleNamespace(
            sport="NFL", status="scheduled", game_date=date(2025, 9, 7),
            home_team="KC", away_team="BUF", home_score=None, away_score=None,
        )

    def test_fills_scores_and_marks_final(self, monkeypatch):
        game = self._game()
        mod = self._patch(monkeypatch, [game], self._sched())
        assert mod.finalize_nfl_games() == 1
        assert game.status == "final"
        assert (game.home_score, game.away_score) == (27, 20)

    def test_unplayed_game_is_left_alone(self, monkeypatch):
        game = self._game()
        sched = self._sched()
        sched.loc[0, ["home_score", "away_score"]] = [None, None]
        mod = self._patch(monkeypatch, [game], sched)
        assert mod.finalize_nfl_games() == 0
        assert game.status == "scheduled"

    def test_date_mismatch_beyond_tolerance_does_not_match(self, monkeypatch):
        game = self._game()
        sched = self._sched()
        sched.loc[0, "game_date"] = pd.Timestamp("2025-10-19")
        mod = self._patch(monkeypatch, [game], sched)
        assert mod.finalize_nfl_games() == 0
        assert game.status == "scheduled"

    def test_no_pending_games_skips_the_load(self, monkeypatch):
        mod = self._patch(monkeypatch, [], self._sched())
        assert mod.finalize_nfl_games() == 0


class TestScheduleHelpers:
    """Exact NFL week from the published schedule (replaces the date heuristic)."""

    def _sched(self):
        return pd.DataFrame({
            "home_team": ["KC", "BUF", "DAL"], "away_team": ["LV", "MIA", "PHI"],
            "season": [2026] * 3, "week": [1, 1, 2],
            "game_date": [pd.Timestamp("2026-09-13"), pd.Timestamp("2026-09-13"),
                          pd.Timestamp("2026-09-20")],
        })

    def test_matchup_and_date_resolve_the_week(self):
        from betting_agent.sports.nfl.props import nfl_week_for
        assert nfl_week_for(date(2026, 9, 13), 2026, self._sched(), "KC", "LV") == 1
        assert nfl_week_for(date(2026, 9, 20), 2026, self._sched(), "DAL", "PHI") == 2

    def test_unknown_matchup_falls_back_to_nearest_date(self):
        from betting_agent.sports.nfl.props import nfl_week_for
        assert nfl_week_for(date(2026, 9, 21), 2026, self._sched(), "NYG", "WAS") == 2

    def test_no_schedule_uses_heuristic(self):
        from betting_agent.sports.nfl.props import nfl_week_for
        assert nfl_week_for(date(2026, 9, 13), 2026, None) == 2
        assert nfl_week_for(date(2026, 9, 13), 2026, pd.DataFrame()) == 2

    def test_far_date_uses_heuristic(self):
        from betting_agent.sports.nfl.props import nfl_week_for
        # Nothing within two days of Oct 30 in this schedule.
        assert nfl_week_for(date(2026, 10, 30), 2026, self._sched()) == 9

    def test_active_window_is_by_rank_not_subtraction(self):
        from betting_agent.sports.nfl.props import active_player_keys
        hist = pd.DataFrame({
            "player_key": ["a", "a", "a", "b"], "t": [202516, 202517, 202518, 202510],
        })
        assert active_player_keys(hist, 202601) == {"a"}
        assert active_player_keys(hist, 202510) == set()

    def test_game_lines_from_schedule(self):
        sched = self._sched().assign(spread_line=[-3.5, 1.0, None], total_line=[47.5, 44.0, 50.0])
        events = [{"id": "e1", "home_team": "Kansas City Chiefs",
                   "away_team": "Las Vegas Raiders", "commence_time": "2026-09-13T17:00:00Z"},
                  {"id": "e3", "home_team": "Dallas Cowboys",
                   "away_team": "Philadelphia Eagles", "commence_time": "2026-09-20T20:25:00Z"}]
        assert props_script.game_lines_from_schedule(events, sched) == {
            "e1": (-3.5, 47.5), "e3": (None, 50.0),
        }


class TestBookPreference:
    def _event(self, *keys, empty=()):
        def book(key):
            markets = [] if key in empty else [{"key": "player_receptions", "outcomes": [
                {"name": "Over", "description": "X", "point": 4.5, "price": -110}]}]
            return {"key": key, "markets": markets}
        return {"id": "e", "bookmakers": [book(k) for k in keys]}

    def test_first_ordered_book_that_posted_wins(self):
        ev = self._event("fanduel", "draftkings")
        assert [b["key"] for b in books_in_preference(ev, ["bet365", "draftkings", "fanduel"])] \
            == ["draftkings"]

    def test_book_with_empty_markets_does_not_count_as_posted(self):
        ev = self._event("bet365", "draftkings", empty=("bet365",))
        assert [b["key"] for b in books_in_preference(ev, ["bet365", "draftkings"])] \
            == ["draftkings"]

    def test_no_ordered_book_posted_falls_back_to_everything(self):
        ev = self._event("bovada", "betrivers")
        assert [b["key"] for b in books_in_preference(ev, ["bet365"])] == ["bovada", "betrivers"]
        assert [b["key"] for b in books_in_preference(ev, None)] == ["bovada", "betrivers"]

    def test_default_order_is_preferred_then_fallbacks(self, monkeypatch):
        from betting_agent.config import settings
        monkeypatch.setattr(settings, "preferred_bookmakers", "")
        monkeypatch.setattr(settings, "prop_fallback_bookmakers", "draftkings,fanduel")
        assert prop_bookmaker_order() == ["bet365", "draftkings", "fanduel"]
        monkeypatch.setattr(settings, "preferred_bookmakers", "fanduel")
        assert prop_bookmaker_order() == ["fanduel", "draftkings"]
