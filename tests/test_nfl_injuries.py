"""Deterministic injury / QB1 flags on NFL prop candidates."""

from __future__ import annotations

from datetime import date

import pandas as pd

from betting_agent.intelligence.picks import BetCandidate
from betting_agent.intelligence.validator.deterministic_checks import build_deterministic_flags
from betting_agent.sports.nfl.injuries import (
    apply_injury_policy,
    prop_injury_flags,
    qb1_from_depth_chart,
)


def _cand(player, team="KC"):
    return BetCandidate(
        game_id=0, external_id="g1", home_team="Kansas City Chiefs",
        away_team="Buffalo Bills", game_date=date(2026, 9, 13), sport="NFL",
        bet_type="prop", pick_side="over", player=player, market="player_receptions",
        line=4.5, model_prob=0.65, implied_prob=0.5, edge=0.15, odds=-110,
        kelly_fraction=0.04, recommended_bet=40.0, bankroll_at_pick=1000.0,
        extra={"team": team},
    )


def _injuries(rows):
    return pd.DataFrame(rows, columns=[
        "full_name", "team", "position", "report_status", "practice_status", "gsis_id",
    ])


class TestPropInjuryFlags:
    def test_out_and_doubtful_are_drops(self):
        cands = [_cand("Travis Kelce"), _cand("Rashee Rice"), _cand("Xavier Worthy")]
        inj = _injuries([
            ("Travis Kelce", "KC", "TE", "Out", None, "1"),
            ("Rashee Rice", "KC", "WR", "Doubtful", None, "2"),
            ("Xavier Worthy", "KC", "WR", None, "Full Participation in Practice", "3"),
        ])
        flags = prop_injury_flags(cands, inj, {}, {})
        assert {(f.player, f.drop, f.severity) for f in flags} == {
            ("Travis Kelce", True, "high"), ("Rashee Rice", True, "high"),
        }

    def test_questionable_is_a_medium_flag_not_a_drop(self):
        inj = _injuries([("Travis Kelce", "KC", "TE", "Questionable", None, "1")])
        flags = prop_injury_flags([_cand("Travis Kelce")], inj, {}, {})
        assert len(flags) == 1
        assert flags[0].drop is False and flags[0].severity == "medium"

    def test_qb1_out_flags_every_prop_on_that_team_only(self):
        cands = [_cand("Travis Kelce", "KC"), _cand("Khalil Shakir", "BUF")]
        inj = _injuries([("Patrick Mahomes", "KC", "QB", "Out", None, "qb-kc")])
        qb1 = {"KC": ("Patrick Mahomes", "qb-kc"), "BUF": ("Josh Allen", "qb-buf")}
        flags = prop_injury_flags(cands, inj, qb1,
                                  {"travis kelce": "KC", "khalil shakir": "BUF"})
        assert [(f.type, f.player, f.drop) for f in flags] == [("qb_out", "Travis Kelce", False)]

    def test_qb1_matched_by_name_when_gsis_missing(self):
        inj = _injuries([("Patrick Mahomes", "KC", "QB", "Doubtful", None, None)])
        flags = prop_injury_flags([_cand("Travis Kelce")], inj,
                                  {"KC": ("Patrick Mahomes", None)}, {"travis kelce": "KC"})
        assert flags and flags[0].type == "qb_out"

    def test_missing_data_means_no_flags(self):
        assert prop_injury_flags([_cand("Travis Kelce")], None, None) == []
        assert prop_injury_flags([_cand("Travis Kelce")], _injuries([]), {}, {}) == []
        assert prop_injury_flags([], _injuries([("X", "KC", "WR", "Out", None, "1")]), {}) == []


class TestApplyInjuryPolicy:
    def test_drops_flagged_players_and_attaches_the_rest(self):
        kelce, rice = _cand("Travis Kelce"), _cand("Rashee Rice")
        inj = _injuries([
            ("Travis Kelce", "KC", "TE", "Out", None, "1"),
            ("Rashee Rice", "KC", "WR", "Questionable", None, "2"),
        ])
        flags = prop_injury_flags([kelce, rice], inj, {}, {})
        kept = apply_injury_policy([kelce, rice], flags)
        assert kept == [rice]
        assert rice.extra["flags"][0]["detail"].startswith("Rashee Rice listed Questionable")
        assert "Rashee Rice listed Questionable" in rice.agent_reasons[0]

    def test_no_flags_is_a_no_op(self):
        cands = [_cand("Travis Kelce")]
        assert apply_injury_policy(cands, []) is cands


class TestQb1FromDepthChart:
    def test_latest_chart_per_team_and_qb_rank_one(self):
        dc = pd.DataFrame({
            "team": ["KC", "KC", "KC", "BUF", "BUF"],
            "dt": ["2026-09-01T00:00:00Z", "2026-09-05T00:00:00Z", "2026-09-05T00:00:00Z",
                   "2026-09-03T00:00:00Z", "2026-09-03T00:00:00Z"],
            "pos_abb": ["QB", "QB", "QB", "QB", "WR"],
            "pos_grp": ["3WR 1TE"] * 5,   # formation label, must be ignored
            "pos_rank": [1, 1, 2, 1, 1],
            "player_name": ["Old Starter", "New Starter", "Backup", "Josh Allen", "K Shakir"],
            "gsis_id": ["a", "b", "c", "d", "e"],
        })
        assert qb1_from_depth_chart(dc) == {"KC": ("New Starter", "b"), "BUF": ("Josh Allen", "d")}

    def test_missing_columns_gives_empty(self):
        assert qb1_from_depth_chart(pd.DataFrame({"team": ["KC"]})) == {}
        assert qb1_from_depth_chart(pd.DataFrame()) == {}


class TestBuildDeterministicFlagsForProps:
    def test_nfl_prop_flags_flow_through(self):
        inj = _injuries([("Travis Kelce", "KC", "TE", "Out", None, "1")])
        flags = build_deterministic_flags([_cand("Travis Kelce")], "NFL", injuries=inj, qb1={})
        assert [f.type for f in flags] == ["player_injury"]

    def test_no_injury_inputs_no_prop_flags(self):
        assert build_deterministic_flags([_cand("Travis Kelce")], "NFL") == []
