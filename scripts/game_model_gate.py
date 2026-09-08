#!/usr/bin/env python
"""
Gate for the market-anchored NFL game model.

Walk-forward by season against REAL closing lines from nflreadpy. For each
residual learner (none = market only, ridge, xgb) it reports:

  1. Win probability vs the vig-removed closing moneyline: Brier, log loss,
     accuracy, and who is right when they disagree (the anti-predictive
     pattern that sank the from-scratch model must be gone).
  2. Flat-stake ROI at closing prices for moneyline / spread / total at
     several edge floors, with a per-season sign count.

Pass = a learner beats "none" on Brier AND its spread/total ROI at the
chosen floor is positive in most seasons. Anything else means the residual
is noise and the game line on the card should be labelled a market lean.

Usage:
    uv run python scripts/game_model_gate.py --start-season 2012 --end-season 2025 \
        [--min-train-seasons 5] [--kinds none ridge xgb] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import logging
from functools import partial

import numpy as np
import pandas as pd

from betting_agent.intelligence.ev import (
    american_to_implied_prob,
    calculate_edge_fair,
    calculate_spread_edge,
    calculate_total_edge,
    remove_vig,
)
from betting_agent.models.market_anchored import KINDS, TARGET_COLS, MarketAnchoredModel
from betting_agent.sports.nfl import market
from betting_agent.sports.nfl.features import build_nfl_features, normalise_raw_schedules
from betting_agent.sports.nfl.loader import NFLLoader

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EDGE_FLOORS = (0.01, 0.02, 0.03, 0.05)
STAKE = 100.0


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _xy(features: pd.DataFrame):
    complete = features[features["home_score"].notna() & features["away_score"].notna()]
    X = complete.drop(columns=[c for c in TARGET_COLS if c in complete.columns])
    return X, complete["home_score"].astype(float), complete["away_score"].astype(float)


def collect(seasons: list[int], min_train: int, kinds: list[str]) -> pd.DataFrame:
    raw = normalise_raw_schedules(NFLLoader().load_schedules(seasons).to_pandas())
    if not market.has_closing_lines(raw):
        raise SystemExit("schedule data carries no closing lines")
    build = partial(build_nfl_features, market_anchored=True)

    rows: list[dict] = []
    for i in range(min_train, len(seasons)):
        train_seasons, test_season = seasons[:i], seasons[i]
        logger.warning("train %s → test %s", f"{train_seasons[0]}-{train_seasons[-1]}", test_season)
        X_tr, h_tr, a_tr = _xy(build(raw[raw["season"].isin(train_seasons)]))
        models = {k: MarketAnchoredModel(k).fit(X_tr, h_tr, a_tr) for k in kinds}

        test = raw[raw["season"] == test_season].copy().reset_index(drop=True)
        test["_row_id"] = range(len(test))
        feats = build(test)
        feats = feats[feats["home_score"].notna() & feats["away_score"].notna()]
        row_ids = feats["_row_id"].astype(int).tolist()
        X_te = feats.drop(columns=[c for c in (*TARGET_COLS, "_row_id") if c in feats.columns])
        X_te = X_te.reindex(columns=list(X_tr.columns))
        preds = {k: m.predict(X_te) for k, m in models.items()}
        src = test.set_index("_row_id")

        for j, rid in enumerate(row_ids):
            g = src.loc[rid]
            if pd.isna(g.get("spread_line")) or pd.isna(g.get("total_line")):
                continue
            rec = {
                "season": test_season, "home_team": g["home_team"], "away_team": g["away_team"],
                "home_score": float(g["home_score"]), "away_score": float(g["away_score"]),
                "spread_line": float(g["spread_line"]), "total_line": float(g["total_line"]),
                "home_moneyline": g.get("home_moneyline"), "away_moneyline": g.get("away_moneyline"),
                "home_spread_odds": g.get("home_spread_odds"), "away_spread_odds": g.get("away_spread_odds"),
                "over_odds": g.get("over_odds"), "under_odds": g.get("under_odds"),
            }
            if pd.notna(rec["home_moneyline"]) and pd.notna(rec["away_moneyline"]):
                rec["market_prob"], _ = remove_vig(
                    american_to_implied_prob(rec["home_moneyline"]),
                    american_to_implied_prob(rec["away_moneyline"]))
            else:
                rec["market_prob"] = np.nan
            for k in kinds:
                p = preds[k].iloc[j]
                rec[f"{k}_win_prob"] = float(p["win_prob"])
                rec[f"{k}_margin"] = float(p["pred_margin"])
                rec[f"{k}_total"] = float(p["pred_total"])
                rec[f"{k}_margin_sigma"] = models[k].margin_sigma
                rec[f"{k}_total_sigma"] = models[k].total_sigma
            rows.append(rec)
    return pd.DataFrame(rows)


def bets_for(df: pd.DataFrame, kind: str, floor: float) -> pd.DataFrame:
    """Flat-stake bets at closing prices for one learner and edge floor."""
    out = []
    for _, g in df.iterrows():
        hs, as_ = g["home_score"], g["away_score"]
        base = {"season": g["season"]}
        wp = g[f"{kind}_win_prob"]
        if pd.notna(g["home_moneyline"]) and pd.notna(g["away_moneyline"]):
            for side, prob, odds in (("home", wp, g["home_moneyline"]), ("away", 1 - wp, g["away_moneyline"])):
                edge = calculate_edge_fair(prob, g["home_moneyline"], g["away_moneyline"], pick_home=(side == "home"))
                if edge >= floor:
                    res = market.grade_moneyline(hs, as_, side)
                    out.append({**base, "market": "moneyline", "edge": edge, "result": res,
                                "pnl": market.payout(STAKE, odds, res)})
        if pd.notna(g["home_spread_odds"]) and pd.notna(g["away_spread_odds"]):
            cover, edge = calculate_spread_edge(g[f"{kind}_margin"], g["spread_line"], g["home_spread_odds"],
                                               away_odds=g["away_spread_odds"], sigma=g[f"{kind}_margin_sigma"])
            if edge >= floor:
                side, odds = "home", g["home_spread_odds"]
            else:
                fair_away = 1 - cover
                ia, ih = american_to_implied_prob(g["away_spread_odds"]), american_to_implied_prob(g["home_spread_odds"])
                edge = fair_away - ia / (ia + ih)
                side, odds = "away", g["away_spread_odds"]
            if edge >= floor:
                res = market.grade_spread(hs, as_, g["spread_line"], side)
                out.append({**base, "market": "spread", "edge": edge, "result": res,
                            "pnl": market.payout(STAKE, odds, res)})
        if pd.notna(g["over_odds"]) and pd.notna(g["under_odds"]):
            for side, odds, other in (("over", g["over_odds"], g["under_odds"]), ("under", g["under_odds"], g["over_odds"])):
                _, edge = calculate_total_edge(g[f"{kind}_total"], g["total_line"], odds, side,
                                               other_odds=other, sigma=g[f"{kind}_total_sigma"])
                if edge >= floor:
                    res = market.grade_total(hs, as_, g["total_line"], side)
                    out.append({**base, "market": "total", "edge": edge, "result": res,
                                "pnl": market.payout(STAKE, odds, res)})
    return pd.DataFrame(out)


def report(df: pd.DataFrame, kinds: list[str]) -> None:
    y = (df["home_score"] > df["away_score"]).astype(int).to_numpy()
    mkt = df["market_prob"].to_numpy()
    ok = ~np.isnan(mkt)
    print(f"\nGames: {len(df)}  seasons {df['season'].min()}-{df['season'].max()}\n")
    print("== Win probability vs closing moneyline ==")
    print(f"{'':16}{'Brier':>9}{'LogLoss':>10}{'Acc':>8}")
    print(f"{'market (close)':16}{brier(mkt[ok], y[ok]):>9.4f}{log_loss(mkt[ok], y[ok]):>10.4f}"
          f"{((mkt[ok] > .5) == y[ok]).mean():>7.1%}")
    for k in kinds:
        p = df[f"{k}_win_prob"].to_numpy()
        print(f"{k:16}{brier(p[ok], y[ok]):>9.4f}{log_loss(p[ok], y[ok]):>10.4f}{((p[ok] > .5) == y[ok]).mean():>7.1%}")

    print("\n== When model and market pick opposite sides, model right % (n) ==")
    print(f"{'gap':>10}" + "".join(f"{k:>18}" for k in kinds))
    for lo, hi in ((0.02, 0.05), (0.05, 0.10), (0.10, 1.01)):
        cells = []
        for k in kinds:
            p = df[f"{k}_win_prob"]
            gap = (p - df["market_prob"]).abs()
            band = df[ok & gap.between(lo, hi)]
            opp = band[(band[f"{k}_win_prob"] > .5) != (band["market_prob"] > .5)]
            if opp.empty:
                cells.append(f"{'—':>18}")
            else:
                right = ((opp[f"{k}_win_prob"] > .5).astype(int) == (opp["home_score"] > opp["away_score"]).astype(int)).mean()
                cells.append(f"{right:>11.1%} ({len(opp):>4})")
        print(f"{f'{lo:.0%}-{min(hi, 1):.0%}':>10}" + "".join(cells))

    print("\n== Margin / total RMSE (points) ==")
    m_act = df["home_score"] - df["away_score"]
    t_act = df["home_score"] + df["away_score"]
    print(f"{'market line':16} margin {np.sqrt(np.mean((m_act - df['spread_line']) ** 2)):6.2f}"
          f"  total {np.sqrt(np.mean((t_act - df['total_line']) ** 2)):6.2f}")
    for k in kinds:
        print(f"{k:16} margin {np.sqrt(np.mean((m_act - df[f'{k}_margin']) ** 2)):6.2f}"
              f"  total {np.sqrt(np.mean((t_act - df[f'{k}_total']) ** 2)):6.2f}")

    print(f"\n== Flat ${STAKE:.0f} at closing prices: ROI% (bets) [seasons positive/total] ==")
    for k in kinds:
        if k == "none":
            continue
        print(f"\n-- {k} --")
        print(f"{'floor':>6}" + "".join(f"{m:>30}" for m in ("moneyline", "spread", "total")))
        for floor in EDGE_FLOORS:
            b = bets_for(df, k, floor)
            cells = []
            for m in ("moneyline", "spread", "total"):
                sub = b[(b["market"] == m) & (b["result"] != market.PUSH)] if not b.empty else b
                if sub.empty:
                    cells.append(f"{'no bets':>30}")
                    continue
                roi = sub["pnl"].sum() / (STAKE * len(sub)) * 100
                by_season = sub.groupby("season")["pnl"].sum()
                cells.append(f"{roi:+7.2f}% ({len(sub):>5}) [{(by_season > 0).sum()}/{len(by_season)}]".rjust(30))
            print(f"{floor:>6.0%}" + "".join(cells))
    print("\nPass = beats 'none' on Brier AND spread/total ROI positive in most seasons at a floor "
          "with a usable bet count. Otherwise the residual is noise: label the game line a market lean.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Market-anchored NFL game model gate")
    ap.add_argument("--start-season", type=int, default=2012)
    ap.add_argument("--end-season", type=int, default=2025)
    ap.add_argument("--min-train-seasons", type=int, default=5)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--csv", type=str, default=None, help="write per-game predictions here")
    args = ap.parse_args()
    seasons = list(range(args.start_season, args.end_season + 1))
    df = collect(seasons, args.min_train_seasons, args.kinds)
    if args.csv:
        df.to_csv(args.csv, index=False)
    report(df, args.kinds)


if __name__ == "__main__":
    main()
