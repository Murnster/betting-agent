#!/usr/bin/env python
"""
View performance reports for previous picks.

Usage:
    uv run python scripts/report.py [--sport NFL] [--season 2024]
"""

from __future__ import annotations

import argparse
import logging

from betting_agent.accounting.roi import carded_only, format_roi_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="View ROI reports")
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport (NFL, NBA)")
    parser.add_argument("--season", type=int, default=None, help="Filter by season (e.g. 2024)")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--off-card", action="store_true",
                       help="Report the off-card picks instead — everything that cleared the "
                            "floors but never made a card. Model evaluation, not bets.")
    scope.add_argument("--all-picks", action="store_true",
                       help="Report carded and off-card picks together")
    args = parser.parse_args()

    # Default scope is the sport's own: for NFL, the picks that made a card.
    on_card = False if args.off_card else (None if args.all_picks else carded_only(args.sport))

    try:
        report = format_roi_report(sport=args.sport, season=args.season, on_card=on_card)
        print(report)
    except Exception as exc:
        logger.error("Failed to generate report: %s", exc)

if __name__ == "__main__":
    main()
