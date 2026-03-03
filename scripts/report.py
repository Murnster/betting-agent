#!/usr/bin/env python
"""
View performance reports for previous picks.

Usage:
    uv run python scripts/report.py [--sport NFL] [--season 2024]
"""

from __future__ import annotations

import argparse
import logging

from betting_agent.accounting.roi import format_roi_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(description="View ROI reports")
    parser.add_argument("--sport", type=str, default=None, help="Filter by sport (NFL, NBA)")
    parser.add_argument("--season", type=int, default=None, help="Filter by season (e.g. 2024)")
    args = parser.parse_args()

    try:
        report = format_roi_report(sport=args.sport, season=args.season)
        print(report)
    except Exception as exc:
        logger.error("Failed to generate report: %s", exc)

if __name__ == "__main__":
    main()
