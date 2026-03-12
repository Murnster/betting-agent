#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from betting_agent.db.backup import create_postgres_backup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(PROJECT_ROOT)

    from betting_agent.config import settings

    parser = argparse.ArgumentParser(description="Back up the configured PostgreSQL database")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backups/postgres"),
        help="Directory where backup files should be written",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        help="Delete backup files older than this many days",
    )
    args = parser.parse_args()

    result = create_postgres_backup(
        database_url=settings.database_url,
        output_dir=args.output_dir,
        retention_days=args.retention_days,
    )
    logger.info("Database backup created: %s", result.backup_path)
    if result.pruned_paths:
        logger.info("Pruned %d old backup(s)", len(result.pruned_paths))


if __name__ == "__main__":
    main()
