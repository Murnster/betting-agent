from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import scripts.backup_db as backup_db_script


def test_main_uses_settings_database_url_and_cli_defaults(monkeypatch):
    captured = {}
    result = type(
        "BackupResult",
        (),
        {"backup_path": Path("backups/postgres/example.dump"), "pruned_paths": ()},
    )()

    monkeypatch.setattr(
        backup_db_script.argparse.ArgumentParser,
        "parse_args",
        lambda self: Namespace(output_dir=Path("backups/postgres"), retention_days=30),
    )
    monkeypatch.setattr(
        backup_db_script,
        "create_postgres_backup",
        lambda database_url, output_dir, retention_days: (
            captured.update(
                {
                    "database_url": database_url,
                    "output_dir": output_dir,
                    "retention_days": retention_days,
                }
            )
            or result
        ),
    )

    class _Settings:
        database_url = "postgresql://postgres:secret@localhost:5432/betting_agent"

    monkeypatch.setattr("betting_agent.config.settings", _Settings())

    backup_db_script.main()

    assert captured == {
        "database_url": "postgresql://postgres:secret@localhost:5432/betting_agent",
        "output_dir": Path("backups/postgres"),
        "retention_days": 30,
    }
