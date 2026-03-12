from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from betting_agent.db.backup import create_postgres_backup, prune_old_backups


def test_create_postgres_backup_runs_pg_dump_and_prunes(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def fake_run(command, check, env):
        assert check is True
        commands.append(command)
        temp_path = Path(command[3].split("=", 1)[1])
        temp_path.write_bytes(b"backup")
        assert env["PGPASSWORD"] == "secret"
        return None

    stale_backup = tmp_path / "old.dump"
    stale_backup.write_bytes(b"old")
    old_time = (datetime(2026, 3, 12, 3, 15, 0) - timedelta(days=31)).timestamp()
    os.utime(stale_backup, (old_time, old_time))

    monkeypatch.setattr("betting_agent.db.backup.shutil.which", lambda name: "/usr/bin/pg_dump")
    monkeypatch.setattr("betting_agent.db.backup.subprocess.run", fake_run)

    now = datetime(2026, 3, 12, 3, 15, 0)
    result = create_postgres_backup(
        "postgresql://postgres:secret@localhost:5432/betting_agent",
        tmp_path,
        retention_days=30,
        now=now,
    )

    assert result.backup_path == tmp_path / "betting_agent_20260312_031500.dump"
    assert result.backup_path.read_bytes() == b"backup"
    assert result.pruned_paths == (stale_backup,)
    assert not stale_backup.exists()
    assert commands == [
        [
            "/usr/bin/pg_dump",
            "--format=custom",
            "--compress=9",
            f"--file={tmp_path / '.betting_agent_20260312_031500.dump.tmp'}",
            "--dbname=postgresql://postgres@localhost:5432/betting_agent",
        ]
    ]


def test_create_postgres_backup_rejects_non_postgres_url(monkeypatch, tmp_path):
    monkeypatch.setattr("betting_agent.db.backup.shutil.which", lambda name: "/usr/bin/pg_dump")

    with pytest.raises(ValueError, match="PostgreSQL"):
        create_postgres_backup("sqlite:///tmp/test.db", tmp_path)


def test_create_postgres_backup_surfaces_pg_dump_failure(monkeypatch, tmp_path):
    def fake_run(command, check, env):
        raise subprocess.CalledProcessError(returncode=2, cmd=command)

    monkeypatch.setattr("betting_agent.db.backup.shutil.which", lambda name: "/usr/bin/pg_dump")
    monkeypatch.setattr("betting_agent.db.backup.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 2"):
        create_postgres_backup(
            "postgresql://postgres:secret@localhost:5432/betting_agent",
            tmp_path,
        )

    assert list(tmp_path.iterdir()) == []


def test_prune_old_backups_only_removes_files_older_than_cutoff(tmp_path):
    now = datetime(2026, 3, 12, 3, 15, 0)
    fresh_backup = tmp_path / "fresh.dump"
    stale_backup = tmp_path / "stale.dump"
    fresh_backup.write_bytes(b"fresh")
    stale_backup.write_bytes(b"stale")

    fresh_time = (now - timedelta(days=5)).timestamp()
    stale_time = (now - timedelta(days=45)).timestamp()
    os.utime(fresh_backup, (fresh_time, fresh_time))
    os.utime(stale_backup, (stale_time, stale_time))

    pruned = prune_old_backups(tmp_path, retention_days=30, now=now)

    assert pruned == [stale_backup]
    assert fresh_backup.exists()
    assert not stale_backup.exists()
