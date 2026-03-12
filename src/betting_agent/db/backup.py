from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode, urlunsplit

from sqlalchemy.engine import URL, make_url


@dataclass(frozen=True)
class BackupResult:
    backup_path: Path
    pruned_paths: tuple[Path, ...]


def create_postgres_backup(
    database_url: str,
    output_dir: str | Path,
    retention_days: int = 30,
    now: datetime | None = None,
) -> BackupResult:
    if retention_days < 0:
        raise ValueError("retention_days must be >= 0")

    pg_dump_path = shutil.which("pg_dump")
    if pg_dump_path is None:
        raise RuntimeError("pg_dump is not installed or not in PATH")

    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("Database backups are only supported for PostgreSQL DATABASE_URL values")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    backup_time = now or datetime.now()
    backup_name = f"betting_agent_{backup_time.strftime('%Y%m%d_%H%M%S')}.dump"
    backup_path = output_path / backup_name
    temp_path = output_path / f".{backup_name}.tmp"

    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password

    command = [
        pg_dump_path,
        "--format=custom",
        "--compress=9",
        f"--file={temp_path}",
        f"--dbname={_build_pg_dump_dsn(url)}",
    ]

    try:
        subprocess.run(command, check=True, env=env)
        temp_path.replace(backup_path)
    except subprocess.CalledProcessError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed with exit code {exc.returncode}") from exc
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    pruned_paths = tuple(prune_old_backups(output_path, retention_days, now=backup_time))
    return BackupResult(backup_path=backup_path, pruned_paths=pruned_paths)


def prune_old_backups(
    output_dir: str | Path,
    retention_days: int,
    now: datetime | None = None,
) -> list[Path]:
    if retention_days < 0:
        raise ValueError("retention_days must be >= 0")

    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    current_time = now or datetime.now()
    cutoff_timestamp = current_time.timestamp() - (retention_days * 24 * 60 * 60)
    pruned_paths: list[Path] = []

    for backup_path in output_path.glob("*.dump"):
        if backup_path.stat().st_mtime < cutoff_timestamp:
            backup_path.unlink()
            pruned_paths.append(backup_path)

    return pruned_paths


def _build_pg_dump_dsn(url: URL) -> str:
    username = quote(url.username, safe="") if url.username else ""
    host = url.host or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    netloc = ""
    if username:
        netloc += f"{username}@"
    netloc += host
    if url.port is not None:
        netloc += f":{url.port}"

    path = f"/{url.database}" if url.database else ""
    query = urlencode(url.query, doseq=True)
    return urlunsplit(("postgresql", netloc, path, query, ""))
