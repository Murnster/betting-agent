from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_exported_env_overrides_dotenv(tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=fresh-from-dotenv\n", encoding="utf-8")

    env = os.environ.copy()
    env["GEMINI_API_KEY"] = "from-env"

    src_path = str(Path(__file__).resolve().parents[1] / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from betting_agent.config import settings; print(settings.gemini_api_key)",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "from-env"
