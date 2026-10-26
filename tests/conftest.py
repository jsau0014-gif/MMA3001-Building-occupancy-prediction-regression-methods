"""Reusable isolated workspaces for testing the MMA3001 scripts.

The scripts normally locate Data/ relative to their own src/ folder. Each test
copies the requested script into a temporary miniature project, so tests never
read, overwrite or modify the student's actual datasets or results.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def workspace(tmp_path: Path):
    """Return an isolated runner for an unmodified project script."""
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src"
    (tmp_path / "src").mkdir()
    (tmp_path / "Data" / "processed").mkdir(parents=True)
    (tmp_path / "Results").mkdir()

    def run(script_name: str):
        source_path = source_root / script_name
        assert source_path.is_file(), f"Missing project script: {source_path}"
        target_path = tmp_path / "src" / script_name
        shutil.copy2(source_path, target_path)
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Avoid opening graph windows during pytest.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        p = subprocess.run(
            [sys.executable, str(target_path)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert p.returncode == 0, (
            f"Script {script_name} exited {p.returncode}.\n"
            f"STDOUT:\n{p.stdout[-7000:]}\nSTDERR:\n{p.stderr[-7000:]}"
        )
        return p

    return tmp_path, run
