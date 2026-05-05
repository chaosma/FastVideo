# SPDX-License-Identifier: Apache-2.0
"""Smoke-test the inference CLI is wired up.

We don't run actual generation here (that needs weights + GPU). The CLI
surface — `fastvideo --help` and `fastvideo generate --help` — must
import cleanly and return successfully.
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fastvideo.entrypoints.cli.main", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_top_level_help() -> None:
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "FastVideo CLI" in result.stdout
    assert "generate" in result.stdout


def test_cli_generate_help() -> None:
    result = _run("generate", "--help")
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_cli_generate_requires_config() -> None:
    result = _run("generate")
    assert result.returncode != 0
    assert "config" in (result.stdout + result.stderr).lower()


def test_video_generator_importable() -> None:
    from fastvideo import VideoGenerator  # noqa: F401
    from fastvideo.worker import MultiprocExecutor  # noqa: F401
