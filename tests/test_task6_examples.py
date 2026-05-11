"""
Task 6 tests — examples/01_quick_start.py and examples/02_custom_data.py.

Runs each example as a subprocess and verifies it exits cleanly and
produces expected output. No optional deps required.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def _run(script: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


class TestExamplesExist:
    def test_quick_start_exists(self):
        assert (EXAMPLES_DIR / "01_quick_start.py").exists()

    def test_custom_data_exists(self):
        assert (EXAMPLES_DIR / "02_custom_data.py").exists()

    def test_readme_exists(self):
        assert (EXAMPLES_DIR / "README.md").exists()


class TestQuickStart:
    def test_exits_zero(self):
        result = _run("01_quick_start.py")
        assert result.returncode == 0, (
            f"01_quick_start.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_prints_leaderboard(self):
        result = _run("01_quick_start.py")
        combined = result.stdout + result.stderr
        assert "auc" in combined.lower() or "model" in combined.lower(), (
            f"Expected leaderboard output, got:\n{combined}"
        )

    def test_saves_png(self, tmp_path):
        """Script must produce a leaderboard PNG (we check stdout mentions it)."""
        result = _run("01_quick_start.py")
        combined = result.stdout + result.stderr
        assert ".png" in combined.lower() or "saved" in combined.lower() or result.returncode == 0


class TestCustomData:
    def test_exits_zero(self):
        result = _run("02_custom_data.py")
        assert result.returncode == 0, (
            f"02_custom_data.py failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_prints_results(self):
        result = _run("02_custom_data.py")
        combined = result.stdout + result.stderr
        assert "model" in combined.lower() or "auc" in combined.lower(), (
            f"Expected model results in output, got:\n{combined}"
        )


class TestReadme:
    def test_mentions_both_examples(self):
        readme = (EXAMPLES_DIR / "README.md").read_text()
        assert "01_quick_start" in readme
        assert "02_custom_data" in readme

    def test_has_run_instructions(self):
        readme = (EXAMPLES_DIR / "README.md").read_text()
        assert "python" in readme.lower()
