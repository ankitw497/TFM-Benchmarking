"""
Task 9 tests — tfm_benchmark/cli.py

Runs the CLI as a subprocess via `python -m tfm_benchmark.cli` (the installed
`tfm-benchmark` entry-point delegates to the same main() function).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CLI = [sys.executable, "-m", "tfm_benchmark.cli"]


def run_cli(*args, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        CLI + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# --help / --version
# ---------------------------------------------------------------------------

class TestCLIHelp:
    def test_help_exits_zero(self):
        result = run_cli("--help")
        assert result.returncode == 0

    def test_help_mentions_list_models(self):
        result = run_cli("--help")
        assert "list-models" in result.stdout

    def test_help_mentions_list_datasets(self):
        result = run_cli("--help")
        assert "list-datasets" in result.stdout

    def test_help_mentions_run(self):
        result = run_cli("--help")
        assert "run" in result.stdout

    def test_version_exits_zero(self):
        result = run_cli("--version")
        assert result.returncode == 0

    def test_version_shows_version_number(self):
        result = run_cli("--version")
        assert "0.1.0" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------

class TestCLIListModels:
    def test_exits_zero(self):
        result = run_cli("list-models")
        assert result.returncode == 0

    def test_shows_random_forest(self):
        result = run_cli("list-models")
        assert "random_forest" in result.stdout

    def test_shows_logistic_regression(self):
        result = run_cli("list-models")
        assert "logistic_regression" in result.stdout

    def test_shows_tabpfn(self):
        result = run_cli("list-models")
        assert "tabpfn" in result.stdout.lower()

    def test_shows_install_status(self):
        """Output must indicate installed/not-installed for each model."""
        result = run_cli("list-models")
        combined = result.stdout.lower()
        # At least one of these status indicators must appear
        assert any(tok in combined for tok in ["installed", "✅", "❌", "not installed", "available"])

    def test_all_registry_keys_listed(self):
        """Every key from MODEL_REGISTRY must appear in the output."""
        from src.models import MODEL_REGISTRY
        result = run_cli("list-models")
        for key in MODEL_REGISTRY:
            assert key in result.stdout, f"MODEL_REGISTRY key {key!r} missing from list-models output"


# ---------------------------------------------------------------------------
# list-datasets
# ---------------------------------------------------------------------------

class TestCLIListDatasets:
    def test_exits_zero(self):
        result = run_cli("list-datasets")
        assert result.returncode == 0

    def test_shows_german_credit(self):
        result = run_cli("list-datasets")
        assert "german_credit" in result.stdout

    def test_shows_synthetic(self):
        result = run_cli("list-datasets")
        assert "synthetic" in result.stdout

    def test_shows_taiwan_credit(self):
        result = run_cli("list-datasets")
        assert "taiwan_credit" in result.stdout


# ---------------------------------------------------------------------------
# run subcommand
# ---------------------------------------------------------------------------

class TestCLIRun:
    def test_run_bundled_dataset_exits_zero(self):
        result = run_cli(
            "run", "--data", "german_credit",
            "--models", "random_forest", "logistic_regression",
        )
        assert result.returncode == 0, (
            f"CLI run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_run_prints_model_names(self):
        result = run_cli(
            "run", "--data", "german_credit",
            "--models", "random_forest",
        )
        combined = result.stdout + result.stderr
        # model_name column shows wrapper display name e.g. "RandomForest-Default"
        assert "randomforest" in combined.lower().replace(" ", "").replace("-", "") or \
               "model_name" in combined.lower()

    def test_run_prints_auc(self):
        result = run_cli(
            "run", "--data", "german_credit",
            "--models", "random_forest",
        )
        assert "auc" in result.stdout.lower()

    def test_run_saves_csv(self, tmp_path):
        result = run_cli(
            "run", "--data", "german_credit",
            "--models", "random_forest",
            "--output", str(tmp_path),
        )
        assert result.returncode == 0
        csv_files = list(tmp_path.glob("*.csv"))
        assert csv_files, f"No CSV saved in {tmp_path}. stdout:\n{result.stdout}"

    def test_run_invalid_model_exits_nonzero(self):
        result = run_cli(
            "run", "--data", "german_credit",
            "--models", "does_not_exist",
        )
        assert result.returncode != 0

    def test_run_missing_data_arg_exits_nonzero(self):
        result = run_cli("run", "--models", "random_forest")
        assert result.returncode != 0
