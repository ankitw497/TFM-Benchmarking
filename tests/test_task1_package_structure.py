"""
Task 1 tests — package discovery and skeleton.
These tests MUST FAIL before implementation and PASS after.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPackageImports:
    def test_tfm_benchmark_importable(self):
        """Top-level package must be importable after install."""
        import tfm_benchmark  # noqa: F401

    def test_version_attribute(self):
        """Package must expose __version__ string."""
        import tfm_benchmark
        assert hasattr(tfm_benchmark, "__version__")
        assert isinstance(tfm_benchmark.__version__, str)
        assert tfm_benchmark.__version__ == "0.1.0"

    def test_src_imports_still_work(self):
        """Existing src.* imports must not regress."""
        from src.data.loader import load_credit_dataset  # noqa: F401
        from src.models.base import BaseModelWrapper, BenchmarkResult  # noqa: F401
        from src.evaluation.metrics import compute_all_metrics  # noqa: F401


class TestCLI:
    def test_cli_module_importable(self):
        """CLI module must be importable."""
        from tfm_benchmark import cli  # noqa: F401

    def test_cli_main_callable(self):
        """CLI main() must be a callable."""
        from tfm_benchmark.cli import main
        assert callable(main)

    def test_cli_list_models_no_crash(self, capsys):
        """list-models subcommand must run without raising."""
        import sys
        from tfm_benchmark.cli import main
        sys.argv = ["tfm-benchmark", "list-models"]
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
        output = capsys.readouterr().out
        assert len(output) > 0

    def test_cli_list_datasets_no_crash(self, capsys):
        """list-datasets subcommand must run without raising."""
        import sys
        from tfm_benchmark.cli import main
        sys.argv = ["tfm-benchmark", "list-datasets"]
        try:
            main()
        except SystemExit as e:
            assert e.code == 0
        output = capsys.readouterr().out
        assert len(output) > 0
