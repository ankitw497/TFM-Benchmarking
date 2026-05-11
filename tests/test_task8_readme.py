"""
Task 8 tests — README.md content and runnable quick-start example.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
README = REPO_ROOT / "README.md"


def readme_text() -> str:
    return README.read_text()


# ---------------------------------------------------------------------------
# Structure checks
# ---------------------------------------------------------------------------

class TestReadmeStructure:
    def test_has_quick_start_section(self):
        assert "Quick Start" in readme_text()

    def test_quick_start_before_experimental_design(self):
        text = readme_text()
        qs_pos = text.find("Quick Start")
        ed_pos = text.find("Experimental Design")
        assert qs_pos < ed_pos, "Quick Start section must appear before Experimental Design"

    def test_has_installation_section(self):
        assert "Installation" in readme_text()

    def test_installation_covers_base(self):
        text = readme_text()
        assert 'pip install -e "."' in text or "pip install tfm-benchmark" in text

    def test_installation_covers_gbdt_extra(self):
        assert "[gbdt]" in readme_text()

    def test_installation_mentions_no_gpu(self):
        text = readme_text().lower()
        assert "no gpu" in text or "no-gpu" in text or "cpu" in text

    def test_uses_new_api_not_src_imports(self):
        """Quick-start example must use `from tfm_benchmark import`, not `from src.`."""
        text = readme_text()
        # Find the Quick Start code block
        qs_start = text.find("Quick Start")
        qs_end = text.find("\n---", qs_start)
        qs_section = text[qs_start:qs_end] if qs_end > qs_start else text[qs_start:]
        assert "from tfm_benchmark" in qs_section, \
            "Quick Start must use `from tfm_benchmark import ...`"
        # Old src.X imports should not appear in the quick start block
        assert "from src." not in qs_section, \
            "Quick Start must not use `from src.X import ...`"

    def test_existing_models_table_preserved(self):
        """Detailed models table must still be present."""
        assert "TabPFN" in readme_text()
        assert "XGBoost" in readme_text()

    def test_existing_references_preserved(self):
        assert "References" in readme_text()

    def test_has_version_badge_or_install_badge(self):
        text = readme_text()
        assert "python" in text.lower()


# ---------------------------------------------------------------------------
# Runnable example
# ---------------------------------------------------------------------------

class TestReadmeExample:
    def test_five_line_example_runs(self):
        """Extract the 5-line quick-start snippet from README and execute it."""
        text = readme_text()

        # Find a python code block that contains `run_benchmark` or `load_dataset`
        pattern = r"```python\n(.*?)```"
        blocks = re.findall(pattern, text, re.DOTALL)

        api_blocks = [b for b in blocks if "tfm_benchmark" in b and
                      ("run_benchmark" in b or "load_dataset" in b)]
        assert api_blocks, "No tfm_benchmark code block found in README"

        snippet = api_blocks[0]
        # Strip lines that reference placeholder values like YOUR_USERNAME
        runnable_lines = [
            line for line in snippet.splitlines()
            if not line.strip().startswith("#") or "import" in line
        ]
        code = "\n".join(runnable_lines)

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"README 5-line example failed:\nCode:\n{code}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
