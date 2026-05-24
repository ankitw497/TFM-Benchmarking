"""
v2 Task 4 tests — auto mode skip warnings and RuntimeError when zero models available.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def splits():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=150, n_features=6, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y_s = pd.Series(y, name="target")
    return X_df[:120], y_s[:120], X_df[120:], y_s[120:]


def _make_failing_registry(original_registry, fail_keys):
    """Return a copy of the registry where fail_keys raise ImportError."""
    def _fail_factory(key):
        def _inner():
            raise ImportError(f"mock: {key} not installed")
        return _inner

    patched = dict(original_registry)
    for k in fail_keys:
        patched[k] = _fail_factory(k)
    return patched


# ---------------------------------------------------------------------------
# warnings.warn emitted when models are skipped in auto mode
# ---------------------------------------------------------------------------

class TestAutoModeWarnings:
    def test_import_error_triggers_warn(self):
        """When auto mode skips models, warnings.warn must be called."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        # Build a registry where only random_forest succeeds
        fail_keys = [k for k in MODEL_REGISTRY if k != "random_forest"]
        patched_registry = _make_failing_registry(MODEL_REGISTRY, fail_keys)

        with patch.object(bm_mod, "MODEL_REGISTRY", patched_registry):
            b = bm_mod.Benchmarker(models="auto", verbose=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                b._resolve_wrappers()

        assert len(caught) >= 1, "Expected at least one warning when models are skipped"

    def test_warn_message_contains_skipped_key(self):
        """Warning message should mention the skipped model key."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        patched_registry = _make_failing_registry(MODEL_REGISTRY, ["xgboost"])

        with patch.object(bm_mod, "MODEL_REGISTRY", patched_registry):
            b = bm_mod.Benchmarker(models="auto", verbose=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                b._resolve_wrappers()

        warning_messages = " ".join(str(w.message) for w in caught)
        assert "xgboost" in warning_messages.lower(), (
            f"Expected 'xgboost' in warning message, got: {warning_messages!r}"
        )

    def test_warn_emitted_when_verbose_false(self):
        """warnings.warn must fire even when verbose=False (not just print)."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        fail_keys = [k for k in MODEL_REGISTRY if k != "random_forest"]
        patched_registry = _make_failing_registry(MODEL_REGISTRY, fail_keys)

        with patch.object(bm_mod, "MODEL_REGISTRY", patched_registry):
            b = bm_mod.Benchmarker(models="auto", verbose=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                b._resolve_wrappers()

        assert len(caught) >= 1, (
            "warnings.warn should fire even when verbose=False"
        )

    def test_no_warn_when_explicit_model_list(self):
        """Explicit model list: no UserWarning/RuntimeWarning issued."""
        import tfm_benchmark.benchmarker as bm_mod

        b = bm_mod.Benchmarker(models=["random_forest"], verbose=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b._resolve_wrappers()

        our_warns = [
            w for w in caught
            if issubclass(w.category, (UserWarning, RuntimeWarning))
        ]
        assert len(our_warns) == 0, (
            f"No warnings expected for explicit model list, got: {our_warns}"
        )

    def test_warn_is_user_warning(self):
        """The emitted warning should be a UserWarning."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        fail_keys = [k for k in MODEL_REGISTRY if k != "random_forest"]
        patched_registry = _make_failing_registry(MODEL_REGISTRY, fail_keys)

        with patch.object(bm_mod, "MODEL_REGISTRY", patched_registry):
            b = bm_mod.Benchmarker(models="auto", verbose=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                b._resolve_wrappers()

        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warns) >= 1, "Expected at least one UserWarning"


# ---------------------------------------------------------------------------
# RuntimeError when zero models can be instantiated in auto mode
# ---------------------------------------------------------------------------

class TestRuntimeErrorOnZeroModels:
    def test_raises_runtime_error_when_all_skip(self):
        """auto mode must raise RuntimeError if no models can be instantiated."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        # Make ALL registry entries fail
        all_failing = _make_failing_registry(MODEL_REGISTRY, list(MODEL_REGISTRY.keys()))

        with patch.object(bm_mod, "MODEL_REGISTRY", all_failing):
            b = bm_mod.Benchmarker(models="auto", verbose=False)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError, match="No models"):
                    b._resolve_wrappers()

    def test_runtime_error_message_is_helpful(self):
        """RuntimeError message should tell the user what to do."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        all_failing = _make_failing_registry(MODEL_REGISTRY, list(MODEL_REGISTRY.keys()))

        with patch.object(bm_mod, "MODEL_REGISTRY", all_failing):
            b = bm_mod.Benchmarker(models="auto", verbose=False)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError) as exc_info:
                    b._resolve_wrappers()

        msg = str(exc_info.value).lower()
        assert "model" in msg or "install" in msg, (
            f"RuntimeError message should be helpful, got: {exc_info.value!r}"
        )

    def test_fit_evaluate_propagates_runtime_error(self, splits):
        """fit_evaluate() should propagate the RuntimeError from _resolve_wrappers."""
        import tfm_benchmark.benchmarker as bm_mod
        from src.models import MODEL_REGISTRY

        all_failing = _make_failing_registry(MODEL_REGISTRY, list(MODEL_REGISTRY.keys()))

        with patch.object(bm_mod, "MODEL_REGISTRY", all_failing):
            b = bm_mod.Benchmarker(models="auto", verbose=False)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(RuntimeError):
                    b.fit_evaluate(*splits)

    def test_no_runtime_error_when_explicit_list(self, splits):
        """Explicit model list: no RuntimeError (handled differently)."""
        from tfm_benchmark import Benchmarker
        b = Benchmarker(models=["random_forest"], verbose=False)
        results = b.fit_evaluate(*splits)
        assert len(results) == 1
