"""
src.data.preprocessor — BasicPreprocessor stub.

Provides a sklearn-compatible transformer for minimal preprocessing.
Currently a pass-through; extend in a future sprint with imputation,
scaling, and encoding steps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BasicPreprocessor:
    """Minimal sklearn-compatible preprocessor (pass-through stub).

    Follows the sklearn fit/transform/fit_transform convention so it can
    be dropped into any sklearn Pipeline.  The current implementation
    returns data unchanged; override in subclasses or extend here when
    real preprocessing is needed.
    """

    def fit(self, X, y=None) -> "BasicPreprocessor":
        self._feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
