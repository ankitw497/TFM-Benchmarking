"""
Wrapper for TabICL v1.1 and TabICLv2.
Fully open-source (BSD-3-Clause), scales to 1M+ rows.
"""

import numpy as np
import pandas as pd
from .base import BaseModelWrapper, ModelLimitations


class TabICLModel(BaseModelWrapper):
    """
    Wrapper for TabICL / TabICLv2.

    Usage:
        model = TabICLModel(version="v2")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        version: str = "v2",
        n_estimators: int = 8,
        device: str = "auto",
        **kwargs,
    ):
        self.version = version
        self.n_estimators = n_estimators
        self.device = device
        self._model = None

        name = f"TabICL-{version}"
        super().__init__(name=name, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        if self.version == "v2":
            from tabicl import TabICLClassifier
            self._model = TabICLClassifier(
                n_estimators=self.n_estimators,
                version="v2",
            )
        elif self.version == "v1.1":
            from tabicl import TabICLClassifier
            self._model = TabICLClassifier(
                n_estimators=self.n_estimators,
                version="v1.1",
            )
        else:
            raise ValueError(f"Unknown TabICL version: {self.version}")

        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        proba = self._model.predict_proba(X_test)
        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    def get_limitations(self) -> ModelLimitations:
        if self.version == "v2":
            return ModelLimitations(
                max_rows=None,  # Can handle 1M+ with offloading
                max_features=2_000,
                recommended_max_rows=100_000,
                recommended_max_features=2_000,
                supports_missing=True,
                supports_categorical=True,
                supports_regression=True,
                requires_gpu=False,  # Works on CPU, GPU recommended
                supports_finetuning=False,
                license="BSD-3-Clause",
                commercial_use=True,
                notes=(
                    "Fully open-source. SOTA on TabArena/TALENT. "
                    "Pre-trained on datasets up to 48K rows but generalizes to 600K+. "
                    "Scales to 1M rows with disk offloading (50GB GPU, 24GB CPU). "
                    "10x faster than TabPFN-2.5. No fine-tuning needed or supported."
                ),
            )
        else:  # v1.1
            return ModelLimitations(
                max_rows=None,
                max_features=500,
                recommended_max_rows=50_000,
                recommended_max_features=500,
                supports_missing=True,
                supports_categorical=True,
                supports_regression=False,  # v1.1 is classification only
                requires_gpu=False,
                supports_finetuning=False,
                license="BSD-3-Clause",
                commercial_use=True,
                notes="TabICL v1 post-trained on v2 prior. Classification only.",
            )
