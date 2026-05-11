"""
Wrapper for TabDPT — Scaling Tabular Foundation Models on Real Data.
License: MIT (fully commercial)
"""

import numpy as np
import pandas as pd
from .base import BaseModelWrapper, ModelLimitations


class TabDPTModel(BaseModelWrapper):
    """
    TabDPT: combines ICL with self-supervised learning on real datasets.

    NOTE: TabDPT must be installed from GitHub:
        pip install git+https://github.com/layer6ai-labs/TabDPT.git

    Usage:
        model = TabDPTModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
    """

    def __init__(self, device: str = "auto", **kwargs):
        self.device = device
        self._model = None
        super().__init__(name="TabDPT", **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            from tabdpt import TabDPTClassifier
        except ImportError:
            raise ImportError(
                "TabDPT not installed. Install with:\n"
                "  pip install git+https://github.com/layer6ai-labs/TabDPT.git"
            )

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = TabDPTClassifier(device=device)
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted
        proba = self._model.predict_proba(X_test)
        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None,
            max_features=500,
            recommended_max_rows=10_000,
            recommended_max_features=500,
            supports_missing=True,
            supports_categorical=True,
            supports_regression=True,
            requires_gpu=True,
            supports_finetuning=False,
            license="MIT",
            commercial_use=True,
            notes=(
                "Trained on 123 real OpenML datasets with ICL + SSL. "
                "Demonstrates power-law scaling like LLMs. "
                "Uses k-NN retrieval at inference time. "
                "Best on small-to-medium datasets."
            ),
        )
