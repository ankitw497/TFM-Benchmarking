"""
Wrapper for Mitra (Amazon/AutoGluon) — Apache 2.0 licensed.
"""

import numpy as np
import pandas as pd
from .base import BaseModelWrapper, ModelLimitations


class MitraModel(BaseModelWrapper):
    """
    Mitra via AutoGluon integration.

    Usage:
        model = MitraModel()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
    """

    def __init__(self, device: str = "auto", **kwargs):
        self.device = device
        self._predictor = None
        self._label_col = "__target__"
        super().__init__(name="Mitra", **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from autogluon.tabular import TabularPredictor

        train_data = X_train.copy()
        train_data[self._label_col] = y_train.values

        self._predictor = TabularPredictor(
            label=self._label_col,
            eval_metric="roc_auc",
            verbosity=0,
        ).fit(
            train_data,
            hyperparameters={"MITRA": {}},
            num_cpus=4,
            num_gpus=1 if self.device != "cpu" else 0,
        )
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        proba = self._predictor.predict_proba(X_test)
        if isinstance(proba, pd.DataFrame):
            return proba.values
        return proba

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None,
            max_features=None,
            recommended_max_rows=10_000,
            recommended_max_features=100,
            supports_missing=True,
            supports_categorical=True,
            supports_regression=True,
            requires_gpu=True,
            supports_finetuning=False,
            license="Apache 2.0",
            commercial_use=True,
            notes=(
                "Amazon's TFM with curated synthetic prior mixture. "
                "Native AutoGluon integration. Best on small datasets (<10K). "
                "ICL context limited to ~10K samples and ~100 features."
            ),
        )
