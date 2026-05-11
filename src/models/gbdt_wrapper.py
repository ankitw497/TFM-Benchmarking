"""
Gradient Boosted Decision Tree baselines: XGBoost, CatBoost, LightGBM.
These serve as the 'gold standard' to beat.
"""

import numpy as np
import pandas as pd
from .base import BaseModelWrapper, ModelLimitations


class XGBoostModel(BaseModelWrapper):
    """XGBoost with default or tuned hyperparameters."""

    def __init__(self, tuned: bool = False, random_state: int = 42, **kwargs):
        self.tuned = tuned
        self.random_state = random_state
        self._model = None
        name = "XGBoost-Tuned" if tuned else "XGBoost-Default"
        super().__init__(name=name, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from xgboost import XGBClassifier

        if self.tuned:
            # Best-practice defaults for credit scoring
            params = dict(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, gamma=0.1,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                eval_metric="auc", random_state=self.random_state,
                tree_method="hist", enable_categorical=True,
            )
        else:
            params = dict(
                n_estimators=100, random_state=self.random_state,
                eval_metric="auc", tree_method="hist", enable_categorical=True,
            )

        self._model = XGBClassifier(**params)
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted
        return self._model.predict_proba(X_test)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None, max_features=None,
            supports_missing=True, supports_categorical=True,
            supports_regression=True, requires_gpu=False,
            supports_finetuning=False,
            license="Apache 2.0", commercial_use=True,
            notes="Industry-standard GBDT. No row/feature limits.",
        )


class CatBoostModel(BaseModelWrapper):
    """CatBoost with default or tuned hyperparameters."""

    def __init__(self, tuned: bool = False, random_state: int = 42, **kwargs):
        self.tuned = tuned
        self.random_state = random_state
        self._model = None
        name = "CatBoost-Tuned" if tuned else "CatBoost-Default"
        super().__init__(name=name, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from catboost import CatBoostClassifier

        cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        if self.tuned:
            params = dict(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3, auto_class_weights="Balanced",
                random_seed=self.random_state, verbose=0,
                cat_features=cat_features if cat_features else None,
            )
        else:
            params = dict(
                iterations=100, random_seed=self.random_state,
                verbose=0,
                cat_features=cat_features if cat_features else None,
            )

        self._model = CatBoostClassifier(**params)
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted
        return self._model.predict_proba(X_test)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None, max_features=None,
            supports_missing=True, supports_categorical=True,
            supports_regression=True, requires_gpu=False,
            supports_finetuning=False,
            license="Apache 2.0", commercial_use=True,
            notes="Best native categorical support among GBDTs.",
        )


class LightGBMModel(BaseModelWrapper):
    """LightGBM with default hyperparameters."""

    def __init__(self, random_state: int = 42, **kwargs):
        self.random_state = random_state
        self._model = None
        super().__init__(name="LightGBM-Default", **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from lightgbm import LGBMClassifier

        self._model = LGBMClassifier(
            n_estimators=100, random_state=self.random_state, verbose=-1,
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted
        return self._model.predict_proba(X_test)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None, max_features=None,
            supports_missing=True, supports_categorical=True,
            supports_regression=True, requires_gpu=False,
            supports_finetuning=False,
            license="MIT", commercial_use=True,
            notes="Fastest GBDT. Histogram-based splitting.",
        )
