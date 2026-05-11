"""
Proper sklearn wrappers for RandomForest and LogisticRegression.
These serve as traditional ML baselines.
License: BSD-3-Clause (scikit-learn)

Key design: preprocessors are fitted ONLY on training data (no leakage).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from .base import BaseModelWrapper, ModelLimitations


class RandomForestWrapper(BaseModelWrapper):
    """
    RandomForest with proper preprocessing pipeline.
    Handles missing values (median imputation) and categoricals (ordinal encoding).
    Preprocessor is fitted on training data only — no leakage.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth=None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None
        self._preprocessor = None
        super().__init__(name="RandomForest-Default", **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_prepared, self._preprocessor = _build_and_fit_preprocessor(X_train, scale=False)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_prepared, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        X_prepared = self._preprocessor.transform(X_test)
        return self._model.predict_proba(X_prepared)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None,
            max_features=None,
            supports_missing=True,
            supports_categorical=True,
            supports_regression=True,
            requires_gpu=False,
            supports_finetuning=False,
            license="BSD-3-Clause (scikit-learn)",
            commercial_use=True,
            notes=(
                "Ensemble of decision trees. Strong non-parametric baseline. "
                "Handles missing values via median imputation and categoricals via ordinal encoding. "
                "No row/feature limits."
            ),
        )


class LogisticRegressionWrapper(BaseModelWrapper):
    """
    Logistic Regression with proper preprocessing pipeline.
    Median imputation + StandardScaler + ordinal encoding for categoricals.
    Preprocessor fitted on training data only — no leakage.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None
        self._preprocessor = None
        super().__init__(name="LogisticRegression", **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_prepared, self._preprocessor = _build_and_fit_preprocessor(X_train, scale=True)
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._model.fit(X_prepared, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        X_prepared = self._preprocessor.transform(X_test)
        return self._model.predict_proba(X_prepared)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None,
            max_features=None,
            supports_missing=True,
            supports_categorical=True,
            supports_regression=True,
            requires_gpu=False,
            supports_finetuning=False,
            license="BSD-3-Clause (scikit-learn)",
            commercial_use=True,
            notes=(
                "Linear model with L2 regularization. Well-calibrated probabilities. "
                "Uses median imputation + StandardScaler + ordinal encoding. "
                "Good interpretable baseline for credit scoring."
            ),
        )


# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------

def _build_and_fit_preprocessor(
    X_train: pd.DataFrame,
    scale: bool = False,
) -> tuple:
    """
    Build and fit a ColumnTransformer preprocessor on training data only.

    Returns (X_transformed_array, fitted_preprocessor).
    Apply to test data via preprocessor.transform(X_test) — no leakage.
    """
    num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(num_steps)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    if not transformers:
        # All-numeric, no categoricals — just passthrough with imputation
        transformers.append(("num", num_pipeline, X_train.columns.tolist()))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_prepared = preprocessor.fit_transform(X_train)
    return X_prepared, preprocessor
