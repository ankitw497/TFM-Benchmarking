"""
src.data.preprocessor — BasicPreprocessor.

Provides a sklearn-compatible transformer that handles the most common
tabular preprocessing steps:

  1. Median imputation for numeric columns
  2. Mode imputation for categorical columns
  3. StandardScaler for numeric columns (zero mean, unit variance)
  4. OneHotEncoder for categorical columns (handle_unknown="ignore")

The preprocessor is fit on training data only; transform() applies the
learned statistics to any split without re-fitting (no data leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class BasicPreprocessor:
    """Fit-once, transform-many preprocessing for mixed-dtype tabular data.

    Follows the sklearn fit / transform / fit_transform convention.

    Steps applied per column type
    ------------------------------
    Numeric columns
        1. Median imputation  (``SimpleImputer(strategy="median")``)
        2. Standard scaling   (``StandardScaler()``)

    Categorical columns  (object / string dtype)
        1. Mode imputation    (``SimpleImputer(strategy="most_frequent")``)
        2. One-hot encoding   (``OneHotEncoder(handle_unknown="ignore", sparse_output=False)``)

    Parameters
    ----------
    None — all hyper-parameters use sklearn defaults for now.

    Examples
    --------
    >>> pp = BasicPreprocessor()
    >>> X_train_pp = pp.fit_transform(X_train)
    >>> X_test_pp  = pp.transform(X_test)
    """

    def __init__(self) -> None:
        self._ct: ColumnTransformer | None = None
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._output_columns: list[str] | None = None

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "BasicPreprocessor":
        """Learn imputation statistics, scaler params, and OHE categories from X.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.  Must have column names.
        y : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        self
        """
        X = self._validate_input(X)

        self._numeric_cols = X.select_dtypes(include="number").columns.tolist()
        self._categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = []
        if self._numeric_cols:
            transformers.append(("num", numeric_pipeline, self._numeric_cols))
        if self._categorical_cols:
            transformers.append(("cat", categorical_pipeline, self._categorical_cols))

        self._ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )
        self._ct.fit(X)

        # Pre-compute output column names so they are deterministic
        self._output_columns = self._build_column_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformations to X.

        Parameters
        ----------
        X : pd.DataFrame
            Features to transform.  May have missing values and/or
            unseen categorical levels (those rows get all-zero OHE columns).

        Returns
        -------
        pd.DataFrame
            Dense DataFrame with no missing values.  Column names are strings.
        """
        if self._ct is None:
            raise RuntimeError("BasicPreprocessor must be fit before calling transform().")

        X = self._validate_input(X)
        arr = self._ct.transform(X)
        return pd.DataFrame(arr, columns=self._output_columns, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit then transform X in a single call.

        Equivalent to ``self.fit(X).transform(X)`` but slightly more
        efficient because the internal sklearn ColumnTransformer already
        caches the fitted array after fitting.
        """
        self.fit(X, y)
        return self.transform(X)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"BasicPreprocessor expects a pandas DataFrame, got {type(X)!r}."
            )
        return X

    def _build_column_names(self) -> list[str]:
        """Return deterministic string column names for the transformed output."""
        names: list[str] = []

        # Numeric columns keep their original names (same order as self._numeric_cols)
        names.extend(self._numeric_cols)

        # Categorical columns: OHE creates one column per category per feature.
        if self._categorical_cols and self._ct is not None:
            # Find the 'cat' transformer
            for name, transformer, _ in self._ct.transformers_:
                if name == "cat":
                    ohe: OneHotEncoder = transformer.named_steps["ohe"]
                    for feature_idx, feature_name in enumerate(self._categorical_cols):
                        for category in ohe.categories_[feature_idx]:
                            names.append(f"{feature_name}_{category}")
                    break

        return [str(c) for c in names]
