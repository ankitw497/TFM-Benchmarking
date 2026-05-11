"""
Wrapper for TabNet — Google's attention-based tabular model.
Supports both supervised training and self-supervised pre-training + fine-tuning.
License: Apache 2.0 (fully commercial)

Bug fixed: imputer/encoder statistics are fitted on X_train only (no leakage).
"""

import numpy as np
import pandas as pd
from .base import BaseModelWrapper, ModelLimitations


class TabNetModel(BaseModelWrapper):
    """
    TabNet with optional self-supervised pretraining.

    Usage:
        model = TabNetModel()           # Supervised only
        model = TabNetModel(pretrain=True)  # Self-supervised + supervised
    """

    def __init__(
        self,
        pretrain: bool = False,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        max_epochs: int = 200,
        patience: int = 20,
        pretrain_epochs: int = 100,
        pretrain_ratio: float = 0.8,
        random_state: int = 42,
        device: str = "auto",
        **kwargs,
    ):
        self.pretrain = pretrain
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_ratio = pretrain_ratio
        self.random_state = random_state
        self.device = device
        self._model = None
        self._pretrain_model = None

        # Preprocessing state — fitted on X_train only (no leakage)
        self._col_medians: dict = {}
        self._cat_codes: dict = {}

        name = "TabNet-Pretrained" if pretrain else "TabNet"
        super().__init__(name=name, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        from pytorch_tabnet.tab_model import TabNetClassifier

        # Fit preprocessing statistics on training data
        self._fit_preprocessor(X_train)
        X_np = self._transform(X_train)
        y_np = y_train.values.astype(int)

        device = self.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self._model = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            seed=self.random_state, device_name=device, verbose=0,
        )

        fit_kwargs = {
            "X_train": X_np, "y_train": y_np,
            "max_epochs": self.max_epochs, "patience": self.patience,
        }

        if self.pretrain:
            from pytorch_tabnet.pretraining import TabNetPretrainer
            self._pretrain_model = TabNetPretrainer(
                n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
                seed=self.random_state, device_name=device, verbose=0,
            )
            self._pretrain_model.fit(
                X_train=X_np,
                max_epochs=self.pretrain_epochs,
                pretraining_ratio=self.pretrain_ratio,
            )
            fit_kwargs["from_unsupervised"] = self._pretrain_model

        self._model.fit(**fit_kwargs)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        X_np = self._transform(X_test)
        return self._model.predict_proba(X_np)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None,
            max_features=None,
            supports_missing=False,  # Requires imputation (handled internally)
            supports_categorical=False,  # Requires encoding (handled internally)
            supports_regression=True,
            requires_gpu=False,
            supports_finetuning=True,
            license="Apache 2.0",
            commercial_use=True,
            notes=(
                "Google's attention-based model with built-in interpretability. "
                "Requires imputation and categorical encoding — both handled internally. "
                "Self-supervised pretraining can improve results with limited labels. "
                "Generally underperforms tuned GBDTs but provides feature importance masks."
            ),
        )

    # -----------------------------------------------------------------------
    # Preprocessing — stats fitted on X_train, applied to X_test (no leakage)
    # -----------------------------------------------------------------------

    def _fit_preprocessor(self, X_train: pd.DataFrame) -> None:
        """Compute and store imputation/encoding stats from training data."""
        self._col_medians = {}
        self._cat_codes = {}

        for col in X_train.columns:
            if X_train[col].dtype in ["object", "category"]:
                # Store category → int mapping derived from training data
                cats = X_train[col].fillna("__MISSING__").astype("category")
                self._cat_codes[col] = dict(enumerate(cats.cat.categories))
                # Invert: category → code
                self._cat_codes[col] = {v: k for k, v in self._cat_codes[col].items()}
            elif X_train[col].isna().any():
                self._col_medians[col] = X_train[col].median()

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted preprocessing statistics to any split."""
        X = X.copy()

        for col in X.columns:
            if col in self._cat_codes:
                X[col] = (
                    X[col]
                    .fillna("__MISSING__")
                    .astype(str)
                    .map(lambda v: self._cat_codes[col].get(v, -1))
                )
            elif X[col].isna().any():
                fill = self._col_medians.get(col, 0.0)
                X[col] = X[col].fillna(fill)

        # Encode any remaining object/category columns
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes

        return X.values.astype(np.float32)
