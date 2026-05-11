"""
Wrapper for TabPFN v1, v2, v2.5, and Real-TabPFN-2.5.
Handles version-specific APIs, limitations, and fine-tuning.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .base import BaseModelWrapper, ModelLimitations


# Version-specific configurations
TABPFN_VERSIONS = {
    "v1": {
        "max_rows": 1_000,
        "max_features": 100,
        "recommended_max_rows": 1_000,
        "recommended_max_features": 100,
        "supports_categorical": False,
        "supports_missing": False,
        "supports_regression": False,
        "supports_finetuning": False,
        "license": "Apache 2.0",
        "commercial_use": True,
        "notes": "Original model (ICLR 2023). Numerical features only.",
    },
    "v2": {
        "max_rows": 10_000,
        "max_features": 500,
        "recommended_max_rows": 10_000,
        "recommended_max_features": 500,
        "supports_categorical": True,
        "supports_missing": True,
        "supports_regression": True,
        "supports_finetuning": True,
        "license": "Prior Labs License (commercial with attribution)",
        "commercial_use": True,
        "notes": "Nature 2025 model. Commercial OK with 'Built with PriorLabs-TabPFN' attribution.",
    },
    "v2.5": {
        "max_rows": 50_000,
        "max_features": 2_000,
        "recommended_max_rows": 50_000,
        "recommended_max_features": 2_000,
        "supports_categorical": True,
        "supports_missing": True,
        "supports_regression": True,
        "supports_finetuning": True,
        "license": "TabPFN-2.5 License v1.0 (NON-COMMERCIAL)",
        "commercial_use": False,
        "notes": (
            "NON-COMMERCIAL license. Research/eval only. "
            "Cannot be used for production, business decisions, or revenue. "
            "Contact sales@priorlabs.ai for commercial license. "
            "Requires HuggingFace login and license acceptance."
        ),
    },
    "v2.5-real": {
        "max_rows": 50_000,
        "max_features": 2_000,
        "recommended_max_rows": 50_000,
        "recommended_max_features": 2_000,
        "supports_categorical": True,
        "supports_missing": True,
        "supports_regression": True,
        "supports_finetuning": True,
        "license": "TabPFN-2.5 License v1.0 (NON-COMMERCIAL)",
        "commercial_use": False,
        "notes": (
            "Real-TabPFN-2.5: TabPFN-2.5 fine-tuned on 43 real-world datasets. "
            "Same license restrictions as v2.5. This is the default v2.5 classifier."
        ),
    },
}


class TabPFNModel(BaseModelWrapper):
    """
    Unified wrapper for all TabPFN versions.

    Usage:
        model = TabPFNModel(version="v2")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        version: str = "v2",
        device: str = "auto",
        n_estimators: int = 8,
        **kwargs,
    ):
        assert version in TABPFN_VERSIONS, f"Unknown version: {version}"
        self.version = version
        self.device = device
        self.n_estimators = n_estimators
        self._model = None

        name = f"TabPFN-{version}"
        super().__init__(name=name, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Initialize and fit the TabPFN model."""
        if self.version == "v1":
            self._fit_v1(X_train, y_train)
        else:
            self._fit_v2_plus(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Get probability predictions."""
        assert self._is_fitted, "Must call fit() first"
        proba = self._model.predict_proba(X_test)
        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    def get_limitations(self) -> ModelLimitations:
        config = TABPFN_VERSIONS[self.version]
        return ModelLimitations(
            max_rows=config["max_rows"],
            max_features=config["max_features"],
            recommended_max_rows=config["recommended_max_rows"],
            recommended_max_features=config["recommended_max_features"],
            supports_missing=config["supports_missing"],
            supports_categorical=config["supports_categorical"],
            supports_regression=config["supports_regression"],
            requires_gpu=(self.version in ["v2.5", "v2.5-real"]),
            supports_finetuning=config["supports_finetuning"],
            license=config["license"],
            commercial_use=config["commercial_use"],
            notes=config["notes"],
        )

    def _fit_v1(self, X_train, y_train):
        """TabPFN v1: original ICLR 2023 model."""
        from tabpfn import TabPFNClassifier
        self._model = TabPFNClassifier(
            device=self.device if self.device != "auto" else "cpu",
            N_ensemble_configurations=min(self.n_estimators, 32),
        )
        self._model.fit(X_train.values if hasattr(X_train, "values") else X_train, y_train)

    def _fit_v2_plus(self, X_train, y_train):
        """TabPFN v2, v2.5, Real-TabPFN-2.5."""
        from tabpfn import TabPFNClassifier

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        init_kwargs = {"device": device, "n_estimators": self.n_estimators}

        # Version-specific model loading
        if self.version == "v2":
            from tabpfn.constants import ModelVersion
            self._model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, **init_kwargs
            )
        elif self.version == "v2.5":
            # v2.5 synthetic-only (not fine-tuned on real data)
            self._model = TabPFNClassifier(
                model_path="tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
                **init_kwargs,
            )
        elif self.version == "v2.5-real":
            # Real-TabPFN-2.5 (default v2.5 classifier, fine-tuned on real data)
            self._model = TabPFNClassifier(**init_kwargs)
        else:
            self._model = TabPFNClassifier(**init_kwargs)

        self._model.fit(X_train, y_train)


class FinetunedTabPFNModel(BaseModelWrapper):
    """
    Fine-tuned TabPFN wrapper.

    Performs gradient-based fine-tuning of TabPFN v2/v2.5 weights
    on a target dataset.

    Usage:
        model = FinetunedTabPFNModel(version="v2", epochs=30, lr=1e-5)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
    """

    def __init__(
        self,
        version: str = "v2",
        epochs: int = 30,
        lr: float = 1e-5,
        batch_size: int = 20,
        device: str = "auto",
        **kwargs,
    ):
        self.version = version
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self._model = None

        name = f"TabPFN-{version}-finetuned"
        super().__init__(name=name, **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fine-tune TabPFN on the target dataset."""
        from tabpfn.finetuning import FinetunedTabPFNClassifier

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = FinetunedTabPFNClassifier(
            device=device,
            n_epochs=self.epochs,
            learning_rate=self.lr,
            batch_size=self.batch_size,
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        proba = self._model.predict_proba(X_test)
        if proba.ndim == 1:
            return np.column_stack([1 - proba, proba])
        return proba

    def get_limitations(self) -> ModelLimitations:
        config = TABPFN_VERSIONS[self.version]
        return ModelLimitations(
            max_rows=config["max_rows"],
            max_features=config["max_features"],
            recommended_max_rows=config["recommended_max_rows"],
            recommended_max_features=config["recommended_max_features"],
            supports_missing=config["supports_missing"],
            supports_categorical=config["supports_categorical"],
            requires_gpu=True,
            supports_finetuning=True,
            license=config["license"],
            commercial_use=config["commercial_use"],
            notes=f"Fine-tuned version. {config['notes']}",
        )
