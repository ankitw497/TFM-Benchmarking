"""
Abstract base class for all model wrappers.
Ensures consistent API across TFMs, deep learning models, and baselines.
"""

import time
import traceback
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, brier_score_loss,
    f1_score, classification_report
)


@dataclass
class ModelLimitations:
    """Documents hard and soft limits for each model."""
    max_rows: Optional[int] = None          # Hard limit on training rows
    max_features: Optional[int] = None      # Hard limit on features
    recommended_max_rows: Optional[int] = None   # Soft limit (degrades above)
    recommended_max_features: Optional[int] = None
    supports_missing: bool = True
    supports_categorical: bool = True
    supports_regression: bool = False
    requires_gpu: bool = False
    supports_finetuning: bool = False
    license: str = "Unknown"
    commercial_use: bool = False
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Stores all results from a single model evaluation."""
    model_name: str
    dataset_name: str
    phase: str  # "zero_shot", "finetuned", "scaling"

    # Metrics
    auc_roc: float = 0.0
    accuracy: float = 0.0
    log_loss_val: float = 0.0
    brier_score: float = 0.0
    f1_macro: float = 0.0
    ece: float = 0.0  # Expected Calibration Error

    # Timing
    fit_time: float = 0.0
    predict_time: float = 0.0
    total_time: float = 0.0

    # Resources
    peak_memory_mb: float = 0.0

    # Dataset info
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0

    # Status
    success: bool = True
    error_message: str = ""
    warnings: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class BaseModelWrapper(ABC):
    """
    Abstract base class. Every model wrapper must implement:
      - fit(X_train, y_train)
      - predict_proba(X_test) → np.ndarray of shape (n, 2)
      - get_limitations() → ModelLimitations
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._is_fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model (or store context for ICL models)."""
        pass

    @abstractmethod
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities, shape (n_samples, n_classes)."""
        pass

    @abstractmethod
    def get_limitations(self) -> ModelLimitations:
        """Return model limitations and metadata."""
        pass

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return class predictions."""
        proba = self.predict_proba(X_test)
        return (proba[:, 1] >= 0.5).astype(int)

    def can_handle_dataset(self, n_rows: int, n_features: int) -> bool:
        """Check if dataset is within model's limits."""
        lim = self.get_limitations()
        if lim.max_rows and n_rows > lim.max_rows:
            return False
        if lim.max_features and n_features > lim.max_features:
            return False
        return True

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        dataset_name: str = "unknown",
        phase: str = "zero_shot",
    ) -> BenchmarkResult:
        """
        Full evaluation pipeline: fit → predict → compute metrics.
        Handles errors gracefully and tracks timing.
        """
        result = BenchmarkResult(
            model_name=self.name,
            dataset_name=dataset_name,
            phase=phase,
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=X_train.shape[1],
        )

        # Check limits
        if not self.can_handle_dataset(len(X_train), X_train.shape[1]):
            lim = self.get_limitations()
            result.success = False
            result.error_message = (
                f"Dataset exceeds limits: {len(X_train)} rows "
                f"(max {lim.max_rows}), {X_train.shape[1]} features "
                f"(max {lim.max_features})"
            )
            return result

        try:
            # Fit
            t0 = time.perf_counter()
            self.fit(X_train, y_train)
            result.fit_time = time.perf_counter() - t0

            # Predict
            t0 = time.perf_counter()
            y_proba = self.predict_proba(X_test)
            result.predict_time = time.perf_counter() - t0
            result.total_time = result.fit_time + result.predict_time

            # Handle different proba shapes
            if y_proba.ndim == 1:
                y_score = y_proba
            else:
                y_score = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba[:, 1]

            y_pred = (y_score >= 0.5).astype(int)

            # Compute metrics
            result.auc_roc = roc_auc_score(y_test, y_score)
            result.accuracy = accuracy_score(y_test, y_pred)
            result.log_loss_val = log_loss(y_test, y_score, labels=[0, 1])
            result.brier_score = brier_score_loss(y_test, y_score)
            result.f1_macro = f1_score(y_test, y_pred, average="macro")
            result.ece = _compute_ece(y_test.values, y_score, n_bins=15)

            # Track memory
            result.peak_memory_mb = _get_peak_memory()

        except Exception as e:
            result.success = False
            result.error_message = f"{type(e).__name__}: {str(e)}"
            result.warnings.append(traceback.format_exc())

        return result


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)


def _get_peak_memory() -> float:
    """Get peak memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0
