"""
Base classes and data structures for fine-tuning tabular foundation models.
All fine-tuners inherit from BaseFineTuner and use FineTuningConfig.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any

import numpy as np
import pandas as pd


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning any supported model."""
    epochs: int = 30
    lr: float = 1e-5
    batch_size: int = 20
    patience: int = 10          # Early stopping: stop after this many epochs without improvement
    val_fraction: float = 0.15  # Fraction of training data to hold out for validation
    checkpoint_dir: Optional[Path] = None
    resume_from_checkpoint: bool = True
    random_state: int = 42


@dataclass
class FineTuningResult:
    """Results from a single fine-tuning run."""
    best_epoch: int = 0
    train_loss_curve: List[float] = field(default_factory=list)
    val_loss_curve: List[float] = field(default_factory=list)
    final_val_auc: float = 0.0
    checkpoint_path: Optional[str] = None
    converged: bool = False
    total_epochs_run: int = 0

    def to_dict(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "final_val_auc": self.final_val_auc,
            "total_epochs_run": self.total_epochs_run,
            "converged": self.converged,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
        }


class BaseFineTuner(ABC):
    """Abstract base class for all model fine-tuners."""

    @abstractmethod
    def fine_tune(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: FineTuningConfig,
    ) -> FineTuningResult:
        """
        Fine-tune a model on the given dataset.

        Parameters
        ----------
        model : model instance (usage depends on subclass — may be unused if
                the tuner constructs the model internally)
        X_train : training features
        y_train : training labels
        config : fine-tuning hyperparameters

        Returns
        -------
        FineTuningResult with loss curves and best checkpoint path
        """
        pass

    # -----------------------------------------------------------------------
    # Shared checkpoint utilities
    # -----------------------------------------------------------------------

    def _find_latest_checkpoint(self, checkpoint_dir: Path, prefix: str) -> Optional[Path]:
        """Find the most recent checkpoint file matching prefix."""
        if not checkpoint_dir or not checkpoint_dir.exists():
            return None
        checkpoints = sorted(checkpoint_dir.glob(f"{prefix}_epoch_*.pkl"))
        return checkpoints[-1] if checkpoints else None

    def _save_checkpoint(
        self, obj: Any, checkpoint_dir: Path, prefix: str, epoch: int
    ) -> Path:
        """Pickle-serialize obj to checkpoint_dir/prefix_epoch_NNNN.pkl."""
        import pickle  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"{prefix}_epoch_{epoch:04d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
        return path

    def _load_checkpoint(self, path: Path) -> Any:
        """Load a pickled checkpoint. Only load files written by _save_checkpoint."""
        import pickle  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
        with open(path, "rb") as f:
            return pickle.load(f)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
