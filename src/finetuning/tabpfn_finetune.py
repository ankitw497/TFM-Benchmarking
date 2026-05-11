"""
Fine-tuning infrastructure for TabPFN v2 and v2.5.
Provides checkpoint save/resume, OOM recovery, and validation AUC tracking.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .trainer import BaseFineTuner, FineTuningConfig, FineTuningResult


class TabPFNFineTuner(BaseFineTuner):
    """
    Fine-tunes TabPFN v2/v2.5 with:
    - Checkpoint save & resume (avoids repeating long runs)
    - OOM recovery (halves batch size on GPU out-of-memory)
    - Validation AUC tracking

    Usage:
        config = FineTuningConfig(epochs=30, lr=1e-5,
                                  checkpoint_dir=Path("checkpoints"))
        tuner = TabPFNFineTuner(version="v2")
        result = tuner.fine_tune(None, X_train, y_train, config)
        model  = tuner.get_finetuned_model()
        proba  = model.predict_proba(X_test)
    """

    def __init__(self, version: str = "v2", device: str = "auto"):
        self.version = version
        self.device = device
        self._finetuned_model = None

    def fine_tune(
        self,
        model,  # Unused — TabPFN loads its own weights internally
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: FineTuningConfig,
    ) -> FineTuningResult:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        result = FineTuningResult()

        try:
            from tabpfn.finetuning import FinetunedTabPFNClassifier
        except ImportError:
            raise ImportError(
                "tabpfn>=2.0 required for fine-tuning. "
                "Install with: pip install tabpfn>=2.0"
            )

        device = self.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # ── Check for an existing checkpoint ───────────────────────────────
        if config.resume_from_checkpoint and config.checkpoint_dir:
            prefix = f"tabpfn_{self.version}"
            ckpt = self._find_latest_checkpoint(Path(config.checkpoint_dir), prefix)
            if ckpt:
                print(f"  Resuming TabPFN fine-tuning from checkpoint: {ckpt.name}")
                self._finetuned_model = self._load_checkpoint(ckpt)
                result.checkpoint_path = str(ckpt)
                result.converged = True
                return result

        # ── Carve off a validation set for monitoring ─────────────────────
        if config.val_fraction > 0 and len(X_train) > 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train,
                test_size=config.val_fraction,
                stratify=y_train,
                random_state=config.random_state,
            )
        else:
            X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

        # ── OOM recovery: retry with smaller batch size ────────────────────
        batch_size = config.batch_size
        for attempt in range(3):
            try:
                self._finetuned_model = FinetunedTabPFNClassifier(
                    device=device,
                    n_epochs=config.epochs,
                    learning_rate=config.lr,
                    batch_size=batch_size,
                )
                self._finetuned_model.fit(X_tr, y_tr)

                # Validation AUC
                val_proba = self._finetuned_model.predict_proba(X_val)
                if val_proba.ndim > 1:
                    val_proba = val_proba[:, 1]
                result.final_val_auc = roc_auc_score(y_val, val_proba)
                result.best_epoch = config.epochs
                result.total_epochs_run = config.epochs
                result.converged = True

                # Save checkpoint
                if config.checkpoint_dir:
                    ckpt_path = self._save_checkpoint(
                        self._finetuned_model,
                        Path(config.checkpoint_dir),
                        f"tabpfn_{self.version}",
                        config.epochs,
                    )
                    result.checkpoint_path = str(ckpt_path)

                break  # Success

            except (MemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() and attempt < 2:
                    batch_size = max(1, batch_size // 2)
                    print(f"  OOM — retrying TabPFN fine-tuning with batch_size={batch_size}")
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    raise

        return result

    def get_finetuned_model(self):
        """Return the fine-tuned classifier. Call after fine_tune()."""
        if self._finetuned_model is None:
            raise RuntimeError("Call fine_tune() before get_finetuned_model()")
        return self._finetuned_model
