"""
Fine-tuning / pretraining infrastructure for TabNet.
Separates self-supervised pretraining from supervised fine-tuning.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .trainer import BaseFineTuner, FineTuningConfig, FineTuningResult


class TabNetFineTuner(BaseFineTuner):
    """
    Two-phase TabNet training pipeline:

    Phase 1 — Self-supervised pretraining (TabNetPretrainer on all training data)
    Phase 2 — Supervised fine-tuning using the pretrained encoder

    The pretrained encoder is cached as a checkpoint so it can be reused
    across datasets without re-pretraining.

    Usage:
        config = FineTuningConfig(epochs=100, checkpoint_dir=Path("checkpoints"))
        tuner = TabNetFineTuner()
        result = tuner.fine_tune(None, X_train, y_train, config)
        model  = tuner.get_finetuned_model()
        proba  = model.predict_proba(X_np)
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        device: str = "auto",
        pretrain_ratio: float = 0.8,
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.device = device
        self.pretrain_ratio = pretrain_ratio
        self._pretrained_encoder = None
        self._finetuned_model = None

    def fine_tune(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: FineTuningConfig,
    ) -> FineTuningResult:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            from pytorch_tabnet.pretraining import TabNetPretrainer
        except ImportError:
            raise ImportError(
                "pytorch-tabnet required. Install with: pip install pytorch-tabnet"
            )

        result = FineTuningResult()

        device = self.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Prepare data
        X_np = self._prepare(X_train)
        y_np = y_train.values.astype(int)

        # Split for early stopping monitoring
        if config.val_fraction > 0 and len(X_np) > 20:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_np, y_np,
                test_size=config.val_fraction,
                stratify=y_np,
                random_state=config.random_state,
            )
        else:
            X_tr, X_val, y_tr, y_val = X_np, X_np, y_np, y_np

        # ── Phase 1: Load pretrained encoder or train from scratch ─────────
        pretrained = None
        if config.resume_from_checkpoint and config.checkpoint_dir:
            ckpt = self._find_latest_checkpoint(
                Path(config.checkpoint_dir), "tabnet_pretrained"
            )
            if ckpt:
                pretrained = self._load_checkpoint(ckpt)
                print(f"  TabNet: loaded pretrained encoder from {ckpt.name}")

        if pretrained is None:
            print("  TabNet Phase 1: self-supervised pretraining...")
            pretrained = TabNetPretrainer(
                n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
                device_name=device, verbose=0,
            )
            pretrained.fit(
                X_train=X_tr,
                max_epochs=config.epochs,
                pretraining_ratio=self.pretrain_ratio,
            )
            self._pretrained_encoder = pretrained
            if config.checkpoint_dir:
                self._save_checkpoint(
                    pretrained, Path(config.checkpoint_dir),
                    "tabnet_pretrained", config.epochs,
                )

        # ── Phase 2: Supervised fine-tuning ───────────────────────────────
        print("  TabNet Phase 2: supervised fine-tuning...")
        self._finetuned_model = TabNetClassifier(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            device_name=device, verbose=0,
        )
        self._finetuned_model.fit(
            X_train=X_tr, y_train=y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=config.epochs,
            patience=config.patience,
            from_unsupervised=pretrained,
        )

        val_proba = self._finetuned_model.predict_proba(X_val)[:, 1]
        result.final_val_auc = roc_auc_score(y_val, val_proba)
        result.total_epochs_run = config.epochs
        result.converged = True

        return result

    def get_finetuned_model(self):
        if self._finetuned_model is None:
            raise RuntimeError("Call fine_tune() before get_finetuned_model()")
        return self._finetuned_model

    @staticmethod
    def _prepare(X: pd.DataFrame) -> np.ndarray:
        """Median imputation + ordinal encoding + float32 cast for TabNet."""
        X = X.copy()
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype in ["object", "category"]:
                    X[col] = X[col].fillna("__MISSING__")
                else:
                    X[col] = X[col].fillna(X[col].median())
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].astype("category").cat.codes
        return X.values.astype(np.float32)
