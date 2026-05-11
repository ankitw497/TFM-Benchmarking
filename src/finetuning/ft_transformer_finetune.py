"""
Fine-tuning infrastructure for FT-Transformer.
Proper PyTorch training loop with AdamW optimizer, cosine LR scheduler,
and early stopping based on validation loss.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .trainer import BaseFineTuner, FineTuningConfig, FineTuningResult


class FTTransformerFineTuner(BaseFineTuner):
    """
    Fine-tunes an FT-Transformer model (from FTTransformerModel.fit()).

    Provides:
    - AdamW + cosine annealing LR scheduler
    - Early stopping with patience
    - Per-epoch train/val loss curves
    - Checkpoint save/resume

    Usage:
        # First fit zero-shot model
        model = FTTransformerModel()
        model.fit(X_train, y_train)

        # Fine-tune further
        config = FineTuningConfig(epochs=50, lr=1e-5)
        tuner = FTTransformerFineTuner()
        result = tuner.fine_tune(model, X_train, y_train, config)
        # model._model now holds fine-tuned weights
        proba = model.predict_proba(X_test)
    """

    def __init__(self, batch_size: int = 256):
        self.batch_size = batch_size

    def fine_tune(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: FineTuningConfig,
    ) -> FineTuningResult:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("torch required. Install with: pip install torch>=2.1")

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        if model is None or not hasattr(model, "_model") or model._model is None:
            raise ValueError(
                "FTTransformerFineTuner requires a fitted FTTransformerModel. "
                "Call model.fit() first."
            )

        result = FineTuningResult()
        device = model._device
        net = model._model
        n_num = len(model._num_cols)

        # ── Validation split ───────────────────────────────────────────────
        y_np = y_train.values.astype(np.int64)
        if config.val_fraction > 0 and len(X_train) > 20:
            X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
                X_train, y_train,
                test_size=config.val_fraction,
                stratify=y_train,
                random_state=config.random_state,
            )
        else:
            X_tr = X_val = X_train
            y_tr_raw = y_val_raw = y_train

        X_num_tr, X_cat_tr = model._preprocess_transform(X_tr)
        X_num_val, X_cat_val = model._preprocess_transform(X_val)
        y_tr_np = y_tr_raw.values.astype(np.int64)
        y_val_np = y_val_raw.values.astype(np.int64)

        # ── Optimizer + scheduler ─────────────────────────────────────────
        optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        rng = np.random.RandomState(config.random_state)
        n_tr = len(y_tr_np)

        for epoch in range(config.epochs):
            # ── Training epoch ─────────────────────────────────────────────
            net.train()
            perm = rng.permutation(n_tr)
            epoch_loss, n_batches = 0.0, 0
            for i in range(0, n_tr, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xn = self._arr_to_tensor(X_num_tr, idx, torch.float32, device)
                xc = self._arr_to_tensor(X_cat_tr, idx, torch.long, device)
                yt = torch.tensor(y_tr_np[idx], dtype=torch.long, device=device)

                optimizer.zero_grad()
                out = net(xn if n_num > 0 else None, xc)
                loss = loss_fn(out, yt)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            result.train_loss_curve.append(epoch_loss / max(n_batches, 1))

            # ── Validation ─────────────────────────────────────────────────
            net.eval()
            with torch.no_grad():
                xn_v = self._arr_to_tensor(X_num_val, None, torch.float32, device)
                xc_v = self._arr_to_tensor(X_cat_val, None, torch.long, device)
                yv = torch.tensor(y_val_np, dtype=torch.long, device=device)
                out_v = net(xn_v if n_num > 0 else None, xc_v)
                val_loss = loss_fn(out_v, yv).item()
            result.val_loss_curve.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                result.best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                if config.checkpoint_dir:
                    ckpt = self._save_checkpoint(
                        best_state, Path(config.checkpoint_dir),
                        "ft_transformer", epoch + 1,
                    )
                    result.checkpoint_path = str(ckpt)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    result.converged = True
                    break

        # Restore best weights
        if best_state is not None:
            net.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Final validation AUC
        net.eval()
        with torch.no_grad():
            xn_v = self._arr_to_tensor(X_num_val, None, torch.float32, device)
            xc_v = self._arr_to_tensor(X_cat_val, None, torch.long, device)
            out_v = net(xn_v if n_num > 0 else None, xc_v)
            proba = F.softmax(out_v, dim=-1).cpu().numpy()[:, 1]
        result.final_val_auc = roc_auc_score(y_val_np, proba)
        result.total_epochs_run = len(result.train_loss_curve)

        return result

    @staticmethod
    def _arr_to_tensor(arr, idx, dtype, device):
        """Slice and convert a numpy array to a device tensor (or None)."""
        import torch
        if arr is None:
            return None
        sub = arr[idx] if idx is not None else arr
        return torch.tensor(sub, dtype=dtype, device=device)
