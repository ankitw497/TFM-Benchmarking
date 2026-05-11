"""
Wrapper for FT-Transformer (Feature Tokenizer + Transformer).
Reference: "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021).
Install: pip install rtdl  OR  pip install git+https://github.com/yandex-research/rtdl.git
License: MIT (fully commercial)
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from .base import BaseModelWrapper, ModelLimitations


class FTTransformerModel(BaseModelWrapper):
    """
    FT-Transformer: Feature Tokenizer + Transformer for tabular data.

    Competitive with GBDTs on medium-to-large datasets when tuned.
    Trains from scratch per dataset with early stopping.
    Handles missing values (median imputation) and categoricals (embeddings).

    Install: pip install rtdl
    """

    def __init__(
        self,
        d_token: int = 64,
        n_blocks: int = 3,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        random_state: int = 42,
        device: str = "auto",
        **kwargs,
    ):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device
        self._model = None
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._cat_encoders = {}
        self._cat_cardinalities: List[int] = []
        self._num_imputer = None
        self._num_scaler = None
        self._device = None
        super().__init__(name="FT-Transformer", **kwargs)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            import rtdl
        except ImportError:
            raise ImportError(
                "rtdl not installed. Install with:\n"
                "  pip install rtdl\n"
                "or: pip install git+https://github.com/yandex-research/rtdl.git"
            )
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        device_str = self.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)

        # Identify column types
        self._num_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()
        self._cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        # Fit preprocessors on training data only
        X_num, X_cat = self._preprocess_fit(X_train)

        # Split validation set for early stopping
        y_np = y_train.values.astype(np.int64)
        indices = np.arange(len(y_np))
        if self.val_fraction > 0 and len(y_np) > 20:
            idx_tr, idx_val = train_test_split(
                indices,
                test_size=self.val_fraction,
                stratify=y_np,
                random_state=self.random_state,
            )
        else:
            idx_tr = idx_val = indices

        X_num_tr = X_num[idx_tr] if X_num is not None else None
        X_num_val = X_num[idx_val] if X_num is not None else None
        X_cat_tr = X_cat[idx_tr] if X_cat is not None else None
        X_cat_val = X_cat[idx_val] if X_cat is not None else None
        y_tr, y_val = y_np[idx_tr], y_np[idx_val]

        n_num = len(self._num_cols)

        # Build model
        try:
            self._model = rtdl.FTTransformer.make_default(
                n_num_features=n_num if n_num > 0 else None,
                cat_cardinalities=self._cat_cardinalities if self._cat_cardinalities else None,
                last_layer_query_idx=[-1],
                d_out=2,
                d_token=self.d_token,
                n_blocks=self.n_blocks,
                attention_dropout=self.attention_dropout,
                ffn_dropout=self.ffn_dropout,
                residual_dropout=self.residual_dropout,
            ).to(self._device)
        except TypeError:
            # Fallback for different rtdl API versions
            self._model = rtdl.FTTransformer.make_default(
                n_num_features=n_num if n_num > 0 else None,
                cat_cardinalities=self._cat_cardinalities if self._cat_cardinalities else None,
                d_out=2,
            ).to(self._device)

        optimizer = self._model.make_default_optimizer()
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        n_tr = len(y_tr)

        for epoch in range(self.n_epochs):
            self._model.train()
            perm = rng.permutation(n_tr)
            for i in range(0, n_tr, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xn = self._to_tensor(X_num_tr[idx] if X_num_tr is not None else None,
                                     torch.float32, len(idx), n_num)
                xc = self._cat_to_tensor(X_cat_tr[idx] if X_cat_tr is not None else None)
                yt = torch.tensor(y_tr[idx], dtype=torch.long, device=self._device)

                optimizer.zero_grad()
                out = self._model(xn if n_num > 0 else None, xc)
                loss = loss_fn(out, yt)
                loss.backward()
                optimizer.step()

            # Validation pass
            self._model.eval()
            with torch.no_grad():
                xn_v = self._to_tensor(X_num_val, torch.float32, len(y_val), n_num)
                xc_v = self._cat_to_tensor(X_cat_val)
                yv = torch.tensor(y_val, dtype=torch.long, device=self._device)
                out_v = self._model(xn_v if n_num > 0 else None, xc_v)
                val_loss = loss_fn(out_v, yv).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})

        self._is_fitted = True

    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        assert self._is_fitted, "Must call fit() first"
        import torch
        import torch.nn.functional as F

        X_num, X_cat = self._preprocess_transform(X_test)
        n_num = len(self._num_cols)
        all_proba = []

        self._model.eval()
        with torch.no_grad():
            for i in range(0, len(X_test), self.batch_size):
                end = min(i + self.batch_size, len(X_test))
                xn = self._to_tensor(
                    X_num[i:end] if X_num is not None else None,
                    torch.float32, end - i, n_num,
                )
                xc = self._cat_to_tensor(X_cat[i:end] if X_cat is not None else None)
                out = self._model(xn if n_num > 0 else None, xc)
                proba = F.softmax(out, dim=-1).cpu().numpy()
                all_proba.append(proba)

        return np.vstack(all_proba)

    def get_limitations(self) -> ModelLimitations:
        return ModelLimitations(
            max_rows=None,
            max_features=None,
            recommended_max_rows=100_000,
            recommended_max_features=1_000,
            supports_missing=True,
            supports_categorical=True,
            supports_regression=True,
            requires_gpu=False,
            supports_finetuning=True,
            license="MIT",
            commercial_use=True,
            notes=(
                "Feature Tokenizer + Transformer (NeurIPS 2021). "
                "Competitive with GBDTs on medium-large datasets. "
                "Slower without GPU. Trains from scratch per dataset. "
                "Install: pip install rtdl"
            ),
        )

    # -----------------------------------------------------------------------
    # Preprocessing helpers (all stats fitted on X_train only)
    # -----------------------------------------------------------------------

    def _preprocess_fit(self, X_train: pd.DataFrame):
        """Fit preprocessors and transform training data. Call once in fit()."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        X_num = None
        X_cat = None

        if self._num_cols:
            X_raw = X_train[self._num_cols].values.astype(np.float32)
            self._num_imputer = SimpleImputer(strategy="median")
            self._num_scaler = StandardScaler()
            X_num = self._num_scaler.fit_transform(
                self._num_imputer.fit_transform(X_raw)
            ).astype(np.float32)

        if self._cat_cols:
            self._cat_encoders = {}
            self._cat_cardinalities = []
            cols = []
            for col in self._cat_cols:
                le = LabelEncoder()
                vals = X_train[col].fillna("__MISSING__").astype(str)
                le.fit(vals)
                self._cat_encoders[col] = le
                self._cat_cardinalities.append(len(le.classes_))
                cols.append(le.transform(vals))
            X_cat = np.column_stack(cols).astype(np.int64)

        return X_num, X_cat

    def _preprocess_transform(self, X: pd.DataFrame):
        """Transform data using already-fitted preprocessors."""
        X_num = None
        X_cat = None

        if self._num_cols:
            X_raw = X[self._num_cols].values.astype(np.float32)
            X_num = self._num_scaler.transform(
                self._num_imputer.transform(X_raw)
            ).astype(np.float32)

        if self._cat_cols:
            cols = []
            for col in self._cat_cols:
                le = self._cat_encoders[col]
                known = set(le.classes_)
                vals = X[col].fillna("__MISSING__").astype(str)
                vals = vals.map(lambda v: v if v in known else le.classes_[0])
                cols.append(le.transform(vals))
            X_cat = np.column_stack(cols).astype(np.int64)

        return X_num, X_cat

    def _to_tensor(self, arr, dtype, n, n_cols):
        """Convert numpy array (or None) to a device tensor."""
        import torch
        if arr is None or n_cols == 0:
            return None
        return torch.tensor(arr, dtype=dtype, device=self._device)

    def _cat_to_tensor(self, arr):
        """Convert categorical array (or None) to a device tensor."""
        import torch
        if arr is None or not self._cat_cols:
            return None
        return torch.tensor(arr, dtype=torch.long, device=self._device)
