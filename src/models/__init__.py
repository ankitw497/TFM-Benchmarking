from .base import BaseModelWrapper, BenchmarkResult, ModelLimitations

# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# Maps a stable string key to a zero-arg factory function that returns
# a BaseModelWrapper instance ready for fit/evaluate.
# Wrappers for optional dependencies use lambda so the import is deferred
# until the factory is actually called — the package stays importable even
# when tabpfn, xgboost, etc. are not installed.
# ---------------------------------------------------------------------------

def _make_tabpfn(version: str, device: str = "auto"):
    from .tabpfn_wrapper import TabPFNModel
    return TabPFNModel(version=version, device=device)


def _make_tabicl(version: str):
    from .tabicl_wrapper import TabICLModel
    return TabICLModel(version=version)


def _make_tabdpt():
    from .tabdpt_wrapper import TabDPTModel
    return TabDPTModel()


def _make_mitra():
    from .mitra_wrapper import MitraModel
    return MitraModel()


def _make_tabnet(pretrain: bool = False):
    from .tabnet_wrapper import TabNetModel
    return TabNetModel(pretrain=pretrain)


def _make_ft_transformer():
    from .ft_transformer_wrapper import FTTransformerModel
    return FTTransformerModel()


def _make_xgboost(tuned: bool = False):
    from .gbdt_wrapper import XGBoostModel
    return XGBoostModel(tuned=tuned)


def _make_catboost(tuned: bool = False):
    from .gbdt_wrapper import CatBoostModel
    return CatBoostModel(tuned=tuned)


def _make_lightgbm():
    from .gbdt_wrapper import LightGBMModel
    return LightGBMModel()


def _make_random_forest():
    from .sklearn_wrapper import RandomForestWrapper
    return RandomForestWrapper()


def _make_logistic_regression():
    from .sklearn_wrapper import LogisticRegressionWrapper
    return LogisticRegressionWrapper()


MODEL_REGISTRY: dict = {
    # ── Tabular Foundation Models ─────────────────────────────────────────
    "tabpfn_v1":          lambda: _make_tabpfn("v1", device="cpu"),
    "tabpfn_v2":          lambda: _make_tabpfn("v2"),
    "tabpfn_v2_5":        lambda: _make_tabpfn("v2.5"),
    "tabpfn_v2_5_real":   lambda: _make_tabpfn("v2.5-real"),
    "tabicl_v2":          lambda: _make_tabicl("v2"),
    "tabicl_v1_1":        lambda: _make_tabicl("v1.1"),
    "mitra":              lambda: _make_mitra(),
    "tabdpt":             lambda: _make_tabdpt(),
    # ── Deep Learning ─────────────────────────────────────────────────────
    "tabnet":             lambda: _make_tabnet(pretrain=False),
    "ft_transformer":     lambda: _make_ft_transformer(),
    # ── Gradient Boosted Trees ────────────────────────────────────────────
    "xgboost":            lambda: _make_xgboost(tuned=False),
    "xgboost_tuned":      lambda: _make_xgboost(tuned=True),
    "catboost":           lambda: _make_catboost(tuned=False),
    "catboost_tuned":     lambda: _make_catboost(tuned=True),
    "lightgbm":           lambda: _make_lightgbm(),
    # ── Traditional ML ────────────────────────────────────────────────────
    "random_forest":      lambda: _make_random_forest(),
    "logistic_regression": lambda: _make_logistic_regression(),
}

__all__ = [
    "BaseModelWrapper",
    "BenchmarkResult",
    "ModelLimitations",
    "MODEL_REGISTRY",
]
