"""
Comprehensive evaluation metrics for tabular model benchmarking.
Includes classification metrics, calibration analysis, and statistical tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, accuracy_score, log_loss, brier_score_loss,
    f1_score, precision_score, recall_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    threshold : float
        Decision threshold for class predictions.

    Returns
    -------
    dict
        Dictionary of metric_name -> value.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # Discrimination metrics
    metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    metrics["auc_pr"] = average_precision_score(y_true, y_prob)

    # Probability quality
    metrics["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])
    metrics["brier_score"] = brier_score_loss(y_true, y_prob)

    # Calibration
    metrics["ece"] = compute_ece(y_true, y_prob, n_bins=15)
    metrics["mce"] = compute_mce(y_true, y_prob, n_bins=15)

    # Classification at threshold
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_binary"] = f1_score(y_true, y_pred, average="binary")
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / n if n > 0 else 0.0


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Maximum Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))

    return mce


def get_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get calibration curve data for plotting reliability diagrams.

    Returns (fraction_of_positives, mean_predicted_value)
    """
    return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "youden",
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold.

    Parameters
    ----------
    method : str
        'youden' (maximizes sensitivity + specificity - 1)
        'f1' (maximizes F1 score)

    Returns
    -------
    threshold : float
    metrics_at_threshold : dict
    """
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal = thresholds[best_idx]
    elif method == "f1":
        thresholds = np.arange(0.01, 1.0, 0.01)
        f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
        optimal = thresholds[np.argmax(f1s)]
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = compute_all_metrics(y_true, y_prob, threshold=optimal)
    return optimal, metrics


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired model comparison.

    Parameters
    ----------
    scores_a, scores_b : arrays of per-fold scores for model A and B
    alternative : 'two-sided', 'greater', 'less'

    Returns
    -------
    statistic, p_value
    """
    from scipy.stats import wilcoxon

    # Remove ties
    diff = np.array(scores_a) - np.array(scores_b)
    if np.all(diff == 0):
        return 0.0, 1.0

    stat, pval = wilcoxon(scores_a, scores_b, alternative=alternative)
    return stat, pval


def friedman_test(score_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Friedman test for comparing multiple models across multiple datasets.

    Parameters
    ----------
    score_matrix : array of shape (n_datasets, n_models)

    Returns
    -------
    statistic, p_value
    """
    from scipy.stats import friedmanchisquare

    if score_matrix.shape[1] < 3:
        raise ValueError("Friedman test requires at least 3 models")

    columns = [score_matrix[:, i] for i in range(score_matrix.shape[1])]
    stat, pval = friedmanchisquare(*columns)
    return stat, pval


def compute_ranks(score_matrix: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    Compute average ranks across datasets for each model.

    Parameters
    ----------
    score_matrix : array of shape (n_datasets, n_models)
    higher_is_better : if True, rank 1 = highest score

    Returns
    -------
    average_ranks : array of shape (n_models,)
    """
    from scipy.stats import rankdata

    n_datasets, n_models = score_matrix.shape
    ranks = np.zeros_like(score_matrix, dtype=float)

    for i in range(n_datasets):
        if higher_is_better:
            ranks[i] = rankdata(-score_matrix[i])
        else:
            ranks[i] = rankdata(score_matrix[i])

    return ranks.mean(axis=0)


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def aggregate_cv_results(
    fold_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate cross-validation fold results into mean ± std.

    Parameters
    ----------
    fold_results : list of metric dicts from each fold

    Returns
    -------
    dict of metric_name -> {"mean": ..., "std": ..., "values": [...]}
    """
    if not fold_results:
        return {}

    all_metrics = fold_results[0].keys()
    aggregated = {}

    for metric in all_metrics:
        values = [r[metric] for r in fold_results if isinstance(r.get(metric), (int, float))]
        if values:
            aggregated[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

    return aggregated


def create_comparison_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    sort_by: str = "auc_roc",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Create a formatted comparison table from model results.

    Parameters
    ----------
    results : dict of model_name -> metric_dict
    metrics : list of metric names to include
    sort_by : metric to sort by
    ascending : sort direction

    Returns
    -------
    DataFrame with one row per model
    """
    if metrics is None:
        metrics = ["auc_roc", "log_loss", "brier_score", "f1_macro", "ece"]

    rows = []
    for model_name, metric_dict in results.items():
        row = {"model": model_name}
        for m in metrics:
            if m in metric_dict:
                val = metric_dict[m]
                if isinstance(val, dict):
                    row[m] = f"{val['mean']:.4f} ± {val['std']:.4f}"
                    row[f"{m}_mean"] = val["mean"]
                else:
                    row[m] = f"{val:.4f}"
                    row[f"{m}_mean"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    if f"{sort_by}_mean" in df.columns:
        df = df.sort_values(f"{sort_by}_mean", ascending=ascending)

    # Clean up helper columns
    drop_cols = [c for c in df.columns if c.endswith("_mean")]
    display_df = df.drop(columns=drop_cols, errors="ignore")

    return display_df.reset_index(drop=True)
